import asyncio
import base64
import hashlib
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1]/'data/runtime/ultralytics'))
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .ai import Pipeline, decode, engine, jpeg
from .face_flow import FaceFlow, decode_tracking_frame
from .config import DATA, ROOT, MODEL_ID
from .sessions import Broker, VideoJob, camera_matches, camera_timestamp, event_with_evidence
from .store import Store
from .calibration import fit_profiles, apply_profiles
from .media import normalize_video
from .diagnostics import Diagnostic, SCENARIOS, live_coverage
from .video_evaluation import (read_truth, save_truth, evaluate as evaluate_video,
                               annotation_faces, save_face_truth, read_face_truth, reuse_face_truth)
from .diagnostic_analysis import ANALYSIS_VERSION, analyze_saved, next_pending, read_analysis

store = Store()
broker = Broker()
jobs = {}
runtime = None
active_live = None
session_lock = asyncio.Lock()
model_loading = False
reindex_errors = []
diagnostics = {}


@asynccontextmanager
async def lifespan(app):
    global runtime, model_loading
    runtime = tempfile.TemporaryDirectory(prefix='facewatch-', dir=DATA/'runtime')
    broker.loop = asyncio.get_running_loop()
    model_loading = True
    async def warmup():
        global model_loading, reindex_errors
        try:
            await asyncio.to_thread(engine.load)
            reindex_errors = await asyncio.to_thread(store.reindex, engine)
        except Exception:
            pass  # Exposed by /health; CRUD remains accessible for existing records.
        finally:
            model_loading = False
    task = asyncio.create_task(warmup())
    async def diagnostic_watchdog():
        while True:
            await asyncio.sleep(.1)
            for run in list(diagnostics.values()):
                now = time.monotonic()
                ended = run.state == 'running' and now-run.started >= run.duration
                disconnected = ((run.state in ('preparing','running') and run.last_frame is not None and now-run.last_frame > 8)
                                or (run.state == 'opening' and now-run.created > 30))
                if ended or disconnected:
                    if active_live and active_live['id'] == run.session_id:
                        active_live['stopped'] = True
                    run.finish('complete' if ended else 'error')
                    await broker.publish(dict(type='diagnostic_status', **run.view()))
    watchdog = asyncio.create_task(diagnostic_watchdog())
    async def analyze_completed_runs():
        while True:
            run_id = await asyncio.to_thread(next_pending, store)
            if run_id:
                try:
                    result = await asyncio.to_thread(analyze_saved, store, run_id)
                    if result:
                        await broker.publish(dict(type='diagnostic_analysis', id=run_id, analysis=result))
                except Exception as exc:
                    # Keep an explicit error for this run; do not retry in a tight loop.
                    with store.db() as db:
                        db.execute('INSERT OR REPLACE INTO diagnostic_analysis(run_id,version,result) VALUES (?,?,?)',
                            (run_id, ANALYSIS_VERSION, json.dumps(dict(error=str(exc), title='Không phân tích được số đo; cần kiểm tra lỗi.'))))
            else:
                await asyncio.sleep(.5)
    analyzer = asyncio.create_task(analyze_completed_runs())
    yield
    watchdog.cancel()
    analyzer.cancel()
    await asyncio.gather(watchdog, analyzer, return_exceptions=True)
    for run in diagnostics.values():
        run.finish('cancelled')
    for job in list(jobs.values()):
        await asyncio.to_thread(job.close)
    jobs.clear()
    await task
    if not Path(runtime.name).resolve().is_relative_to((DATA/'runtime').resolve()):
        raise RuntimeError('Invalid runtime cleanup path')
    runtime.cleanup()


app = FastAPI(title='FaceWatch AI', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver', '[::1]'])


@app.middleware('http')
async def local_only(request: Request, call_next):
    origin = request.headers.get('origin')
    if origin and origin not in ('http://localhost:5173', 'http://127.0.0.1:5173',
                                  'http://localhost:8000', 'http://127.0.0.1:8000'):
        return JSONResponse({'detail': 'Chỉ chấp nhận ứng dụng localhost.'}, status_code=403)
    if int(request.headers.get('content-length', '0')) > 4*1024**3:
        return JSONResponse({'detail': 'Upload tối đa 4 GB.'}, status_code=413)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def fail(exc, status=400):
    raise HTTPException(status, str(exc)) from exc


def get_job(job_id):
    if job_id not in jobs:
        raise HTTPException(404, 'Tác vụ không tồn tại trong phiên hiện tại.')
    return jobs[job_id]


@app.get('/api/health')
def health():
    return dict(ready=engine.ready and not model_loading, loading=model_loading, error=engine.error, device=engine.device,
                detector_provider=getattr(engine, 'detector_provider', None), model=MODEL_ID,
                models=engine.status, versions=engine.versions, reindex_errors=reindex_errors,
                gpu=engine.gpu() if engine.ready else None,
                settings=store.settings(engine.versions), live_session=active_live,
                recognition_policy=dict(priority_range_m=[0,6],max_distance_m=None,distance_used_for_rejection=False,
                    fusion='calibrated_model_gates_and_temporal_evidence',raw_cosine_averaging=False,
                    uncertain_model='abstain',qualified_conflict='unknown_and_reverify',temporal_bridge_max_seconds=.5,
                    adaptation_features=['face_pixels','native_face_height_for_profile','alignment_residual','sharpness'],
                    track_reverification='two_current_calibrated_unique_matches_after_initial_confirmation',
                    identity_continuity='tracked_label_only_with_fresh_unique_unbroken_visual_links',
                    tracking_emits_alerts=False,
                    separate_mask_pose_calibration=False),
                license='Demo học phần; weights và dataset tuân theo giấy phép của tác giả, không khẳng định quyền thương mại.')


@app.post('/api/calibration/preview')
async def calibration_preview():
    if not engine.ready or model_loading:
        raise HTTPException(503, 'Backend AI chưa sẵn sàng.')
    return await asyncio.to_thread(fit_profiles,store,dict(engine.versions))


@app.post('/api/calibration/apply')
async def calibration_apply():
    if not engine.ready or model_loading:
        raise HTTPException(503, 'Backend AI chưa sẵn sàng.')
    if (active_live and not active_live['stopped']) or any(job.state == 'analyzing' for job in jobs.values()):
        raise HTTPException(409, 'Dừng camera hoặc phân tích video trước khi áp dụng hiệu chuẩn.')
    versions = dict(engine.versions)
    result = await asyncio.to_thread(fit_profiles,store,versions)
    def commit():
        with engine.lock:
            return apply_profiles(store,result,versions)
    try:
        settings = await asyncio.to_thread(commit)
    except ValueError as exc:
        fail(exc,409)
    return dict(settings=settings, calibration=result)


class DiagnosticChoice(BaseModel):
    purpose: str = Field('validation', pattern='^(validation|calibration)$')
    eyewear: str | None = Field(None, pattern='^(unchanged|none)$')
    ready: bool = False
    scenario: str
    person_id: str | None = None
    expected_present: bool
    expert: str | None = Field(None,pattern='^(kprpe12m|maskinv_hg|aroface|lvface_eye)$')
    expert_mode: str = Field('shadow',pattern='^(shadow|calibrated)$')
    additional_experts: list[str] = Field(default_factory=list,max_length=2)


@app.get('/api/diagnostics')
def diagnostic_history():
    with store.db() as db:
        rows = [dict(r) for r in db.execute('''SELECT d.*,a.result analysis FROM diagnostics d
            LEFT JOIN diagnostic_analysis a ON a.run_id=d.id ORDER BY d.created_at DESC,d.rowid DESC LIMIT 20''')]
    for row in rows:
        row['summary'] = json.loads(row['summary']) if row['summary'] else None
        row['analysis'] = json.loads(row['analysis']) if row['analysis'] else None
        row['context'] = json.loads(row['context']) if row['context'] else None
    return dict(scenarios=SCENARIOS, runs=rows, live_cases=live_coverage(store))


@app.post('/api/diagnostics', status_code=201)
def create_diagnostic(body: DiagnosticChoice):
    if body.ready is not True:
        raise HTTPException(422,'Cần xác nhận sẵn sàng cho từng lượt kiểm thử.')
    if not engine.ready or model_loading:
        raise HTTPException(503, 'Backend AI chưa sẵn sàng.')
    if any(r.state in ('opening','preparing','running') for r in diagnostics.values()):
        raise HTTPException(409, 'Hãy kết thúc lượt kiểm thử hiện tại.')
    try:
        options=body.model_dump()
        expert=options.pop('expert')
        expert_mode=options.pop('expert_mode')
        additional_experts=options.pop('additional_experts')
        if any(key not in ('kprpe12m','lvface_eye') for key in additional_experts):
            raise ValueError('Model bổ trợ thêm chưa được hỗ trợ.')
        expert_keys=list(dict.fromkeys(([expert] if expert else [])+additional_experts))
        if expert_mode=='calibrated' and (not expert_keys or options['purpose']!='validation'):
            raise ValueError('Chế độ Fusion đối chứng cần model cụ thể và lượt validation mới.')
        if any(key in ('maskinv_hg','aroface') for key in expert_keys) and expert_mode!='shadow':
            raise ValueError('Model chuyên gia này chỉ đo đối chứng; chưa được bật xác nhận danh tính.')
        overrides={}
        from .experts import build_expert
        for expert_key in expert_keys:
            with engine.lock:
                def load_expert():
                    import torch,gc
                    try:
                        return build_expert(expert_key,ROOT/'models/candidates',engine.device,engine.models.get('lvface'))
                    except torch.cuda.OutOfMemoryError:
                        pass
                    gc.collect();torch.cuda.empty_cache()
                    return build_expert(expert_key,ROOT/'models/candidates','cpu',engine.models.get('lvface'))
                adapter=load_expert()
                if expert_mode=='calibrated':
                    allowed=[]
                    profiles=store.settings({expert_key:adapter.version})['calibration_profiles']
                    for profile in profiles:
                        validation_count=profile.get('validation_negative_observations',0)
                        stress_count=profile.get('cross_quality_negative_observations',0)
                        validation_passed=profile.get('validation_false_passes')==0
                        stress_passed=profile.get('cross_quality_false_passes')==0
                        if ((validation_count>0 or stress_count>0) and
                            (validation_count==0 or validation_passed) and
                            (stress_count==0 or stress_passed) and
                            profile.get('validation_false_passes') in (None,0) and
                            profile.get('cross_quality_false_passes') in (None,0)):
                            allowed.append(profile['band'])
                    if not allowed:
                        raise ValueError('Chưa có profile model đã qua đối chứng cho lượt Fusion mới.')
                    adapter.allowed_bands=set(allowed)
                    if expert_key=='lvface_eye':adapter.shadow_only=False
                for person in store.people():
                    for photo in person['images']:
                        if adapter.version in photo['models']:
                            continue
                        raw=Path(store.image_path(photo['id'])).read_bytes()
                        image=decode(raw);faces=engine.detect(image)
                        if len(faces)!=1:
                            raise ValueError('Ảnh đăng ký cần đúng một mặt để đối chứng model mới.')
                        box,points,_=faces[0]
                        quality=engine.quality(image,box,store.settings())
                        vector,info=adapter.embed(image,points)
                        with store.db() as db:
                            db.execute('INSERT OR IGNORE INTO model_embeddings VALUES (?,?,?,?)',
                                (photo['id'],adapter.version,vector.astype(np.float32).tobytes(),json.dumps(dict(quality,model=info))))
                overrides[expert_key]=adapter
        run = Diagnostic(store, **options)
        run.runtime_models=overrides
        run.runtime_shadow_models=[] if expert_mode=='calibrated' else None
        if expert_mode=='calibrated':run.context['calibration_role']='validation'
        if expert_keys:
            run.context['shadow_models' if expert_mode=='shadow' else 'fusion_experts']={key:adapter.version for key,adapter in overrides.items()}
            with store.db() as db:
                db.execute('UPDATE diagnostics SET context=? WHERE id=?',(json.dumps(run.context),run.id))
    except ValueError as exc:
        fail(exc)
    except (OSError,RuntimeError) as exc:
        fail(exc,503)
    for previous in diagnostics.values():
        previous.examples.clear()
    diagnostics[run.id] = run
    return run.view()


def diagnostic_run(run_id):
    if run_id not in diagnostics:
        raise HTTPException(404, 'Không có lượt kiểm thử trong phiên này.')
    return diagnostics[run_id]


@app.get('/api/diagnostics/{run_id}/examples')
def diagnostic_examples(run_id: str):
    with store.db() as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='diagnostic_inputs'").fetchone()
        rows = db.execute('SELECT metadata FROM diagnostic_inputs WHERE run_id=? ORDER BY frame_id',(run_id,)).fetchall() if exists else []
    if rows:
        return dict(volatile=False, examples=[json.loads(row['metadata']) for row in rows])
    run = diagnostic_run(run_id)
    return dict(volatile=True, examples=[{k:v for k,v in item.items() if k!='raw'}
                for item in sorted(run.examples.values(),key=lambda value:value['elapsed'])])


@app.get('/api/diagnostics/{run_id}/examples/{frame_id}')
def diagnostic_example(run_id: str, frame_id: int):
    with store.db() as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='diagnostic_inputs'").fetchone()
        row = db.execute('SELECT jpeg FROM diagnostic_inputs WHERE run_id=? AND frame_id=?',(run_id,frame_id)).fetchone() if exists else None
    if row:
        return Response(row['jpeg'],media_type='image/jpeg',headers={'Cache-Control':'no-store'})
    run = diagnostic_run(run_id)
    item = run.examples.get(frame_id)
    if item is None:
        raise HTTPException(404,'Không có ảnh replay cho frame này.')
    return Response(item['raw'],media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.post('/api/diagnostics/{run_id}/begin')
def begin_diagnostic(run_id: str):
    try:
        return diagnostic_run(run_id).begin()
    except ValueError as exc:
        run = diagnostic_run(run_id)
        run.context['last_begin_error'] = str(exc)
        with store.db() as db:
            db.execute('UPDATE diagnostics SET context=? WHERE id=?',(json.dumps(run.context),run.id))
        fail(exc, 409)


@app.post('/api/diagnostics/{run_id}/stop')
def end_diagnostic(run_id: str):
    run = diagnostic_run(run_id)
    result = run.finish('cancelled')
    if active_live and active_live['id'] == run.session_id:
        active_live['stopped'] = True
    return result


@app.get('/api/diagnostics/{run_id}')
def get_diagnostic(run_id: str):
    if run_id in diagnostics:
        result = diagnostics[run_id].view()
    else:
        with store.db() as db:
            row = db.execute('SELECT * FROM diagnostics WHERE id=?', (run_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Không tìm thấy lượt kiểm thử.')
        result = dict(id=row['id'], scenario=row['scenario'], state=row['state'], remaining=None,
                      prepare_remaining=None, summary=json.loads(row['summary']) if row['summary'] else None)
    result['analysis'] = read_analysis(store, run_id)
    return result


class DiagnosticTruth(BaseModel):
    only_target: bool
    face_visible_throughout: bool
    boxes_match_target: bool
    stage_actions: dict[str,bool] = Field(default_factory=dict)


@app.put('/api/diagnostics/{run_id}/ground-truth')
def review_diagnostic(run_id: str, body: DiagnosticTruth):
    with store.db() as db:
        run = db.execute('SELECT * FROM diagnostics WHERE id=?', (run_id,)).fetchone()
        if not run or run['state'] != 'complete' or not run['expected_present']:
            raise HTTPException(409, 'Chỉ xác nhận nhãn cho lượt đã hoàn thành có người mục tiêu.')
        previous = db.execute('SELECT details FROM diagnostic_ground_truth WHERE run_id=?',(run_id,)).fetchone()
        if previous is None or json.loads(previous[0]) != body.model_dump():
            store.labels_changed(db)
        db.execute('''INSERT INTO diagnostic_ground_truth(run_id,details) VALUES (?,?)
            ON CONFLICT(run_id) DO UPDATE SET details=excluded.details,updated_at=CURRENT_TIMESTAMP''',
            (run_id,json.dumps(body.model_dump())))
    return dict(analysis=analyze_saved(store,run_id))


@app.get('/api/persons')
def persons():
    return store.people()


async def samples(files):
    if not files:
        raise HTTPException(400, 'Cần ít nhất một ảnh hợp lệ.')
    result = []
    for file in files:
        raw = await file.read(20*1024**2+1)
        await file.close()
        if len(raw) > 20*1024**2:
            raise HTTPException(413, 'Mỗi ảnh tối đa 20 MB.')
        try:
            result.append(await asyncio.to_thread(engine.register, raw, store.settings()))
        except ValueError as exc:
            fail(ValueError(f'{file.filename}: {exc}'))
        except Exception as exc:
            fail(exc, 503)
    return result


@app.post('/api/persons', status_code=201)
async def create_person(name: str = Form(...), files: list[UploadFile] = File(...)):
    name = name.strip()
    if not 1 <= len(name) <= 120:
        raise HTTPException(400, 'Tên cần từ 1 đến 120 ký tự.')
    values = await samples(files)
    person_id = await asyncio.to_thread(store.add, name, values)
    return next(p for p in store.people() if p['id'] == person_id)


class PersonUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    selected: bool | None = None


@app.patch('/api/persons/{person_id}')
def update_person(person_id: str, body: PersonUpdate):
    if body.name is not None and not body.name.strip():
        raise HTTPException(400, 'Tên không được để trống.')
    try:
        store.update(person_id, body.name.strip() if body.name is not None else None, body.selected)
    except ValueError as exc:
        fail(exc, 404)
    return dict(ok=True)


@app.delete('/api/persons/{person_id}')
def delete_person(person_id: str):
    store.delete(person_id)
    return dict(ok=True)


@app.post('/api/persons/{person_id}/images', status_code=201)
async def add_images(person_id: str, files: list[UploadFile] = File(...)):
    if not store.exists(person_id):
        raise HTTPException(404, 'Không tìm thấy hồ sơ.')
    values = await samples(files)
    await asyncio.to_thread(store.add, '', values, person_id)
    return dict(ok=True)


@app.get('/api/images/{image_id}')
def image(image_id: str):
    path = store.image_path(image_id)
    if not path or not Path(path).is_file():
        raise HTTPException(404, 'Không tìm thấy ảnh.')
    return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.delete('/api/persons/{person_id}/images/{image_id}')
def delete_image(person_id: str, image_id: str):
    try:
        store.delete_image(person_id, image_id)
    except ValueError as exc:
        fail(exc)
    return dict(ok=True)


class Settings(BaseModel):
    similarity: float = Field(ge=.1, le=.99)
    margin: float = Field(ge=.01, le=.5)
    confirm_frames: int = Field(ge=2, le=20)


@app.put('/api/settings')
def settings(body: Settings):
    store.save_settings({**body.model_dump(), 'calibrated': False})
    return store.settings()


class CameraChoice(BaseModel):
    kind: str
    label: str = Field(min_length=1, max_length=300)


@app.post('/api/camera/validate')
def validate_camera(body: CameraChoice):
    # Browser deviceId is opaque and resolved from actual permission-granted device labels.
    if not camera_matches(body.kind, body.label):
        raise HTTPException(400, 'Thiết bị không khớp nguồn camera đã chọn.')
    return dict(ok=True, source='Camo Camera (iPhone 13)' if body.kind == 'camo' else 'Webcam laptop')


@app.post('/api/camera/start')
async def start_camera(body: CameraChoice):
    global active_live
    validate_camera(body)
    async with session_lock:
        if active_live and not active_live['stopped']:
            if active_live['attached'] or time.monotonic()-active_live['created'] < 30:
                raise HTTPException(409, 'Một phiên camera khác đang hoạt động.')
        if any(j.state == 'analyzing' for j in jobs.values()):
            raise HTTPException(409, 'Dừng phân tích video trước khi bắt đầu camera.')
        if not engine.ready:
            raise HTTPException(503, engine.error or 'Model đang khởi động.')
        active_live = dict(id=uuid.uuid4().hex, source=body.label, kind=body.kind, stopped=False,
                           created=time.monotonic(), attached=False)
        return dict(session_id=active_live['id'], source=body.label)


class CameraStop(BaseModel):
    session_id: str


@app.post('/api/camera/stop')
async def stop_camera(body: CameraStop):
    if active_live and active_live['id'] == body.session_id:
        active_live['stopped'] = True
    return dict(ok=True)


@app.post('/api/videos', status_code=201)
async def upload_video(file: UploadFile = File(...)):
    if Path(file.filename or '').suffix.lower() not in ('.mp4', '.mov', '.webm', '.m4v'):
        raise HTTPException(400, 'Hỗ trợ MP4, MOV, M4V và WebM.')
    async with session_lock:
        if active_live and not active_live['stopped']:
            raise HTTPException(409, 'Dừng camera trước khi phân tích video.')
        for job in list(jobs.values()):
            await asyncio.to_thread(job.dispose)
        jobs.clear()
        path = Path(runtime.name)/(uuid.uuid4().hex+Path(file.filename).suffix.lower())
        total = 0
        digest = hashlib.sha256()
        try:
            with path.open('wb') as out:
                while chunk := await file.read(1024**2):
                    total += len(chunk)
                    if total > 4*1024**3:
                        raise ValueError('Video tối đa 4 GB.')
                    out.write(chunk)
                    digest.update(chunk)
            normalized = path.with_name(path.stem+'-playback.mp4')
            await asyncio.to_thread(normalize_video, path, normalized)
            path.unlink(missing_ok=True)
            path = normalized
            job = await asyncio.to_thread(VideoJob, path, Path(runtime.name), engine, store, broker,
                                         digest.hexdigest(), file.filename)
            jobs[job.id] = job
            return job.status()
        except Exception as exc:
            path.unlink(missing_ok=True)
            fail(exc)
        finally:
            await file.close()


@app.get('/api/videos/{job_id}')
def job_status(job_id: str):
    return get_job(job_id).status()


@app.get('/api/video-evaluations')
def video_evaluation_history():
    with store.db() as db:
        rows = db.execute('SELECT result,updated_at FROM video_evaluations ORDER BY updated_at DESC,rowid DESC LIMIT 20').fetchall()
    return [dict(json.loads(row['result']), evaluated_at=row['updated_at']) for row in rows]


@app.get('/api/videos/{job_id}/media')
def media(job_id: str):
    return FileResponse(get_job(job_id).path)


class VideoTruth(BaseModel):
    annotations: list[dict] = Field(default_factory=list, max_length=2000)


@app.get('/api/videos/{job_id}/ground-truth')
def video_truth(job_id: str):
    job = get_job(job_id)
    annotations = read_truth(job)
    with store.db() as db:
        row = db.execute('SELECT result FROM video_evaluations WHERE run_id=?', (job.id,)).fetchone()
    previous = json.loads(row['result']) if row else None
    face_labels = read_face_truth(job)
    if previous and (previous['annotations'] != annotations or previous.get('face_labels', []) != face_labels):
        previous = None
    return dict(annotations=annotations, enrollment=job.enrollment, source_hash=job.source_hash,
                evaluation=previous, face_labels=face_labels)


@app.put('/api/videos/{job_id}/ground-truth')
def update_video_truth(job_id: str, body: VideoTruth):
    try:
        return dict(annotations=save_truth(get_job(job_id), body.annotations))
    except (ValueError, KeyError, TypeError) as exc:
        fail(ValueError(str(exc)))


class VideoFaceTruth(BaseModel):
    labels: list[dict] = Field(default_factory=list, max_length=1000)


@app.post('/api/videos/{job_id}/reuse-face-ground-truth')
def reuse_video_face_truth(job_id: str):
    return reuse_face_truth(get_job(job_id))


@app.get('/api/videos/{job_id}/face-ground-truth/{frame_id}')
def video_faces_for_annotation(job_id: str, frame_id: int):
    try:
        return annotation_faces(get_job(job_id),frame_id)
    except (ValueError, KeyError, TypeError) as exc:
        fail(ValueError(str(exc)))


@app.put('/api/videos/{job_id}/face-ground-truth/{frame_id}')
def update_video_face_truth(job_id: str, frame_id: int, body: VideoFaceTruth):
    try:
        return dict(labels=save_face_truth(get_job(job_id),frame_id,body.labels))
    except (ValueError, KeyError, TypeError) as exc:
        fail(ValueError(str(exc)))


@app.post('/api/videos/{job_id}/evaluate')
def evaluate_video_job(job_id: str):
    return evaluate_video(get_job(job_id))


@app.get('/api/videos/{job_id}/frames')
def frames(job_id: str, start: float = 0, end: float = 3):
    return get_job(job_id).frames(start, end)


class Control(BaseModel):
    action: str
    timestamp: float = Field(0, ge=0)


@app.post('/api/videos/{job_id}/control')
def video_control(job_id: str, body: Control):
    if body.action in ('play', 'seek') and active_live and not active_live['stopped']:
        raise HTTPException(409, 'Dừng camera trước khi tiếp tục phân tích video.')
    try:
        return get_job(job_id).control(body.action, body.timestamp)
    except ValueError as exc:
        fail(exc)


@app.delete('/api/videos/{job_id}')
async def delete_job(job_id: str):
    job = get_job(job_id)
    await asyncio.to_thread(job.dispose)
    jobs.pop(job_id, None)
    return dict(ok=True)


def valid_ws(ws):
    return ws.headers.get('origin', '') in ('http://localhost:5173', 'http://127.0.0.1:5173',
        'http://localhost:8000', 'http://127.0.0.1:8000', '') and ws.url.hostname in ('localhost', '127.0.0.1', 'testserver')


@app.websocket('/ws/events')
async def events_ws(ws: WebSocket):
    if not valid_ws(ws):
        await ws.close(code=1008); return
    await broker.attach(ws)
    try:
        await ws.send_json(dict(type='snapshot', jobs=[j.status() for j in jobs.values()]))
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        broker.clients.discard(ws)


@app.websocket('/ws/live')
async def live_ws(ws: WebSocket):
    global active_live
    if not valid_ws(ws):
        await ws.close(code=1008); return
    await ws.accept()
    session = None
    try:
        header = await ws.receive_json()
        return_image = header.get('return_image', True)
        track_between_frames = header.get('track_between_frames') is True
        choice = CameraChoice(**header)
        if not camera_matches(choice.kind, choice.label):
            raise ValueError('Camera không đúng nguồn đã chọn.')
        if not header.get('session_id'):
            header['session_id'] = (await start_camera(choice))['session_id']
        async with session_lock:
            if (not active_live or active_live['id'] != header['session_id'] or active_live['stopped']
                    or active_live['attached'] or active_live['source'] != choice.label or active_live['kind'] != choice.kind):
                raise ValueError('Phiên camera không hợp lệ hoặc đã đóng.')
            session = active_live
            session['attached'] = True
        diagnostic = None
        if header.get('diagnostic_id'):
            diagnostic = diagnostic_run(header['diagnostic_id'])
            if choice.kind != 'camo' or diagnostic.state != 'opening' or diagnostic.session_id:
                raise ValueError('Lượt kiểm thử cần xác nhận mới và nguồn camera điện thoại thật.')
            diagnostic.session_id = session['id']
        await asyncio.to_thread(engine.load)
        pipeline = Pipeline(engine, store,model_overrides=getattr(diagnostic,'runtime_models',{}),
                            shadow_models=getattr(diagnostic,'runtime_shadow_models',None))
        await ws.send_json(dict(type='ready',session_id=session['id'],track_between_frames=track_between_frames))
        flow = FaceFlow()
        send_lock = asyncio.Lock()
        visual_task = None
        inference_task = None
        async def send_payload(payload):
            async with send_lock:
                await ws.send_json(payload)
        async def publish_visual(image,result,frames,captured,tracking_payloads):
            try:
                tracked = await asyncio.to_thread(flow.between,image,result,frames)
                if diagnostic and not session['stopped']:
                    for frame in tracked:
                        important=(not frame['overlays'] and diagnostic.expected_present) or len(frame['overlays'])<len(result['overlays']) or any(face.get('flow_direction')=='forward' for face in frame['overlays'])
                        payload=tracking_payloads.get(frame['frame_id'])
                        if important and payload is not None:
                            at=captured+(frame['captured_at']-result['captured_at'])/1000
                            diagnostic.retain_visual_example(frame,at,payload,result['frame_id'])
                if tracked and not session['stopped']:
                    await send_payload(dict(type='visual_frames',session_id=session['id'],frames=tracked,
                                            visual_tracking_ms=result.get('visual_tracking_ms'),visual_tracking_summary=result.get('visual_tracking_summary')))
            except (WebSocketDisconnect,RuntimeError):
                pass
            except Exception as exc:
                try:
                    await send_payload(dict(type='error',detail='Lỗi theo dõi hình ảnh: '+str(exc)))
                except Exception:
                    pass
        last_frame = -1
        last_camera_timestamp = None
        completed_frames = 0
        measurement_start = None
        started = time.perf_counter()
        while not session['stopped']:
            header = await asyncio.wait_for(ws.receive_json(),timeout=20)
            raw = await asyncio.wait_for(ws.receive_bytes(),timeout=20)
            if len(raw)>20*1024**2:
                raise ValueError('Frame camera quá lớn.')
            frame_id = int(header['frame_id'])
            tracking_headers = header.get('tracking_frames',[])
            if not isinstance(tracking_headers,list) or len(tracking_headers)>12 or (tracking_headers and not track_between_frames):
                raise ValueError('Gói frame theo dõi không hợp lệ.')
            image = await asyncio.to_thread(decode,raw)
            received = time.monotonic()
            captured = diagnostic.capture_time(header,received) if diagnostic else received
            timestamp = camera_timestamp(header.get('timestamp'),last_camera_timestamp if frame_id>last_frame else None)
            if frame_id > last_frame:
                fresh_measurement = bool(diagnostic and diagnostic.state=='running' and diagnostic.started is not None and
                                         captured>=diagnostic.started and measurement_start!=diagnostic.started)
                if fresh_measurement:
                    measurement_start = diagnostic.started
                inference_task = asyncio.create_task(asyncio.to_thread(pipeline.run,image,frame_id,timestamp,
                                                                        reset_temporal=fresh_measurement,diagnostic_details=bool(diagnostic)))
            tracking_frames = []
            tracking_payloads = {}
            payload_size,pixels = len(raw),image.shape[0]*image.shape[1]
            previous_id,previous_pts = last_frame,None
            for metadata in tracking_headers:
                identifier,pts = int(metadata['frame_id']),float(metadata['timestamp'])
                width,height = int(metadata['width']),int(metadata['height'])
                pixels += width*height
                if width<=0 or height<=0 or pixels>32_000_000:
                    raise ValueError('Gói theo dõi vượt giới hạn số pixel trong bộ nhớ.')
                if not previous_id<identifier<frame_id or not np.isfinite(pts) or (previous_pts is not None and pts<=previous_pts):
                    raise ValueError('Frame theo dõi sai thứ tự hoặc timestamp.')
                payload = await asyncio.wait_for(ws.receive_bytes(),timeout=20)
                payload_size += len(payload)
                if payload_size>40*1024**2:
                    raise ValueError('Gói frame camera vượt giới hạn bộ nhớ.')
                decoded = await asyncio.to_thread(decode_tracking_frame,payload)
                if decoded.shape != (height,width) or decoded.shape != image.shape[:2]:
                    raise ValueError('Kích thước frame theo dõi không khớp camera.')
                tracking_frames.append((metadata,decoded))
                tracking_payloads[identifier]=payload
                previous_id,previous_pts = identifier,pts
            if frame_id <= last_frame:
                continue
            last_frame = frame_id
            last_camera_timestamp = timestamp
            result = await inference_task
            inference_task = None
            if session['stopped']:
                break
            completed_frames += 1
            result['ai_throughput_fps'] = completed_frames/max(.001,time.perf_counter()-started)
            result.update(type='live_frame',source=session['source'],session_id=session['id'],
                          captured_at=header.get('captured_at'),media_timestamp=header.get('timestamp'),
                          sampling=header.get('sampling',{}),overlay_origin='ai_detection',live_timing_revision='camera-media-clock-v1',
                          diagnostic_timestamp_basis='browser_capture_elapsed' if diagnostic and header.get('diagnostic_elapsed') is not None else 'server_receive_legacy')
            if diagnostic and (diagnostic.state!='running' or diagnostic.started is None or
                               captured<diagnostic.started or captured>=diagnostic.started+diagnostic.duration or
                               time.monotonic()>=diagnostic.started+diagnostic.duration):
                result['events'] = []
            for event in result['events']:
                event_with_evidence(event,image,session['source'],session['id'])
            if diagnostic:
                diagnostic.retain_example(result,captured,raw)
                diagnostic.observe(result,captured)
                result['diagnostic'] = diagnostic.view()
            if return_image:
                result['image'] = 'data:image/jpeg;base64,'+base64.b64encode(jpeg(image,85)).decode()
            await send_payload(result)
            # One CPU tracking job at a time. GPU recognition never waits for this job.
            if track_between_frames and (visual_task is None or visual_task.done()):
                visual_task = asyncio.create_task(publish_visual(image,result,tracking_frames,captured,tracking_payloads))
            await broker.publish(dict(type='live_events',session_id=session['id'],events=result['events']))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json(dict(type='error', detail=str(exc)))
        except Exception:
            pass
    finally:
        pending = [task for task in (locals().get('visual_task'),locals().get('inference_task')) if task is not None]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending,return_exceptions=True)
        if 'diagnostic' in locals() and diagnostic:
            diagnostic.finish('cancelled')
            diagnostic.runtime_models={}
        if session and active_live is session:
            session['stopped'] = True
            active_live = None
        try:
            await ws.close()
        except Exception:
            pass


dist = ROOT/'frontend/dist'
if dist.exists():
    app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
