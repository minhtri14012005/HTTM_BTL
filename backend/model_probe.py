"""Offline model evidence on every original-resolution video frame; no identity events."""
import hashlib
import json
import tempfile
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from .ai import Engine
from .models import RetinaR50,YoloFace
from .media import normalize_video
from .video_evaluation import truth_at
from .diagnostics import enrollment_snapshot


def probe_video(store,adapter,source,model_key='kprpe12m',device='cuda',progress=None):
    source=Path(source)
    with source.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    run_id=uuid.uuid4().hex
    enrollment=enrollment_snapshot(store)
    with store.db() as db:
        labels=[dict(r) for r in db.execute('SELECT * FROM video_face_ground_truth WHERE source_hash=?',(source_hash,))]
        row=db.execute('SELECT annotations FROM video_ground_truth WHERE source_hash=?',(source_hash,)).fetchone()
        annotations=json.loads(row[0]) if row else []
        db.execute('INSERT INTO model_probes VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)',
            (run_id,adapter.version,source_hash,'all_native_frames_no_fusion',json.dumps(dict(state='analyzing'))))
    detector=Engine();root=Path(__file__).resolve().parents[1]/'models/hybrid'
    detector.detector=RetinaR50(root,device);detector.fallback=YoloFace(root/'yolov5m-face.onnx',device)
    gallery=store.gallery_model(adapter.version)
    if not gallery:raise ValueError('Chưa có embedding đăng ký đúng version của model đối chứng.')
    settings=store.settings();started=time.perf_counter();negative_max=-1.;negative=known=correct=errors=quality_rejects=alignment_rejects=0
    count=faces_count=0
    try:
        with tempfile.TemporaryDirectory(prefix='facewatch-model-probe-') as folder:
            target=Path(folder)/'normalized.mp4';normalize_video(source,target)
            cap=cv2.VideoCapture(str(target))
            try:
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
                last=-1.
                while True:
                    ok,image=cap.read()
                    if not ok:break
                    timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if timestamp<=last:timestamp=max(last+1/fps,count/fps)
                    last=timestamp;overlays=[];frame_started=time.perf_counter()
                    for box,points,detector_name in detector.detect(image):
                        face=dict(face_bbox=box[:4].tolist(),bbox=box[:4].tolist(),detector=detector_name,model_scores={})
                        try:
                            face['quality']=detector.quality(image,box,settings,inference=True)
                        except ValueError:
                            quality_rejects+=1;continue
                        try:
                            vector,info=adapter.embed(image,points,strict_alignment=False)
                            ranked={}
                            for pid,name,reference in gallery:
                                value=float(np.clip(vector@reference,-1,1))
                                if pid not in ranked or value>ranked[pid]:ranked[pid]=value
                            ranking=sorted(ranked.items(),key=lambda item:item[1],reverse=True)
                            pid,value=ranking[0];margin=value-ranking[1][1] if len(ranking)>1 else None
                            face['model_scores'][model_key]=dict(person_id=pid,score=value,margin=margin,**info)
                        except Exception as exc:
                            message=str(exc)
                            if any(word in message for word in ('Alignment','Landmarks','landmarks','Crop','căn chỉnh')):alignment_rejects+=1
                            else:errors+=1
                            face['model_scores'][model_key]=dict(error=message);overlays.append(face);continue
                        for label in labels:
                            details=json.loads(label['details'])
                            if label['frame_id']!=count or details.get('bbox')!=face['face_bbox']:continue
                            with store.db() as db:
                                previous=db.execute('SELECT timestamp,metrics FROM video_measurements WHERE run_id=? AND frame_id=?',(label['run_id'],count)).fetchone()
                            if not previous or abs(previous['timestamp']-timestamp)>.001:continue
                            previous_frame=json.loads(previous['metrics'])
                            if (previous_frame.get('width'),previous_frame.get('height'))!=(image.shape[1],image.shape[0]):continue
                            face['ground_truth']=dict(details,source='user_label_reused_exact_frame',source_run_id=label['run_id'])
                            break
                        label=face.get('ground_truth')
                        if label and label.get('kind')=='person':
                            known+=1;correct+=int(label['person_id']==pid)
                        elif truth_at(annotations,pid,timestamp) is False:
                            negative+=1;negative_max=max(negative_max,value)
                        overlays.append(face)
                    metrics=dict(frame_id=count,timestamp=timestamp,width=image.shape[1],height=image.shape[0],
                        overlays=overlays,events=[],inference_only=True,model_version=adapter.version,
                        processing_ms=(time.perf_counter()-frame_started)*1000)
                    with store.db() as db:
                        db.execute('INSERT INTO video_measurements VALUES (?,?,?,?)',(run_id,count,timestamp,json.dumps(metrics)))
                    count+=1;faces_count+=len(overlays)
                    if progress and count%150==0:progress(dict(id=run_id,processed=count,total=total,negative_faces=negative))
            finally:cap.release()
        if count!=total:raise ValueError('Video đối chứng chưa đọc đủ mọi frame.')
        result=dict(state='complete',frames=count,total=total,face_observations=faces_count,source_hash=source_hash,
            model=adapter.version,protocol='all_native_frames_no_fusion',enrollment_changed=enrollment_snapshot(store)!=enrollment,declared_absence_comparisons=negative,
            declared_absence_max=negative_max if negative else None,human_labelled_faces=known,top1_correct=correct,
            quality_rejected=quality_rejects,alignment_rejected=alignment_rejects,inference_errors=errors,seconds=time.perf_counter()-started,
            note='Full-frame model inference only; no live confirmation, tracking or notification accuracy claimed.')
    except Exception as exc:
        result=dict(state='error',frames=count,error=str(exc));raise
    finally:
        with store.db() as db:
            db.execute('UPDATE model_probes SET result=?,updated_at=CURRENT_TIMESTAMP WHERE run_id=?',(json.dumps(result),run_id))
    return dict(id=run_id,**result)
