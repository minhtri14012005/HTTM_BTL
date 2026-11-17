import io
import json
import time
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.ai import engine, jpeg, Pipeline
from backend.store import Store
from backend.temporal import Temporal, match
from backend.sessions import camera_matches, VideoJob, Broker


@pytest.fixture(scope='session')
def faces():
    from skimage import data
    import matplotlib.cbook as cbook
    astronaut = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    hopper = cv2.imread(str(cbook.get_sample_data('grace_hopper.jpg', asfileobj=False)))
    return astronaut, hopper


@pytest.fixture(scope='session')
def registered(faces, tmp_path_factory):
    store = Store(tmp_path_factory.mktemp('registration'))
    return [engine.register(jpeg(img), store.settings()) for img in faces]


def test_real_hybrid_known_unknown_and_quality(registered, faces):
    first, second = registered
    for _, embeddings, quality in registered:
        assert len(embeddings) == 3
        for vector in embeddings.values():
            assert vector.shape == (512,)
            assert np.isfinite(vector).all()
            assert abs(np.linalg.norm(vector)-1) < 1e-5
        assert set(quality['models']) == {'lvface', 'antelopev2', 'petalface'}
    key=engine.versions['lvface']
    gallery = [('a', 'Astronaut test', first[1][key])]
    assert match(first[1][key], gallery, .62, .08)[0] == 'a'
    assert match(second[1][key], gallery, .62, .08)[0] is None
    # Small exposure/JPEG variation: real pipeline must recognize the registered face.
    variant = cv2.convertScaleAbs(faces[0], alpha=.95, beta=4)
    embedding = engine.register(jpeg(variant, 85), Store().settings())[1]
    assert match(embedding[key], gallery, .62, .08)[0] == 'a'
    with pytest.raises(ValueError, match='đúng 1'):
        engine.register(jpeg(np.full((400,400,3), 200, np.uint8)), Store().settings())
    with pytest.raises(Exception):
        engine.register(b'bad image', Store().settings())


def test_temporal_competition_dedup_reentry():
    state = Temporal(); settings = Store().settings()
    assert state.update(1, 'a', 0., settings)[0] == 'candidate'
    assert state.update(1, 'a', .1, settings)[0] == 'candidate'
    assert state.update(1, 'a', .2, settings) == ('confirmed', True, 1)
    assert state.update(1, 'a', .3, settings) == ('confirmed', False, 1)
    assert state.update(1, None, .4, settings)[0] == 'unknown'
    assert state.update(1, 'b', .5, settings)[0] == 'candidate'
    for t in (5., 5.1): state.update(2, 'a', t, settings)
    assert state.update(2, 'a', 5.2, settings) == ('confirmed', True, 2)
    e=np.array([1.,0.],np.float32)
    assert match(e,[('a','A',e),('b','B',e)],.62,.08)[0] is None


def test_store_persistence_no_empty_or_orphan_images(tmp_path, registered):
    st = Store(tmp_path)
    pid = st.add('One photo', [registered[0]])
    assert len(st.people()[0]['images']) == 1
    st.add('', [registered[0], registered[1]], pid)
    st.update(pid, name='Renamed', selected=False)
    reopened = Store(tmp_path)
    assert reopened.people()[0]['name']=='Renamed'
    assert reopened.gallery()==[]
    imgs = reopened.people()[0]['images']
    reopened.delete_image(pid, imgs[0]['id']); reopened.delete_image(pid, imgs[1]['id'])
    with pytest.raises(ValueError): reopened.delete_image(pid, imgs[2]['id'])
    reopened.delete(pid)
    assert reopened.people()==[]
    assert list((tmp_path/'people').glob('*'))==[]


def test_real_tracking_confirmation(tmp_path, registered, faces):
    st=Store(tmp_path);pid=st.add('Astronaut test',[registered[0]])
    pipe=Pipeline(engine,st)
    results=[pipe.run(faces[0],i,i*.2) for i in range(5)]
    assert any(o['status']=='candidate' for o in results[-1]['overlays'])
    assert sum(len(r['events']) for r in results)==0
    assert all(o['bbox']==o['face_bbox'] for r in results for o in r['overlays'])
    unknown=pipe.run(faces[1],5,1.2)
    assert not unknown['events']
    assert all(o['person_id'] is None for o in unknown['overlays'])


def make_video(path, image, count=8, fps=8):
    h,w=image.shape[:2]
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    assert writer.isOpened()
    for _ in range(count): writer.write(image)
    writer.release()


def wait_until(predicate, timeout=60):
    started=time.monotonic()
    while not predicate():
        assert time.monotonic()-started < timeout
        time.sleep(.05)


def test_video_every_frame_pause_seek_timestamps_and_portrait(tmp_path,registered,faces):
    st=Store(tmp_path);st.add('Astronaut test',[registered[0]])
    path=tmp_path/'test.mp4';make_video(path,faces[0],12,8)
    broker=Broker();job=VideoJob(path,tmp_path,engine,st,broker)
    try:
        assert job.state=='ready'
        job.control('play');wait_until(lambda:job.processed>=2 or job.error)
        job.control('pause');time.sleep(.8)
        stopped=job.processed;time.sleep(.3)
        assert job.processed==stopped
        job.control('seek');wait_until(lambda:job.state in ('complete','error'))
        assert job.error is None
        frames=job.frames(0,10)
        assert len(frames)==12
        assert [r['frame_id'] for r in frames]==list(range(12))
        assert all(a['timestamp']<b['timestamp'] for a,b in zip(frames,frames[1:]))
        assert len(job.events)==0  # Uncalibrated models must never emit confirmed events.
        count=len(job.events);job.control('seek');job.control('play');assert len(job.events)==count
    finally:job.close()
    portrait=np.zeros((640,360,3),np.uint8)
    p=tmp_path/'portrait.mp4';make_video(p,portrait,2,4)
    j=VideoJob(p,tmp_path,engine,st,broker)
    assert j.width==360 and j.height==640
    j.close()
    bad=tmp_path/'bad.mp4';bad.write_bytes(b'invalid')
    with pytest.raises(ValueError):VideoJob(bad,tmp_path,engine,st,broker)


def test_api_real_registration_crud_ws_and_upload(tmp_path,faces,monkeypatch):
    import backend.main as main
    monkeypatch.setattr(main,'store',Store(tmp_path))
    with TestClient(main.app) as client:
        wait_until(lambda:client.get('/api/health').json()['ready'])
        assert client.get('/api/persons').json()==[]
        assert client.post('/api/calibration/preview',json={}).json()['profiles']==[]
        assert client.post('/api/calibration/apply',json={}).status_code==409
        bad=client.post('/api/persons',data={'name':'bad'},files={'files':('bad.jpg',b'broken','image/jpeg')})
        assert bad.status_code in (400,503)
        response=client.post('/api/persons',data={'name':'Test person'},files={'files':('face.jpg',jpeg(faces[0]),'image/jpeg')})
        assert response.status_code==201,response.text
        person=response.json();pid=person['id']
        assert len(person['images'])==1
        assert client.get(person['images'][0]['url']).status_code==200
        assert client.patch('/api/persons/'+pid,json={'name':'Updated','selected':False}).status_code==200
        assert client.post('/api/persons/'+pid+'/images',files=[('files',('other.jpg',jpeg(faces[0]),'image/jpeg'))]).status_code==201
        images=client.get('/api/persons').json()[0]['images']
        assert client.delete(f'/api/persons/{pid}/images/{images[0]["id"]}').status_code==200
        assert client.delete(f'/api/persons/{pid}/images/{images[1]["id"]}').status_code==400
        assert client.post('/api/camera/validate',json={'kind':'camo','label':'Integrated Camera'}).status_code==400
        assert client.post('/api/camera/validate',json={'kind':'camo','label':'Camo Camera'}).status_code==200
        assert client.get('/api/persons',headers={'origin':'https://untrusted.example'}).status_code==403
        with client.websocket_connect('/ws/events') as ws:
            assert ws.receive_json()['type']=='snapshot'
        with client.websocket_connect('/ws/live') as ws:
            reserved=client.post('/api/camera/start',json={'kind':'laptop','label':'Integrated Camera'}).json()
            ws.send_json({'kind':'laptop','label':'Integrated Camera','session_id':reserved['session_id'],'return_image':False})
            ready=ws.receive_json();assert ready['type']=='ready'
            ws.send_json({'frame_id':1,'timestamp':0,'captured_at':123})
            ws.send_bytes(jpeg(faces[0]))
            frame=ws.receive_json();assert frame['type']=='live_frame'
            assert frame['frame_id']==1 and frame['session_id']==ready['session_id']
            assert frame['media_timestamp']==0 and frame['captured_at']==123
            assert frame['timestamp']==0 and frame['live_timing_revision']=='camera-media-clock-v1'
            assert 'image' not in frame  # The canvas already owns the exact captured frame.
            assert client.post('/api/camera/start',json={'kind':'laptop','label':'Integrated Camera'}).status_code==409
            assert client.post('/api/camera/stop',json={'session_id':'stale-session'}).status_code==200
            assert main.active_live['stopped'] is False
            assert client.post('/api/camera/stop',json={'session_id':reserved['session_id']}).status_code==200
        with client.websocket_connect('/ws/live') as ws:
            reserved=client.post('/api/camera/start',json={'kind':'laptop','label':'Integrated Camera'}).json()
            ws.send_json(dict(kind='laptop',label='Integrated Camera',session_id=reserved['session_id'],
                              return_image=False,track_between_frames=True))
            ready=ws.receive_json()
            assert ready['track_between_frames'] is True
            ws.send_json(dict(frame_id=1,timestamp=0,captured_at=100))
            ws.send_bytes(jpeg(faces[0]))
            same_input=ws.receive_json()
            assert same_input['frame_id']==1
            assert [face['face_bbox'] for face in same_input['overlays']]==[face['face_bbox'] for face in frame['overlays']]
            for before,after in zip(frame['overlays'],same_input['overlays']):
                for key,score in before['model_scores'].items():
                    assert after['model_scores'][key].get('score')==pytest.approx(score.get('score'),abs=1e-6)
            height,width=faces[0].shape[:2]
            ws.send_json(dict(frame_id=4,timestamp=.12,captured_at=220,tracking_frames=[
                dict(frame_id=2,timestamp=.04,captured_at=140,width=width,height=height),
                dict(frame_id=3,timestamp=.08,captured_at=180,width=width,height=height)]))
            ws.send_bytes(jpeg(faces[0]))
            gray=cv2.cvtColor(faces[0],cv2.COLOR_BGR2GRAY)
            ws.send_bytes(jpeg(gray));ws.send_bytes(jpeg(gray))
            analyzed=ws.receive_json()
            assert analyzed['type']=='live_frame' and analyzed['frame_id']==4
            visual=ws.receive_json()
            assert visual['type']=='visual_frames'
            assert [frame['frame_id'] for frame in visual['frames']]==[2,3]
            assert all(frame['overlays'] for frame in visual['frames'])
            assert all(face['tracking_origin']=='optical_flow' for frame in visual['frames'] for face in frame['overlays'])
            assert client.post('/api/camera/stop',json={'session_id':reserved['session_id']}).status_code==200
        path=tmp_path/'input.mp4';make_video(path,faces[0],4,4)
        response=client.post('/api/videos',files={'file':('input.mp4',path.read_bytes(),'video/mp4')})
        assert response.status_code==201,response.text
        job=response.json()
        assert client.get(job['media_url'],headers={'Range':'bytes=0-99'}).status_code==206
        assert client.post(f'/api/videos/{job["id"]}/control',json={'action':'play'}).status_code==200
        wait_until(lambda:main.jobs[job['id']].state in ('complete','error'))
        assert main.jobs[job['id']].error is None
        assert client.delete(f'/api/videos/{job["id"]}').status_code==200
        assert client.delete('/api/persons/'+pid).status_code==200


def test_camera_mapping():
    assert not camera_matches('iriun','Iriun Webcam')
    assert not camera_matches('laptop','Iriun Webcam')
    assert not camera_matches('camo','Iriun Webcam')
    assert not camera_matches('iriun','Camo Camera')
    assert camera_matches('camo','Camo Camera')
    assert not camera_matches('camo','Integrated Webcam')
    assert camera_matches('laptop','Integrated Camera')
    assert not camera_matches('laptop','OBS Virtual Camera')
    assert not camera_matches('laptop','Camo Camera')


def test_model_missing_error_is_explicit(tmp_path,monkeypatch):
    import backend.ai as ai
    monkeypatch.setattr(ai,'MODELS',tmp_path)
    isolated=ai.Engine()
    with pytest.raises(RuntimeError,match='setup'):
        isolated.load()
    assert not isolated.ready and isolated.error


def test_rotation_metadata_normalized_without_stretch(tmp_path):
    import subprocess,imageio_ffmpeg
    from backend.media import normalize_video
    original=tmp_path/'landscape.mp4'
    make_video(original,np.full((72,128,3),120,np.uint8),2,4)
    rotated=tmp_path/'rotated.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-display_rotation:v:0','90',
             '-i',str(original),'-c','copy',str(rotated)]
    subprocess.run(command,check=True,capture_output=True)
    target=tmp_path/'normalized.mp4';normalize_video(rotated,target)
    cap=cv2.VideoCapture(str(target))
    try:
        ok,frame=cap.read()
        assert ok and frame.shape[:2]==(128,72)
    finally:cap.release()


def test_diagnostic_websocket_resets_preparation_votes_before_measured_frames(tmp_path,registered,faces,monkeypatch):
    import backend.main as main
    store=Store(tmp_path);pid=store.add('Fixture',[registered[0]])
    monkeypatch.setattr(main,'store',store);monkeypatch.setattr(main,'diagnostics',{})
    for key in ('lvface','antelopev2'):
        for band in ('normal','small'):
            store.calibrate(engine.versions[key],band,dict(threshold=.8,margin=.08))
    with TestClient(main.app) as client:
        wait_until(lambda:client.get('/api/health').json()['ready'])
        response=client.post('/api/diagnostics',json=dict(ready=True,scenario='Chính diện 1 m',person_id=pid,expected_present=True))
        assert response.status_code==201
        rid=response.json()['id']
        reserved=client.post('/api/camera/start',json=dict(kind='camo',label='Camo Camera QA')).json()
        with client.websocket_connect('/ws/live') as ws:
            ws.send_json(dict(kind='camo',label='Camo Camera QA',session_id=reserved['session_id'],diagnostic_id=rid,return_image=False))
            assert ws.receive_json()['type']=='ready'
            for identifier in range(1,4):
                ws.send_json(dict(frame_id=identifier,timestamp=identifier*.1,captured_at=identifier*100))
                ws.send_bytes(jpeg(faces[0]));warm=ws.receive_json();assert not warm['events']
            assert warm['overlays'][0]['status']=='confirmed'
            # Isolated fixture clock: no real camera/person is involved.
            main.diagnostics[rid].prepared-=6
            assert client.post('/api/diagnostics/'+rid+'/begin',json={}).status_code==200
            measured=[]
            for identifier in range(4,7):
                ws.send_json(dict(frame_id=identifier,timestamp=identifier*.1,captured_at=identifier*100))
                ws.send_bytes(jpeg(faces[0]));measured.append(ws.receive_json())
            examples=client.get(f'/api/diagnostics/{rid}/examples').json()
            assert not examples['volatile'] and len(examples['examples'])==3
            item=examples['examples'][0]
            image_response=client.get(f'/api/diagnostics/{rid}/examples/{item["frame_id"]}')
            assert image_response.content==jpeg(faces[0]) and image_response.headers['cache-control']=='no-store'
            assert 'landmarks' in measured[0]['overlays'][0]
            assert measured[0]['measurement_evidence_reset'] is True
            assert measured[0]['overlays'][0]['status']=='candidate'
            assert not measured[0]['events'] and not measured[1]['events']
            assert len(measured[2]['events'])==1 and measured[2]['overlays'][0]['status']=='confirmed'
            assert client.post('/api/camera/stop',json=dict(session_id=reserved['session_id'])).status_code==200
        with store.db() as db:
            frames=db.execute('SELECT frame_id FROM diagnostic_frames WHERE run_id=? ORDER BY frame_id',(rid,)).fetchall()
            assert [row[0] for row in frames]==[4,5,6]

        response=client.post('/api/diagnostics',json=dict(ready=True,scenario='Chính diện 1 m',person_id=pid,expected_present=True))
        assert response.status_code==201
        assert len(client.get(f'/api/diagnostics/{rid}/examples').json()['examples'])==3


def test_candidate_diagnostic_has_fresh_consent_and_releases_optional_model(tmp_path,registered,faces,monkeypatch):
    import backend.main as main
    import backend.pose_expert as expert_module
    if not (Path(__file__).resolve().parents[2]/'models/candidates/kprpe12m/model.safetensors').is_file():
        pytest.skip('Optional candidate checkpoint is not installed.')
    store=Store(tmp_path);pid=store.add('Fixture',[registered[0]])
    monkeypatch.setattr(main,'store',store);monkeypatch.setattr(main,'diagnostics',{})
    original=Path(store.image_path(store.people()[0]['images'][0]['id'])).read_bytes()
    with TestClient(main.app) as client:
        wait_until(lambda:client.get('/api/health').json()['ready'])
        with monkeypatch.context() as guard:
            def forbidden(*args,**kwargs):raise AssertionError('Model must not load before fresh consent.')
            guard.setattr(expert_module,'KPRPEExpert',forbidden)
            rejected=client.post('/api/diagnostics',json=dict(ready=False,scenario='Chính diện 1 m',person_id=pid,expected_present=True,expert='kprpe12m'))
            assert rejected.status_code==422
        response=client.post('/api/diagnostics',json=dict(ready=True,scenario='Chính diện 1 m',person_id=pid,expected_present=True,expert='kprpe12m'))
        assert response.status_code==201,response.text
        rid=response.json()['id'];run=main.diagnostics[rid]
        assert 'kprpe12m' in run.runtime_models and run.context['shadow_models']['kprpe12m'].startswith('kprpe12m:')
        person=store.people()[0]
        assert len(person['images'])==1 and Path(store.image_path(person['images'][0]['id'])).read_bytes()==original
        assert len(store.gallery_model(run.context['shadow_models']['kprpe12m']))==1
        reserved=client.post('/api/camera/start',json=dict(kind='camo',label='Camo Camera QA')).json()
        with client.websocket_connect('/ws/live') as ws:
            ws.send_json(dict(kind='camo',label='Camo Camera QA',session_id=reserved['session_id'],diagnostic_id=rid,return_image=False))
            assert ws.receive_json()['type']=='ready'
            ws.send_json(dict(frame_id=1,timestamp=.1,captured_at=100));ws.send_bytes(jpeg(faces[0]))
            frame=ws.receive_json()
            assert frame['shadow_models']==['kprpe12m'] and frame['overlays'][0]['model_scores']['kprpe12m']['shadow']
            assert not frame['events']
            assert client.post('/api/diagnostics/'+rid+'/stop').status_code==200
        assert not run.runtime_models and main.active_live is None


def test_calibrated_expert_mode_requires_current_profile_and_validation_role(tmp_path,registered,monkeypatch):
    import backend.main as main
    import backend.pose_expert as expert_module
    from backend.fusion import quality_basis
    store=Store(tmp_path);pid=store.add('Fixture',[registered[0]]);vector=next(iter(registered[0][1].values()))
    monkeypatch.setattr(main,'store',store);monkeypatch.setattr(main,'diagnostics',{})
    class Adapter:
        version='kprpe12m:api-unit'
        def __init__(self,*args):pass
        def embed(self,*args):return vector,dict(version=self.version)
    monkeypatch.setattr(expert_module,'KPRPEExpert',Adapter)
    body=dict(ready=True,scenario='Quay trái/phải 1 m',person_id=pid,expected_present=True,expert='kprpe12m',expert_mode='calibrated',purpose='validation')
    with TestClient(main.app) as client:
        wait_until(lambda:client.get('/api/health').json()['ready'])
        assert client.post('/api/diagnostics',json=body).status_code==400
        assert client.post('/api/diagnostics',json=dict(body,purpose='calibration')).status_code==400
        store.calibrate(Adapter.version,'normal_high_residual',dict(threshold=.8,margin=.1,quality_basis=quality_basis('high_residual'),cross_quality_negative_observations=10,cross_quality_false_passes=0))
        store.calibrate(Adapter.version,'tiny',dict(threshold=.8,margin=.1,validation_negative_observations=20,validation_false_passes=0))
        store.calibrate(Adapter.version,'small',dict(threshold=.8,margin=.1,validation_negative_observations=20,validation_false_passes=1))
        store.calibrate(Adapter.version,'small_high_residual',dict(threshold=.8,margin=.1,quality_basis=quality_basis('high_residual'),validation_negative_observations=20,validation_false_passes=0,cross_quality_negative_observations=0,cross_quality_false_passes=1))
        store.calibrate(Adapter.version,'tiny_high_residual',dict(threshold=.8,margin=.1,quality_basis=quality_basis('high_residual'),validation_negative_observations=20,validation_false_passes=None))
        response=client.post('/api/diagnostics',json=body)
        assert response.status_code==201,response.text
        run=main.diagnostics[response.json()['id']]
        assert run.runtime_shadow_models==[] and run.runtime_models['kprpe12m'].allowed_bands=={'normal_high_residual','tiny'}
        assert run.context['calibration_role']=='validation' and run.context['fusion_experts']['kprpe12m']==Adapter.version
        assert client.post('/api/diagnostics/'+run.id+'/stop').status_code==200
        assert main.active_live is None and not run.runtime_models
