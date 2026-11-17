import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
from backend.ai import engine, Engine, Pipeline, jpeg
from backend.models import align, rgb_tensor, REFERENCE
from backend.store import Store
from backend.diagnostics import Diagnostic


def test_preprocessing_alignment_and_invalid_landmarks():
    image = np.zeros((160,160,3), np.uint8)
    image[:,:,2] = 255
    crop, error = align(image, REFERENCE)
    assert crop.shape == (112,112,3) and error < 1e-3
    tensor = rgb_tensor(crop)
    assert tensor.shape == (1,3,112,112)
    assert tensor[0,0,50,50] == 1 and tensor[0,2,50,50] == -1
    petal, _ = align(image, REFERENCE,120,petal=True)
    assert petal.shape == (120,120,3)
    with pytest.raises(ValueError): align(image, REFERENCE[::-1])
    with pytest.raises(ValueError): align(image, np.full((5,2),np.nan))


def test_petal_alignment_uses_warp_coordinates_for_rejection():
    image=np.full((160,160,3),120,np.uint8)
    points=REFERENCE.copy()+20
    points[2,0]+=30  # Residual is ~10.66 at the 112-pixel warp, not the final 120 input.
    with pytest.raises(ValueError,match='Alignment'):
        align(image,points,112)
    with pytest.raises(ValueError,match='Alignment'):
        align(image,points,120,petal=True)
    measured,error=align(image,points,112,strict=False)
    assert measured.shape==(112,112,3) and error > 112*.09
    measured,error=align(image,points,120,petal=True,strict=False)
    assert measured.shape==(120,120,3) and error > 112*.09


def test_diagnostic_consent_5_and_30_seconds_and_no_preparation_metrics(tmp_path):
    store = Store(tmp_path)
    clock=[100.]
    with pytest.raises(ValueError):Diagnostic(store,'Chính diện 1 m',None,False,False)
    run=Diagnostic(store,'Chính diện 1 m',None,False,True,clock=lambda:clock[0])
    result=dict(frame_id=1,width=100,height=100,overlays=[],events=[],processing_ms=25)
    run.observe(result,100.)
    assert run.state=='preparing'
    with pytest.raises(ValueError):run.begin()
    clock[0]=105.;run.observe(result,104.9);run.begin()
    assert run.view()['remaining']==30
    run.observe(result,104.9)  # Result from before BEGIN cannot be counted.
    with store.db() as db: assert db.execute('SELECT count(*) FROM diagnostic_frames').fetchone()[0]==0
    clock[0]=106.;run.observe(result,105.5)
    clock[0]=135.;run.observe(dict(result,frame_id=2),134.9)
    assert run.state=='complete'
    assert run.summary['frames']==1 and run.summary['duration_seconds']==30
    assert run.summary['detection_recall'] is None
    assert run.finish()['state']=='complete'


def test_cancel_before_camera_collects_nothing(tmp_path):
    store=Store(tmp_path)
    run=Diagnostic(store,'Người ngoài watchlist',None,False,True)
    run.finish()
    assert run.state=='cancelled' and run.summary['frames']==0


def test_quality_gate_is_scale_consistent_without_resizing_inference_source():
    import cv2
    from skimage import data
    base = cv2.cvtColor(data.astronaut()[60:172,170:282], cv2.COLOR_RGB2BGR)
    large = cv2.resize(base,(336,336),interpolation=cv2.INTER_NEAREST)
    before = large.copy()
    settings=dict(min_face_pixels=28,min_sharpness=18)
    small_quality=Engine().quality(base,np.array([0,0,112,112,.99]),settings)
    big_quality=Engine().quality(large,np.array([0,0,336,336,.99]),settings)
    assert big_quality['sharpness'] > 18
    assert abs(big_quality['sharpness']/small_quality['sharpness']-1) < .01
    assert big_quality['native_sharpness'] < big_quality['sharpness']
    assert big_quality['face_pixels'] == 336 and np.array_equal(large,before)
    with pytest.raises(ValueError,match='Mặt mờ'):
        Engine().quality(np.full((336,336,3),120,np.uint8),np.array([0,0,336,336,.99]),settings)


def test_real_detectors_and_large_face_tiles_do_not_split_registration(tmp_path):
    from skimage import data
    import cv2
    image=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2BGR)
    engine.load()
    assert engine.detector.detect(image)
    assert engine.fallback.detect(image)
    large=cv2.resize(image,(1536,1536))
    original=jpeg(large)
    raw,vectors,_=engine.register(original,Store(tmp_path).settings())
    assert raw==original and len(vectors)==3
    st=Store(tmp_path);pid=st.add('Fixture only',[(raw,vectors,{})])
    before=hashlib.sha256(Path(st.image_path(st.people()[0]['images'][0]['id'])).read_bytes()).hexdigest()
    assert st.reindex(engine)==[]
    assert hashlib.sha256(Path(st.image_path(st.people()[0]['images'][0]['id'])).read_bytes()).hexdigest()==before
    with st.db() as db: assert db.execute('SELECT count(*) FROM model_embeddings').fetchone()[0]==3


def test_legacy_rows_never_deleted_by_schema_migration(tmp_path):
    st=Store(tmp_path)
    pid=st.add('Existing',[ (b'original-image',np.ones(512,np.float32),'legacy-quality') ])
    with st.db() as db:
        db.execute("UPDATE images SET model='legacy-model'")
        before=[tuple(r) for r in db.execute('SELECT * FROM images')]
    reopened=Store(tmp_path)
    with reopened.db() as db: assert [tuple(r) for r in db.execute('SELECT * FROM images')]==before
    assert reopened.exists(pid)


def test_consensus_conflict_single_watchlist_and_duplicate_faces(tmp_path):
    """Scripted vectors test decision logic only; not an accuracy benchmark."""
    import threading
    st=Store(tmp_path)
    a,b=np.eye(512,dtype=np.float32)[:2]
    versions={'lvface':'unit-lv','antelopev2':'unit-ant'}
    pid=st.add('A',[(b'test fixture',{v:a for v in versions.values()},{})])
    other=st.add('B',[(b'test fixture',{v:b for v in versions.values()},{})])
    st.update(other,selected=False)
    class ScriptedEngine:
        lock=threading.RLock()
        status={k:{'state':'ready'} for k in versions}
        def load(self):pass
        def detect(self,image):return self.faces
        def quality(self,*args,**kwargs):return dict(face_pixels=100,sharpness=100)
        def embed(self,image,points,key,**kwargs):return self.vectors[key],{}
        def gpu(self):return {'device':'test'}
    e=ScriptedEngine();e.versions=versions;e.vectors={k:a for k in versions}
    face=(np.array([10,10,110,110,.99]),REFERENCE,'test-detector')
    e.faces=[face];image=np.zeros((150,150,3),np.uint8)
    pipe=Pipeline(e,st)
    for i in range(4):
        result=pipe.run(image,i,i*.1)
        assert not result['events'] and result['overlays'][0]['name'] is None
    for v in versions.values():st.calibrate(v,'normal',dict(threshold=.8,margin=.08))
    pipe=Pipeline(e,st)
    results=[pipe.run(image,i,i*.1) for i in range(4)]
    assert sum(len(r['events']) for r in results)==1
    assert results[-1]['overlays'][0]['person_id']==pid
    assert results[-1]['overlays'][0]['bbox']==results[-1]['overlays'][0]['face_bbox']
    e.vectors['antelopev2']=b
    result=pipe.run(image,5,.5)
    assert not result['events'] and result['overlays'][0]['status']=='unknown'
    weak=.4*a+np.sqrt(1-.4**2)*np.eye(512,dtype=np.float32)[2]
    e.vectors={k:weak for k in versions}
    result=pipe.run(image,6,.6)
    assert result['overlays'][0]['status']=='unknown' and not result['events']
    assert result['overlays'][0]['reason'] != 'Các model bất đồng'
    e.vectors={k:a for k in versions};e.faces=[face,face]
    for i in range(6,10):
        result=pipe.run(image,i,i*.1)
        assert not result['events'] and all(o['name'] is None for o in result['overlays'])
    e.faces=[face]
    recovered=[pipe.run(image,i,i*.1) for i in range(10,16)]
    assert len({r['overlays'][0]['track_id'] for r in recovered})==1
    assert recovered[0]['overlays'][0]['status']=='candidate'
    assert recovered[-1]['overlays'][0]['status']=='confirmed'
    assert sum(len(r['events']) for r in recovered)==0  # Same continuous appearance: no duplicate alert.
    e.faces=[]
    assert pipe.run(image,16,1.6)['overlays']==[]


@pytest.mark.parametrize('condition,expect_petal', [
    ('clear_agreement', False), ('weak_core', True), ('disagreement', True),
    ('missing_core', True), ('high_residual', True), ('small_face', True),
])
def test_petal_routing_rechecks_uncertain_large_faces(tmp_path, condition, expect_petal):
    import threading
    store = Store(tmp_path)
    first, second, unrelated = np.eye(512, dtype=np.float32)[:3]
    versions = {key: 'routing-'+key for key in ('lvface', 'antelopev2', 'petalface')}
    store.add('Unit A', [(b'unit fixture', {v:first for v in versions.values()}, {})])
    store.add('Unit B', [(b'unit fixture', {v:second for v in versions.values()}, {})])

    class ScriptedEngine:
        lock = threading.RLock()
        status = {key:dict(state='ready') for key in versions}
        calls = None
        def load(self): pass
        def detect(self, image):
            return [(np.array([10,10,110,110,.99]), REFERENCE, 'unit detector')]
        def quality(self, *args, **kwargs):
            return dict(face_pixels=50 if condition == 'small_face' else 100, sharpness=100)
        def embed(self, image, points, key, **kwargs):
            self.calls.append(key)
            vector = first
            if key == 'antelopev2' and condition == 'weak_core':
                vector = .3*first+np.sqrt(1-.3**2)*unrelated
            if key == 'antelopev2' and condition == 'disagreement':
                vector = second
            return vector, dict(alignment_band='high_residual' if condition == 'high_residual' else 'regular')
        def gpu(self): return dict(device='unit')

    engine = ScriptedEngine()
    engine.versions, engine.calls = versions, []
    if condition == 'missing_core':
        engine.status['antelopev2'] = dict(state='unavailable')
    result = Pipeline(engine, store).run(np.zeros((150,150,3), np.uint8), 0, 0)
    assert ('petalface' in engine.calls) == expect_petal
    assert ('petalface' in result['overlays'][0]['model_scores']) == expect_petal
    assert not result['events']  # Routing cannot grant an uncalibrated identity.
    assert result['overlays'][0]['name'] is None


def test_real_batch_matches_single_model_outputs_and_keeps_invalid_face_isolated():
    import cv2
    from skimage import data
    image = cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2BGR)
    engine.load()
    points = engine.detect(image)[0][1]
    landmarks = [points+np.array([offset*.6,offset*.2],np.float32) for offset in range(5)]
    with_invalid = landmarks[:2]+[np.full((5,2),-1,np.float32)]+landmarks[2:]
    for key in ('lvface','antelopev2','petalface'):
        singles = [engine.embed(image,face,key) for face in landmarks]
        batches = engine.embed_many(image,with_invalid,key)
        assert len(batches) == 6 and isinstance(batches[2],ValueError)
        valid = batches[:2]+batches[3:]
        for (single,reference),(batched,actual) in zip(singles,valid):
            assert np.isfinite(batched).all() and abs(np.linalg.norm(batched)-1) < 1e-5
            assert np.dot(single,batched) > .99999
            assert np.allclose(single,batched,atol=3e-5,rtol=3e-4)
            assert abs(actual['alignment_error']-reference['alignment_error']) < 1e-5
            assert actual['version'] == engine.versions[key]
            assert actual['latency_basis'] == 'batch_amortized'
            if key == 'petalface':
                assert abs(actual['iqa_score']-reference['iqa_score']) < 1e-4
                assert abs(actual['alpha']-reference['alpha']) < 1e-4


def test_batch_retry_preserves_every_face_and_cpu_preprocessing_errors():
    from concurrent.futures import ThreadPoolExecutor
    from backend.models import batched_embeddings
    class LimitedBatch:
        batch_size = 4
        calls = []
        def prepare(self,image,points,strict_alignment):
            if points == -1:
                raise ValueError('unit invalid landmarks')
            return points,dict(alignment_error=points)
        def forward_batch(self,inputs):
            self.calls.append(list(inputs))
            if len(inputs) > 2:
                raise RuntimeError('unit batch limit')
            return [(np.eye(512,dtype=np.float32)[value],dict(tag=value)) for value in inputs]
    model = LimitedBatch()
    with ThreadPoolExecutor(max_workers=2) as executor:
        output = batched_embeddings(model,None,[0,1,-1,2,3,4,5],executor=executor)
    assert model.batch_size == 2 and len(output) == 7
    assert isinstance(output[2],ValueError)
    for result,index in zip(output[:2]+output[3:],range(6)):
        vector,info = result
        assert vector.argmax() == index and info['tag'] == index
    assert model.calls == [[0,1,2,3],[0,1],[2,3],[4,5]]


def test_inference_alignment_tolerates_only_subpixel_profile_corner_overlap():
    image=np.full((200,200,3),120,np.uint8)
    points=REFERENCE.copy()+30
    points[1,0]=points[0,0]-.5
    crop,error=align(image,points,strict=False)
    assert crop.shape==(112,112,3) and np.isfinite(error)
    with pytest.raises(ValueError,match='Sai thứ tự'):
        align(image,points,strict=True)
    points[1,0]=points[0,0]-1.01
    with pytest.raises(ValueError,match='Sai thứ tự'):
        align(image,points,strict=False)
    with pytest.raises(ValueError):
        align(image,REFERENCE[::-1],strict=False)


def test_diagnostic_examples_keep_bounded_original_bytes_only_during_measurement(tmp_path):
    store=Store(tmp_path);pid=store.add('Fixture',[(b'unit',np.ones(512,np.float32),{})])
    run=Diagnostic(store,'Chính diện 1 m',pid,True,True,clock=lambda:10.)
    raw=b'original JPEG bytes'
    frame=dict(frame_id=0,width=1280,height=720,overlays=[])
    run.retain_example(frame,10.,raw)
    assert not run.examples
    run.state='running';run.started=10.
    for index in range(300):
        frame['frame_id']=index
        run.retain_example(frame,10.+index/10,raw)
    assert len(run.examples)==256
    assert all(item['raw'] is raw for item in run.examples.values())
    previous=dict(run.examples)
    run.retain_example(frame,40.,raw)
    run.retain_example(frame,12.,b'x'*(2*1024**2+1))
    run.state='complete';run.retain_example(frame,12.,raw)
    assert run.examples==previous
    with store.db() as db:
        assert db.execute('SELECT count(*) FROM diagnostic_frames').fetchone()[0]==0


def test_diagnostic_retains_adjacent_actual_frames_when_a_name_disappears(tmp_path):
    store=Store(tmp_path);person=store.add('Fixture',[(b'unit',np.ones(512,np.float32),{})])
    run=Diagnostic(store,'Chính diện 3 m',person,True,True,clock=lambda:10.)
    run.state='running';run.started=10.
    for index in range(100):
        status='confirmed' if index%3==0 else 'unknown'
        frame=dict(frame_id=index,width=1280,height=720,overlays=[dict(status=status,person_id=person if status=='confirmed' else None,model_scores={})])
        run.retain_example(frame,10.+index*.2,b'frame '+str(index).encode())
        extra=run.previous_named_example
        unique={item['frame_id'] for item in run.examples.values()}
        if extra:unique.add(extra['frame_id'])
        assert len(unique)<=256
    frames={item['frame_id'] for item in run.examples.values()}
    assert 96 in frames and 97 in frames
    assert all(item['raw']==b'frame '+str(item['frame_id']).encode() for item in run.examples.values())
    run.finish();assert run.previous_named_example is None



def test_native_14_pixel_inference_gate_preserves_enrollment_and_original_input(tmp_path):
    image = np.random.default_rng(19).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    original = image.copy()
    settings = Store(tmp_path).settings()
    engine = Engine()
    for pixels in (14, 19, 20, 28, 112):
        box = np.array([0, 0, pixels, pixels, .99])
        quality = engine.quality(image, box, settings, inference=True)
        assert quality['face_pixels'] == pixels
        assert quality['original_face_size'] == [pixels, pixels]
        assert np.array_equal(image, original)
        if pixels < 28:
            with pytest.raises(ValueError, match='Mặt quá nhỏ'):
                engine.quality(image, box, settings)
    with pytest.raises(ValueError, match='Mặt quá nhỏ'):
        engine.quality(image, np.array([0, 0, 13, 13, .99]), settings, inference=True)
    with pytest.raises(ValueError, match='Mặt mờ'):
        engine.quality(np.full_like(image, 120), np.array([0, 0, 14, 14, .99]), settings, inference=True)
    settings['min_inference_face_pixels'] = 20
    with pytest.raises(ValueError, match='Mặt quá nhỏ'):
        engine.quality(image, np.array([0, 0, 19, 19, .99]), settings, inference=True)
