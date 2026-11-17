"""Regression for face movement larger than the observed-box overlap gate."""
import threading
import numpy as np
from backend.ai import Pipeline
from backend.store import Store


def test_motion_association_keeps_track_without_drawing_prediction_or_carrying_identity(tmp_path):
    store = Store(tmp_path)
    first, second = np.eye(512, dtype=np.float32)[:2]
    versions = {'lvface': 'motion-lv', 'antelopev2': 'motion-ant'}
    person_a = store.add('Unit A', [(b'unit fixture', {v: first for v in versions.values()}, {})])
    store.add('Unit B', [(b'unit fixture', {v: second for v in versions.values()}, {})])
    for version in versions.values():
        store.calibrate(version, 'normal', dict(threshold=.8, margin=.08))

    class ScriptedEngine:
        lock = threading.RLock()
        status = {key: dict(state='ready') for key in versions}
        vector = first
        left = 0
        def load(self): pass
        def detect(self, image):
            return [(np.array([self.left, 20, self.left+100, 120, .99]), None, 'unit detector')]
        def quality(self, *args, **kwargs): return dict(face_pixels=100, sharpness=100)
        def embed(self, image, points, key, **kwargs): return self.vector, {}
        def gpu(self): return dict(device='unit')

    engine = ScriptedEngine()
    engine.versions = versions
    pipeline = Pipeline(engine, store)
    image = np.zeros((180, 800, 3), dtype=np.uint8)
    results = []
    for frame_id, left in enumerate((0, 45, 120, 215, 320)):
        engine.left = left
        result = pipeline.run(image, frame_id, frame_id*.25)
        results.append(result)
        assert result['overlays'][0]['bbox'] == [left, 20, left+100, 120]
    assert len({r['overlays'][0]['track_id'] for r in results}) == 1
    assert sum(len(r['events']) for r in results) == 1
    assert results[-1]['overlays'][0]['person_id'] == person_a
    assert results[-1]['overlays'][0]['tracking']['method'] == 'motion'
    assert results[-1]['overlays'][0]['tracking']['previous_iou'] == 0
    assert results[-1]['overlays'][0]['tracking']['appearance_similarity'] > .99

    for frame_id, left in enumerate((205, 330, 210, 335), start=5):
        engine.left = left
        result = pipeline.run(image, frame_id, frame_id*.25)
        assert result['overlays'][0]['track_id'] == results[-1]['overlays'][0]['track_id']
        assert not result['events']
        assert result['overlays'][0]['bbox'] == [left, 20, left+100, 120]
    assert result['overlays'][0]['tracking']['method'] == 'appearance'

    pipeline = Pipeline(engine, store)
    ghost_results = []
    for frame_id, left in enumerate((0,45,400,225,235)):
        engine.left = left
        ghost_results.append(pipeline.run(image,frame_id,frame_id*.25))
    assert ghost_results[1]['overlays'][0]['track_id'] != ghost_results[2]['overlays'][0]['track_id']
    assert len({r['overlays'][0]['track_id'] for r in ghost_results[2:]}) == 1

    engine.left = 340
    engine.vector = second
    changed = pipeline.run(image, 5, 1.25)
    assert not changed['events']
    assert changed['overlays'][0]['track_id'] != ghost_results[-1]['overlays'][0]['track_id']
    assert changed['overlays'][0]['status'] == 'candidate'
    assert changed['overlays'][0]['name'] is None


def test_weak_appearance_geometry_requires_fresh_biometric_evidence(tmp_path):
    store = Store(tmp_path)
    first, second, noise_a, noise_b = np.eye(512, dtype=np.float32)[:4]
    versions = {key:'unit-geometry-'+key for key in ('lvface','antelopev2','petalface')}
    target = store.add('Unit target', [(b'unit fixture', {v:first for v in versions.values()}, {})])
    store.add('Unit other', [(b'unit fixture', {v:second for v in versions.values()}, {})])
    for version in versions.values():
        store.calibrate(version, 'normal', dict(threshold=.8, margin=.08, iqa_threshold=.5))

    class ScriptedEngine:
        lock = threading.RLock()
        status = {key:dict(state='ready') for key in versions}
        weak = False
        replacement = False
        verification = True
        noise = noise_a
        faces = [np.array([10,10,110,110,.99])]
        def load(self): pass
        def detect(self, image): return [(box,None,'unit detector') for box in self.faces]
        def quality(self, *args, **kwargs): return dict(face_pixels=100, sharpness=100)
        def embed(self, image, points, key, **kwargs):
            vector = second if self.replacement else first
            if self.weak and (key == 'lvface' or not self.verification):
                vector = .3*first+np.sqrt(1-.3**2)*self.noise
            return vector, dict(iqa_threshold=.5,alignment_band='regular')
        def gpu(self): return dict(device='unit')

    engine = ScriptedEngine()
    engine.versions = versions
    pipeline = Pipeline(engine, store)
    image = np.zeros((150,150,3), np.uint8)
    initial = [pipeline.run(image,i,i*.2) for i in range(3)]
    track = initial[-1]['overlays'][0]['track_id']
    assert initial[-1]['overlays'][0]['person_id'] == target
    assert sum(len(r['events']) for r in initial) == 1

    engine.weak = True
    recovered = []
    for i in range(3,6):
        engine.noise = noise_a if i % 2 else noise_b
        result = pipeline.run(image,i,i*.2)
        recovered.append(result)
        face = result['overlays'][0]
        assert face['track_id'] == track
        assert face['bbox'] == [10,10,110,110]
        assert face['tracking']['method'] == 'geometry'
        assert not result['events']
    assert recovered[0]['overlays'][0]['tracking']['identity_evidence_reset']
    assert recovered[0]['overlays'][0]['name'] is None
    assert recovered[0]['overlays'][0]['status'] == 'candidate'
    assert not recovered[1]['overlays'][0]['tracking']['identity_evidence_reset']
    assert recovered[1]['overlays'][0]['status'] == 'candidate'
    assert recovered[2]['overlays'][0]['person_id'] == target  # Three new calibrated independent branch votes.

    engine.verification = False
    engine.noise = noise_b
    unsupported = pipeline.run(image,6,1.2)
    assert unsupported['overlays'][0]['track_id'] == track
    assert unsupported['overlays'][0]['name'] is None
    assert unsupported['overlays'][0]['status'] != 'confirmed'
    assert not unsupported['events']

    # Two overlapping observations cannot share the same geometry-only predecessor.
    engine.faces = [engine.faces[0],engine.faces[0].copy()]
    engine.noise = noise_a
    ambiguous = pipeline.run(image,7,1.4)
    assert all(face['track_id'] != track for face in ambiguous['overlays'])
    assert not ambiguous['events'] and all(face['name'] is None for face in ambiguous['overlays'])

    engine.faces = engine.faces[:1]
    engine.weak = False
    engine.replacement = True
    replacement = pipeline.run(image,8,1.6)
    assert replacement['overlays'][0]['status'] == 'candidate'
    assert replacement['overlays'][0]['person_id'] is None
    assert not replacement['events']


def test_camera_motion_timestamp_rejects_invalid_nonmonotonic_values():
    import pytest
    from backend.sessions import camera_timestamp
    assert camera_timestamp(0)==0
    assert camera_timestamp(99.575313,99.241538)==99.575313
    for value in (None,True,'1',float('nan'),float('inf'),-1):
        with pytest.raises(ValueError):camera_timestamp(value)
    with pytest.raises(ValueError):camera_timestamp(1,1)
    with pytest.raises(ValueError):camera_timestamp(.9,1)


def test_actual_motion_geometry_uses_capture_intervals_not_transport_jitter():
    # Three observed boxes from the 3 m run, no inferred identity labels.
    from backend.ai import predicted_face_box,iou
    boxes=np.array([[422.65109,363.28046,451.74039,411.50613],
                    [383.10989,363.03763,412.98645,415.23230],
                    [283.80060,364.66382,320.55115,419.95081]],dtype=np.float32)
    capture=[99.108905,99.241538,99.575313]
    received=[89.031,89.313,89.516]
    def forecast(times):
        centers=(boxes[:,:2]+boxes[:,2:])*.5
        velocity=(centers[1]-centers[0])/(times[1]-times[0])
        return predicted_face_box(dict(box=boxes[1],time=times[1],velocity=velocity),times[2])
    assert iou(boxes[2],forecast(capture))>.75
    assert iou(boxes[2],forecast(received))<.1
    # Prediction is association geometry; the observed detection must not change.
    assert np.array_equal(boxes[2],np.array([283.80060,364.66382,320.55115,419.95081],np.float32))
