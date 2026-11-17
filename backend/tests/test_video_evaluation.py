"""Isolated metric fixtures: never seed the application's enrollment database."""
import json
import sqlite3
from types import SimpleNamespace
import pytest
from backend.store import Store
from backend.diagnostics import LIVE_CASES, Diagnostic, live_coverage
from backend.video_evaluation import validate_annotations, save_truth, read_truth, evaluate, truth_at, save_measurement


def test_solo_protocol_covers_every_distance_and_separates_mask(tmp_path):
    st = Store(tmp_path)
    assert len(LIVE_CASES) == 23
    for d in (1,3,6,9):
        cases = [c for c in LIVE_CASES if c['distance_m'] == d]
        assert len(cases) == (5 if d==1 else 6)
        assert any(c['condition'] == 'Đeo khẩu trang' for c in cases)
        for c in cases:
            run = Diagnostic(st, c['scenario'], None, False, True)
            assert run.context['single_person'] and run.context['distance_m'] == d
            run.finish()
    assert not any(c['measured'] for c in live_coverage(st))  # Empty/cancelled runs are not measured.


def test_interval_labels_require_explicit_truth_and_survive_new_job(tmp_path):
    enrollment = [dict(person_id='a', name='A', photo_count=1, image_ids=['one'], selected=True),
                  dict(person_id='b', name='B', photo_count=1, image_ids=['two'], selected=True)]
    st = Store(tmp_path)
    job = SimpleNamespace(store=st,source_hash='test-source',filename='unit.mp4',duration=10.,enrollment=enrollment)
    labels = save_truth(job,[dict(start=0,end=4,kind='present',person_id='a',different_source=True),
                            dict(start=4,end=8,kind='outside')])
    assert truth_at(labels,'a',2) is True
    assert truth_at(labels,'b',2) is None  # A is visible: cannot infer B is absent.
    assert truth_at(labels,'a',4) is False
    assert truth_at(labels,'b',9) is None
    assert read_truth(SimpleNamespace(store=Store(tmp_path),source_hash='test-source')) == labels
    with pytest.raises(ValueError,match='mâu thuẫn'):
        validate_annotations(labels + [dict(start=3,end=5,kind='outside')],10,enrollment)
    for start,end in [(0,11),(-1,2),(2,2),(float('nan'),3)]:
        with pytest.raises(ValueError):validate_annotations([dict(start=start,end=end,kind='outside')],10,enrollment)


def test_video_evaluation_never_uses_top1_as_ground_truth_or_claims_face_accuracy(tmp_path):
    st=Store(tmp_path)
    enrollment=[dict(person_id='a',name='A',selected=True,photo_count=1,image_ids=['unit'])]
    path=tmp_path/'unit-frames.sqlite3'
    job=SimpleNamespace(store=st,db_path=path,source_hash='fixture',filename='unit.mp4',duration=10.,
                        enrollment=enrollment,id='unit-job',state='complete')
    save_truth(job,[dict(start=0,end=3,kind='present',person_id='a'),dict(start=3,end=6,kind='outside')])
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE frames(id INTEGER, timestamp REAL, result TEXT)')
        for i,t in enumerate((1.,4.,8.)):
            raw=dict(frame_id=i,timestamp=t,processing_ms=100,overlays=[dict(track_id=i,status='confirmed',person_id='a',
                model_scores={'lvface':dict(person_id='a',score=.99,calibrated=False,version='unit-version')})],
                events=[dict(id=str(i),person_id='a',timestamp=t)])
            db.execute('INSERT INTO frames VALUES (?,?,?)',(i,t,json.dumps(raw)))
            save_measurement(st,job.id,dict(raw,image='do not retain',events=[dict(raw['events'][0],evidence='do not retain')]))
    result=evaluate(job)
    assert result['alerts'] == dict(consistent_with_interval=1,false_alerts=1,unverified=1)
    assert result['alert_identity_verification'] == dict(correct=0,wrong=1,unverified=2)
    assert result['confirmed_during_declared_absence'] == 1
    assert result['detection_recall'] is result['identity_switches'] is result['face_identity_accuracy'] is None
    assert result['models']['lvface']['identity_accuracy'] is None
    assert result['models']['lvface']['declared_absence_samples']==1
    assert result['models']['lvface']['presence_interval_samples']==1
    assert result['models']['lvface']['unlabeled_samples']==1
    assert result['intervals'][0]['time_to_confirmation_seconds'] == 1
    job.db_path=tmp_path/'closed-playback-cache.sqlite3'
    retained=evaluate(job)
    assert retained['frames']==3 and retained['alerts']==result['alerts']
    with st.db() as db:
        assert all('do not retain' not in r[0] for r in db.execute('SELECT metrics FROM video_measurements'))
        assert db.execute('SELECT count(*) FROM calibrations').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM persons').fetchone()[0] == 0


def test_video_face_labels_are_human_truth_and_bound_to_exact_frame(tmp_path):
    import base64
    import cv2
    import numpy as np
    from backend.video_evaluation import save_face_truth, read_face_truth, annotation_faces
    store = Store(tmp_path/'store')
    enrollment = [dict(person_id='a',name='Unit A',selected=True,photo_count=1,image_ids=['photo-a']),
                  dict(person_id='b',name='Unit B',selected=True,photo_count=1,image_ids=['photo-b'])]
    video_path = tmp_path/'unit-labels.mp4'
    writer = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for color in (40,120,200):
        writer.write(np.full((32,32,3),color,np.uint8))
    writer.release()
    job = SimpleNamespace(store=store,id='unit-labelled',source_hash='unit-source',path=video_path,
        filename='unit-labels.mp4',duration=.3,fps=10,enrollment=enrollment,state='complete',
        db_path=tmp_path/'missing-playback.sqlite3')
    original = video_path.read_bytes()
    for index in range(3):
        result = dict(frame_id=index,timestamp=index/10,width=32,height=32,processing_ms=10,
            overlays=[dict(bbox=[4.,4.,24.,24.],track_id=1,status='confirmed',person_id='a',
                model_scores={'lvface':dict(person_id='a',score=.99,passed=True,calibrated=False)})],
            events=[dict(id='label-event-'+str(index),person_id='a',timestamp=index/10,track_id=1)])
        save_measurement(store,job.id,result)
    # Ground truth B overrides the model's very high-scoring A prediction.
    labels = save_face_truth(job,0,[dict(face_index=0,kind='person',person_id='b',different_source=True)])
    save_face_truth(job,1,[dict(face_index=0,kind='outside')])
    response = annotation_faces(job,1)
    assert response['frame_id'] == 1 and response['timestamp'] == .1
    assert response['faces'][0]['bbox'] == [4.,4.,24.,24.]
    encoded = base64.b64decode(response['faces'][0]['thumbnail'].split(',')[1])
    crop = cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
    assert crop.shape == (20,20,3) and 110 < crop.mean() < 125
    assert 'person_id' not in response['faces'][0]  # UI never preselects a model guess.
    assert read_face_truth(job,0) == labels
    evaluated = evaluate(job)
    truth = evaluated['face_ground_truth']
    assert truth['labeled_faces'] == 2 and truth['known_faces'] == truth['outside_faces'] == 1
    assert truth['confirmed_correct'] == 0 and truth['false_accepts'] == 2
    assert evaluated['alert_identity_verification'] == dict(correct=0,wrong=2,unverified=1)
    assert truth['missed_known_faces'] == 1 and truth['different_source_known_faces'] == 1
    assert truth['models']['lvface']['known_top1_wrong'] == 1
    assert truth['models']['lvface']['outside_reference_passes'] == 1
    assert truth['models']['lvface']['outside_calibrated_passes'] == 0
    assert evaluated['face_identity_accuracy'] is None and evaluated['frames'] == 3
    assert video_path.read_bytes() == original
    with store.db() as db:
        assert db.execute('SELECT count(*) FROM persons').fetchone()[0] == 0
    for index in (-1,True,.5,4):
        with pytest.raises(ValueError):
            save_face_truth(job,0,[dict(face_index=index,kind='outside')])
    with pytest.raises(ValueError):
        save_face_truth(job,0,[dict(face_index=0,kind='person',person_id='missing')])
    save_truth(job,[dict(start=.2,end=.3,kind='absent',person_id='b')])
    with pytest.raises(ValueError,match='mâu thuẫn'):
        save_face_truth(job,2,[dict(face_index=0,kind='person',person_id='b')])
    assert read_face_truth(job,0) == labels  # Rejected changes leave saved labels intact.
    store = Store(tmp_path/'store')
    job.store = store
    assert len(read_face_truth(job)) == 2
