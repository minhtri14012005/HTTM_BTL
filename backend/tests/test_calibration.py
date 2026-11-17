"""Calibration provenance and temporal rejection tests; isolated SQLite only."""
import copy
import json
import pytest
from backend.calibration import collect_samples, fit_profiles, apply_profiles
from backend.store import Store
from backend.temporal import Temporal

VERSIONS = dict(lvface='lvface:unit', antelopev2='antelopev2:unit')
REVIEW = dict(only_target=True,face_visible_throughout=True,boxes_match_target=True)


def face(pid='a',value=.8,margin=.6):
    return dict(track_id=1,face_bbox=[10,10,110,110],quality=dict(face_pixels=100),
        model_scores={key:dict(person_id=pid,score=value,margin=margin,version=version,
                              alignment_band='regular') for key,version in VERSIONS.items()})


def labelled_store(tmp_path):
    store = Store(tmp_path)
    with store.db() as db:
        db.execute("INSERT INTO diagnostics(id,scenario,person_id,expected_present,state) VALUES ('live','unit','a',1,'complete')")
        db.execute('INSERT INTO diagnostic_ground_truth(run_id,details) VALUES (?,?)',('live',json.dumps(REVIEW)))
        for index in range(8):
            db.execute('INSERT INTO diagnostic_frames VALUES (?,?,?,?)',('live',index,index,json.dumps(dict(overlays=[face()]))))
        db.execute('INSERT INTO video_evaluations(run_id,source_hash,result) VALUES (?,?,?)',('video','source',json.dumps(dict(state='complete'))))
        db.execute('INSERT INTO video_ground_truth(source_hash,filename,duration,annotations) VALUES (?,?,?,?)',
                   ('source','unit.mp4',4,json.dumps([dict(start=0,end=1,kind='outside',scope=['a','b']),dict(start=1,end=4,kind='present',scope=['b'])])))
        for index,current in enumerate((face(value=.3,margin=.1),face('b',.85,.65),face('b',.9,.7))):
            db.execute('INSERT INTO video_measurements VALUES (?,?,?,?)',('video',index,index,json.dumps(dict(overlays=[current]))))
        db.execute('INSERT INTO video_face_ground_truth(run_id,source_hash,frame_id,face_index,details) VALUES (?,?,?,?,?)',
                   ('video','source',1,0,json.dumps(dict(kind='person',person_id='b',bbox=[10,10,110,110],different_source=True))))
    return store


def test_only_reviewed_live_fits_and_video_is_independent_validation(tmp_path):
    store = labelled_store(tmp_path)
    generation,samples = collect_samples(store,VERSIONS)
    assert generation == 0
    assert all(s['fold']=='fit' for s in samples if s['source'].startswith('live:'))
    assert all(s['fold']=='validation' for s in samples if s['source'].startswith('video:'))
    # Presence-only frame 2 does not label any individual face.
    assert not any(s['time']==2 and s['positive'] for s in samples if s['source'].startswith('video:'))
    result = fit_profiles(store,VERSIONS)
    assert len(result['profiles'])==2
    for profile in result['profiles']:
        config = profile['config']
        assert config['fit_positive_blocks']==8
        assert config['validation_positive_observations']==1
        assert config['validation_positive_passes']==1
        assert config['validation_negative_observations']>=1
        assert config['validation_false_passes']==0
        assert config['positive_subjects']==['a','b']


def test_unknown_or_wrong_top1_never_becomes_a_positive_label(tmp_path):
    store=labelled_store(tmp_path)
    with store.db() as db:
        db.execute('UPDATE diagnostic_ground_truth SET details=?',(json.dumps(dict(REVIEW,boxes_match_target=False)),))
    _,samples=collect_samples(store,VERSIONS)
    assert not any(s['source'].startswith('live:') for s in samples)
    assert fit_profiles(store,VERSIONS)['profiles']==[]
    with store.db() as db:
        db.execute('UPDATE diagnostic_ground_truth SET details=?',(json.dumps(REVIEW),))
        db.execute('UPDATE diagnostic_frames SET metrics=? WHERE frame_id=0',(json.dumps(dict(overlays=[face('b',.95,.7)])),))
    _,samples=collect_samples(store,VERSIONS)
    wrong=[s for s in samples if s['source']=='live:live' and s['time']==0]
    assert all(s['score'] is None for s in wrong if s['positive'])
    assert all(s['score']==.95 for s in wrong if not s['positive'])
    assert fit_profiles(store,VERSIONS)['profiles']==[]


def test_profiles_bind_version_quality_and_label_generation_and_apply_atomically(tmp_path):
    store=labelled_store(tmp_path); result=fit_profiles(store,VERSIONS)
    settings=apply_profiles(store,result,VERSIONS)
    assert settings['calibrated'] and settings['calibration_revision']==1
    assert store.calibration(VERSIONS['lvface'],100)['threshold']==pytest.approx(.8)
    assert store.calibration(VERSIONS['lvface'],40) is None
    assert store.calibration(VERSIONS['lvface'],100,'high_residual') is None
    assert not store.settings({'lvface':VERSIONS['lvface']})['calibrated']
    assert not store.settings({'lvface':'different-version'})['calibrated']
    with store.db() as db:
        before=[tuple(row) for row in db.execute('SELECT * FROM calibrations ORDER BY model')]
    bad=copy.deepcopy(result);bad['profiles'][1]['model']='wrong-version'
    with pytest.raises(ValueError):apply_profiles(store,bad,VERSIONS)
    with store.db() as db:
        assert before==[tuple(row) for row in db.execute('SELECT * FROM calibrations ORDER BY model')]
        store.labels_changed(db)
    assert store.calibration(VERSIONS['lvface'],100) is None
    assert not store.settings(VERSIONS)['calibrated']
    with pytest.raises(ValueError):apply_profiles(store,result,VERSIONS)


def test_independent_impostor_above_fitted_gate_disables_profile(tmp_path):
    store=labelled_store(tmp_path)
    with store.db() as db:
        db.execute('UPDATE video_measurements SET metrics=? WHERE frame_id=0',(json.dumps(dict(overlays=[face(value=.99,margin=.7)])),))
    result=fit_profiles(store,VERSIONS)
    assert result['profiles']==[]
    assert all(item['reason']=='validation_impostor_passes' for item in result['unavailable'])


def test_weak_frame_has_no_identity_but_recent_votes_survive_on_same_track():
    state=Temporal();settings=dict(confirm_frames=3,absence_seconds=3)
    assert state.update(1,'a',0,settings,frame_id=1)[0]=='candidate'
    assert state.update(1,None,.2,settings,frame_id=2)==('unknown',False,None)
    assert state.update(1,'a',.4,settings,frame_id=3)[0]=='candidate'
    assert state.update(1,None,.6,settings,frame_id=4)==('unknown',False,None)
    assert state.update(1,'a',.8,settings,frame_id=5)==('confirmed',True,1)
    assert state.update(1,None,1.,settings,frame_id=6)==('unknown',False,None)
    assert state.update(1,'a',1.2,settings,frame_id=7)==('confirmed',False,1)


def test_conflict_gap_and_duplicate_frames_require_new_evidence():
    state=Temporal();settings=dict(confirm_frames=3,absence_seconds=3)
    state.update(1,'a',0,settings,frame_id=1)
    state.update(1,'a',.2,settings,frame_id=2)
    state.update(1,'a',.3,settings,frame_id=2)
    assert state.tracks[1]['hits']==2
    assert state.update(1,None,.4,settings,reset=True)==('unknown',False,None)
    assert state.update(1,'a',.6,settings)[0]=='candidate'
    assert state.tracks[1]['hits']==1
    assert state.update(1,'a',3.,settings)[0]=='candidate'
    assert state.tracks[1]['hits']==1
    state.update(1,'a',3.2,settings)
    assert state.update(1,'b',3.4,settings)[0]=='candidate'
    assert state.tracks[1]['pid']=='b' and state.tracks[1]['hits']==1

def test_score_replay_preserves_original_measurements_and_does_not_invent_ground_truth(tmp_path):
    from backend.calibration import replay_diagnostic
    store=labelled_store(tmp_path);apply_profiles(store,fit_profiles(store,VERSIONS),VERSIONS)
    # Select the fixture identity using an isolated record, never production people.
    with store.db() as db:
        db.execute("INSERT INTO persons(id,name,selected) VALUES ('a','Fixture',1)")
        original=[tuple(row) for row in db.execute('SELECT * FROM diagnostic_frames ORDER BY frame_id')]
        run=dict(db.execute("SELECT * FROM diagnostics WHERE id='live'").fetchone())
        rows=[(row[2],json.loads(row[3])) for row in original]
    result=replay_diagnostic(store,run,rows,REVIEW)
    assert result['confirmed_faces']==6 and result['correct_confirmed']==6
    assert result['wrong_confirmed']==0 and result['alerts']==1
    assert result['used_for_fitting'] and result['identity_verified']
    unreviewed=replay_diagnostic(store,run,rows,{})
    assert not unreviewed['identity_verified'] and unreviewed['correct_confirmed'] is None
    with store.db() as db:
        assert original==[tuple(row) for row in db.execute('SELECT * FROM diagnostic_frames ORDER BY frame_id')]


def test_saving_same_truth_does_not_invalidate_but_changed_truth_does(tmp_path):
    from types import SimpleNamespace
    from backend.video_evaluation import save_truth
    store=labelled_store(tmp_path)
    job=SimpleNamespace(store=store,source_hash='source',filename='unit.mp4',duration=4,
                        enrollment=[dict(person_id='a',selected=True),dict(person_id='b',selected=True)])
    annotations=[dict(start=0,end=1,kind='outside')]
    save_truth(job,annotations)
    generation=store.settings()['calibration_generation']
    save_truth(job,annotations)
    assert store.settings()['calibration_generation']==generation
    save_truth(job,[dict(start=0,end=2,kind='outside')])
    assert store.settings()['calibration_generation']==generation+1


def test_reuse_labels_requires_exact_source_frame_time_and_box_without_refitting(tmp_path):
    from types import SimpleNamespace
    from backend.video_evaluation import reuse_face_truth
    store=labelled_store(tmp_path)
    job=SimpleNamespace(store=store,id='new-video',source_hash='source',enrollment=[dict(person_id='a'),dict(person_id='b')])
    with store.db() as db:
        rows=db.execute("SELECT frame_id,timestamp,metrics FROM video_measurements WHERE run_id='video'").fetchall()
        for row in rows:
            db.execute('INSERT INTO video_measurements VALUES (?,?,?,?)',(job.id,*row))
    result=reuse_face_truth(job)
    assert result['copied']==1 and result['labels'][0]['person_id']=='b'
    assert result['labels'][0]['source']=='user_label_reused_exact_frame'
    assert store.settings().get('calibration_generation',0)==0
    assert reuse_face_truth(job)['copied']==0
    shifted=SimpleNamespace(store=store,id='shifted',source_hash='source',enrollment=job.enrollment)
    with store.db() as db:
        changed=face('b',.85,.65);changed['face_bbox']=[11,10,111,110]
        db.execute('INSERT INTO video_measurements VALUES (?,?,?,?)',(shifted.id,1,1,json.dumps(dict(overlays=[changed]))))
    assert reuse_face_truth(shifted)['copied']==0
    shifted.source_hash='different-source'
    assert reuse_face_truth(shifted)['copied']==0


def test_new_live_verification_is_never_silently_added_to_fitting_data(tmp_path):
    store=labelled_store(tmp_path)
    with store.db() as db:
        db.execute("INSERT INTO diagnostics(id,scenario,person_id,expected_present,state,context) VALUES ('future','unit','a',1,'complete',?)",
                   (json.dumps(dict(calibration_role='validation',calibration_revision=1)),))
        db.execute('INSERT INTO diagnostic_ground_truth(run_id,details) VALUES (?,?)',('future',json.dumps(REVIEW)))
        db.execute('INSERT INTO diagnostic_frames VALUES (?,?,?,?)',('future',1,0,json.dumps(dict(overlays=[face(value=.85,margin=.65)]))))
    _,samples=collect_samples(store,VERSIONS)
    assert all(sample['fold']=='validation' for sample in samples if sample['source']=='live:future')
    profiles=fit_profiles(store,VERSIONS)['profiles']
    assert len(profiles)==2
    assert all('live:future' not in profile['config']['fit_sources'] for profile in profiles)
    assert all('live:future' in profile['config']['validation_sources'] for profile in profiles)


def test_temporal_bridge_cannot_initialize_extend_forever_or_carry_another_identity():
    state=Temporal();settings=dict(confirm_frames=3,absence_seconds=3)
    assert not state.can_bridge(1,'a',0,settings)
    for index in range(3): state.update(1,'a',index*.1,settings,frame_id=index)
    assert state.can_bridge(1,'a',.3,settings)
    assert not state.can_bridge(1,'b',.3,settings)
    assert not state.can_bridge(2,'a',.3,settings)
    assert state.update(1,'a',.3,settings,multimodel=False)[0]=='confirmed'
    assert state.can_bridge(1,'a',.6,settings)
    state.update(1,'a',.6,settings,multimodel=False)
    assert not state.can_bridge(1,'a',.71,settings)
    assert state.update(1,None,.71,settings)==('unknown',False,None)
    state.update(1,None,.8,settings,reset=True)
    assert not state.can_bridge(1,'a',.9,settings)


def test_diagnostic_distances_have_no_six_or_ten_meter_ceiling():
    from backend.diagnostics import scenario_details
    for distance in (1,3,6,8,12.5,25,100):
        case=scenario_details(f'Chính diện {distance} m')
        assert case['distance_m']==distance
    assert scenario_details('Chính diện 0 m')=={}
    assert scenario_details('Chính diện -1 m')=={}
    assert scenario_details('Chính diện nan m')=={}


def test_distant_diagnostic_and_explicit_calibration_purpose(tmp_path):
    from backend.diagnostics import Diagnostic
    store=labelled_store(tmp_path);apply_profiles(store,fit_profiles(store,VERSIONS),VERSIONS)
    with store.db() as db:db.execute("INSERT INTO persons(id,name,selected) VALUES ('a','Fixture',1)")
    fit=Diagnostic(store,'Quay trái/phải 12.5 m','a',True,True,purpose='calibration')
    assert fit.context['distance_m']==12.5 and fit.context['calibration_role']=='fit'
    validation=Diagnostic(store,'Chính diện 25 m','a',True,True)
    assert validation.context['distance_m']==25 and validation.context['calibration_role']=='validation'
    with pytest.raises(ValueError):Diagnostic(store,'Chính diện 0 m','a',True,True)


def test_quality_gate_uses_image_evidence_and_not_distance_metadata():
    import numpy as np
    from backend.ai import Engine
    image=np.random.default_rng(4).integers(0,256,(80,80,3),dtype=np.uint8)
    box=np.array([10,10,60,60,.99]);settings=dict(min_face_pixels=28,min_inference_face_pixels=20,min_sharpness=18.)
    engine=Engine()
    assert engine.quality(image,box,dict(settings,distance_m=1),inference=True)==engine.quality(image,box,dict(settings,distance_m=1000),inference=True)


def test_new_genuine_distribution_does_not_tighten_previously_safe_gates(tmp_path):
    store=labelled_store(tmp_path);apply_profiles(store,fit_profiles(store,VERSIONS),VERSIONS)
    with store.db() as db:
        db.execute("INSERT INTO diagnostics(id,scenario,person_id,expected_present,state) VALUES ('new-fit','unit','a',1,'complete')")
        db.execute('INSERT INTO diagnostic_ground_truth(run_id,details) VALUES (?,?)',('new-fit',json.dumps(REVIEW)))
        for index in range(90):
            db.execute('INSERT INTO diagnostic_frames VALUES (?,?,?,?)',('new-fit',index,index,json.dumps(dict(overlays=[face(value=.95,margin=.8)]))))
        store.labels_changed(db)
    result=fit_profiles(store,VERSIONS)
    assert len(result['profiles'])==2
    assert all(p['config']['threshold']==pytest.approx(.8) and p['config']['margin']==pytest.approx(.6) for p in result['profiles'])
    assert all(p['config']['threshold_origin']=='retained_after_new_label_revalidation' for p in result['profiles'])
    # A new human-labelled wrong top-1 above the old gate must invalidate that gate.
    with store.db() as db:
        db.execute("UPDATE diagnostic_frames SET metrics=? WHERE run_id='new-fit' AND frame_id=0",(json.dumps(dict(overlays=[face('b',.9,.7)])),))
    result=fit_profiles(store,VERSIONS)
    assert len(result['profiles'])==2
    assert all(p['config']['threshold']>.9 for p in result['profiles'])


def test_high_residual_profile_checks_impostors_from_other_pixel_bands(tmp_path):
    store=labelled_store(tmp_path)
    with store.db() as db:
        for row in db.execute("SELECT frame_id,metrics FROM diagnostic_frames WHERE run_id='live'").fetchall():
            f=json.loads(row['metrics'])
            for score in f['overlays'][0]['model_scores'].values():score['alignment_band']='high_residual'
            db.execute("UPDATE diagnostic_frames SET metrics=? WHERE run_id='live' AND frame_id=?",(json.dumps(f),row['frame_id']))
        impostor=face(value=.99,margin=.7);impostor['quality']['face_pixels']=24;impostor['face_bbox']=[10,10,34,34]
        for score in impostor['model_scores'].values():score['alignment_band']='high_residual'
        db.execute("UPDATE video_measurements SET metrics=? WHERE run_id='video' AND frame_id=0",(json.dumps(dict(overlays=[impostor])),))
    result=fit_profiles(store,VERSIONS)
    assert not result['profiles']
    assert any(p['reason']=='cross_quality_impostor_passes' for p in result['unavailable'])


def test_labelled_negative_blocks_profile_even_when_its_ranking_margin_is_low(tmp_path):
    store=labelled_store(tmp_path)
    # A low ranking margin cannot excuse an outsider above the absolute threshold.
    with store.db() as db:
        entry=face(None,.85,.08)
        for key in entry['model_scores']:entry['model_scores'][key]['person_id']='a'
        db.execute('UPDATE video_measurements SET metrics=? WHERE run_id=? AND frame_id=0',
                   (json.dumps(dict(overlays=[entry])),'video'))
    result=fit_profiles(store,VERSIONS)
    assert not result['profiles']
    assert all(item['reason']=='validation_impostor_passes' for item in result['unavailable'])


def test_full_model_probe_is_validation_only_and_preserves_face_ground_truth(tmp_path):
    store=labelled_store(tmp_path);version=VERSIONS['lvface']
    with store.db() as db:
        db.execute('INSERT INTO model_probes VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)',('probe',version,'source','all_native_frames_no_fusion',json.dumps(dict(state='complete',frames=2,total=2))))
        first=face(value=.3,margin=.1)
        second=face('b',.85,.65);second['ground_truth']=dict(kind='person',person_id='b',bbox=second['face_bbox'],source='user_label_reused_exact_frame')
        for index,current in enumerate((first,second)):
            db.execute('INSERT INTO video_measurements VALUES (?,?,?,?)',('probe',index,index,json.dumps(dict(overlays=[current]))))
    _,samples=collect_samples(store,VERSIONS)
    probe=[s for s in samples if s['source'].startswith('probe:')]
    assert len(probe)==3 and all(s['fold']=='validation' for s in probe)
    assert all(s['version']==version for s in probe)
    assert sum(s['positive'] for s in probe)==1
    with store.db() as db:
        second['ground_truth']['bbox']=[0,0,1,1]
        db.execute('UPDATE video_measurements SET metrics=? WHERE run_id=? AND frame_id=1',(json.dumps(dict(overlays=[second])),'probe'))
    _,samples=collect_samples(store,VERSIONS)
    assert not any(s['positive'] for s in samples if s['source'].startswith('probe:'))
    with store.db() as db:
        db.execute('UPDATE model_probes SET result=? WHERE run_id=?',(json.dumps(dict(state='complete',frames=1,total=2)),'probe'))
    _,samples=collect_samples(store,VERSIONS)
    assert not any(s['source'].startswith('probe:') for s in samples)


def test_known_runner_up_does_not_tighten_another_identity_and_gallery_changes_disable_retention(tmp_path):
    import numpy as np
    from backend.fusion import model_gates
    store=labelled_store(tmp_path)
    vector=np.eye(512,dtype=np.float32)[0]
    person=store.add('A',[(b'unit',{v:vector for v in VERSIONS.values()},{})])
    rival=store.add('B',[(b'unit',{v:vector for v in VERSIONS.values()},{})])
    enrollment=[dict(person_id=p['id'],image_ids=[image['id'] for image in p['images']]) for p in store.people()]
    snapshot=[dict(model=key,version=version,band='normal',threshold=.8,margin=.6,
                   quality_basis='native_min_dimension_v1',verification_enabled=True) for key,version in VERSIONS.items()]
    with store.db() as db:
        db.execute('UPDATE diagnostics SET person_id=?,context=? WHERE id=?',(person,json.dumps(dict(enrollment=enrollment,validation_profile_snapshot=snapshot)),'live'))
        annotations=json.loads(db.execute('SELECT annotations FROM video_ground_truth WHERE source_hash=?',('source',)).fetchone()[0])
        for annotation in annotations:
            if annotation['kind']=='outside':annotation['scope']=sorted(set(annotation['scope']+[person,rival]))
        db.execute('UPDATE video_ground_truth SET annotations=? WHERE source_hash=?',(json.dumps(annotations),'source'))
        for index in range(8):
            current=face(person,.98,.13) if index==0 else face(person,.9,.7)
            db.execute('UPDATE diagnostic_frames SET metrics=? WHERE run_id=? AND frame_id=?',
                       (json.dumps(dict(overlays=[current])),'live',index))
    result=fit_profiles(store,VERSIONS)
    for profile in result['profiles']:
        config=profile['config']
        assert config['threshold']>.85 and config['identity_gates'][person]['threshold']==.8
        assert rival not in config['identity_gates']
        assert model_gates(.81,.65,config,person)[0]
        assert not model_gates(.81,.65,config,rival)[0]
        assert not model_gates(.81,.65,config,'unseen-person')[0]
        assert model_gates(.98,.13,config,person,rival)==(False,True)
        assert model_gates(.98,.13,config,person)==(False,True)
    apply_profiles(store,result,VERSIONS)
    assert store.calibration(VERSIONS['lvface'],100)['identity_gates'][person]['threshold']==.8
    # A real outsider/wrong top-1 remains a negative bound even with a small margin.
    with store.db() as db:
        db.execute('UPDATE video_measurements SET metrics=? WHERE run_id=? AND frame_id=0',
                   (json.dumps(dict(overlays=[face(person,.85,.02)])),'video'))
    result=fit_profiles(store,VERSIONS)
    assert all(person not in p['config'].get('identity_gates',{}) for p in result['profiles'])
    assert result['profiles']
    # New enrollment invalidates retention even if this branch has no new embedding.
    store.add('C',[(b'partial-unit',{VERSIONS['lvface']:vector},{})])
    assert not store.calibration(VERSIONS['antelopev2'],100).get('identity_gates')
    store.add('A',[(b'new-unit',{v:vector for v in VERSIONS.values()},{})],person)
    assert not store.calibration(VERSIONS['lvface'],100).get('identity_gates')


def test_identity_retention_keeps_negative_bounds_for_unobserved_third_references():
    from backend.calibration import retain_reviewed_identity_gates
    values=[dict(positive=True,fold='fit',subject='a',source='live:fit',time=i,score=.9,margin=.7) for i in range(3)]
    values.append(dict(positive=False,fold='validation',subject=None,source='video:negative',time=0.,score=.1,margin=None,label_kind='declared_absence'))
    candidate=dict(threshold=.8,margin=.6,quality_basis='native_min_dimension_v1',source_run_id='reviewed-snapshot')
    def config():return dict(threshold=.9,margin=.7,quality_basis='native_min_dimension_v1')
    # Face B has a high runner-up for C; A's unrecorded score may be almost as high.
    other=dict(positive=False,fold='validation',subject='b',source='video:labelled',time=1.,score=.85,margin=None,label_kind='runner_impostor',comparison_person_id='c')
    guarded=config();retain_reviewed_identity_gates(guarded,values+[other],[candidate],['one','two','three'],[])
    assert 'identity_gates' not in guarded
    # A wrong top-1 for B on the genuine A face still cannot be a negative A face.
    genuine=dict(other,subject='a',label_kind='wrong_top1',score=.99)
    retained=config();retain_reviewed_identity_gates(retained,values+[genuine],[candidate],['one','two','three'],[])
    assert retained['identity_gates']['a']['threshold']==.8


def test_transition_frames_never_fit_calibration(tmp_path):
    store=labelled_store(tmp_path)
    with store.db() as db:
        row=db.execute('SELECT metrics FROM diagnostic_frames WHERE frame_id=0').fetchone()
        metrics=json.loads(row[0]);metrics['diagnostic_phase']={'stage_id':'mask','transition':True}
        db.execute('UPDATE diagnostic_frames SET metrics=? WHERE frame_id=0',(json.dumps(metrics),))
    _,samples=collect_samples(store,VERSIONS)
    assert any(s['source']=='live:live' for s in samples)
    assert not any(s['source']=='live:live' and s['time']==0 for s in samples)


def test_replay_uses_recorded_camera_clock_but_keeps_phase_elapsed(tmp_path,monkeypatch):
    from backend.calibration import replay_diagnostic
    from backend.temporal import Temporal
    store=labelled_store(tmp_path);apply_profiles(store,fit_profiles(store,VERSIONS),VERSIONS)
    with store.db() as db:
        db.execute("INSERT INTO persons(id,name,selected) VALUES ('a','Fixture',1)")
        run=dict(db.execute("SELECT * FROM diagnostics WHERE id='live'").fetchone())
        rows=[(r['elapsed'],json.loads(r['metrics'])) for r in db.execute("SELECT * FROM diagnostic_frames WHERE run_id='live' ORDER BY elapsed")]
    for elapsed,f in rows:f.update(live_timing_revision='camera-media-clock-v1',source_timestamp=100.+elapsed)
    times=[];callback_times=[];original=Temporal.update
    def observe(self,track,pid,now,*a,**kw):
        times.append(now);return original(self,track,pid,now,*a,**kw)
    monkeypatch.setattr(Temporal,'update',observe)
    replay_diagnostic(store,run,rows,REVIEW,on_decision=lambda elapsed,*args:callback_times.append(elapsed))
    assert times==[100.+t for t,f in rows]
    assert callback_times==[t for t,f in rows]
    rows[0][1]['source_timestamp']=float('nan')
    import pytest
    with pytest.raises(ValueError):replay_diagnostic(store,run,rows,REVIEW)


def test_calibration_native_detail_domain_does_not_extend_to_unmeasured_faces(tmp_path):
    store=Store(tmp_path)
    store.calibrate('specialist:domain-unit','tiny',dict(threshold=.8,margin=.2,
        minimum_native_face_pixels=14,maximum_native_face_pixels=27))
    assert store.calibration('specialist:domain-unit',14) is not None
    assert store.calibration('specialist:domain-unit',27) is not None
    assert store.calibration('specialist:domain-unit',13) is None
    assert store.calibration('specialist:domain-unit',28) is None
    store.calibrate('legacy:domain-unit','tiny',dict(threshold=.8,margin=.2))
    assert store.calibration('legacy:domain-unit',13) is not None


def test_adjacent_validated_native_bands_keep_the_same_model_gate(tmp_path):
    from backend.fusion import model_gates
    store=Store(tmp_path)
    for band,low,high in [('tiny',14,27),('small',28,89)]:
        store.calibrate('specialist:adjacent',band,dict(threshold=.4,margin=.2,
            minimum_native_face_pixels=low,maximum_native_face_pixels=high))
    for pixels in (14,27,28,89):
        profile=store.calibration('specialist:adjacent',pixels)
        assert model_gates(.45,.25,profile)[0]
        assert not model_gates(.35,.25,profile)[0]
