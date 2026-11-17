import json
import time
from backend.diagnostic_analysis import analyze_frames, analyze_saved, next_pending, read_analysis, reviewed_metrics
from backend.diagnostics import Diagnostic
from backend.store import Store


def measurement(frame_id=1, face=True, error=None):
    models = {key: dict(person_id='target', score=score, passed=True, calibrated=False,
                       version=key+':unit', latency_ms=10, alignment_error=2)
              for key,score in [('lvface',.56),('antelopev2',.79),('petalface',.69)]}
    models['petalface']['iqa_threshold_verified'] = False
    if error:
        models = {'lvface': {'error': error, 'passed': False, 'calibrated': False}}
    overlay = dict(track_id=1,status='candidate' if not error else 'unknown',
                   quality={'face_pixels':68},model_scores=models)
    return dict(frame_id=frame_id,width=1280,height=720,processing_ms=167,detector_ms=92,
                overlays=[overlay] if face else [],events=[],sampling={'preview_fps':27})


def test_consensus_identifies_calibration_gate_without_claiming_accuracy():
    run=dict(id='unit',scenario='Chính diện 1 m',expected_present=1,person_id='target')
    analysis=analyze_frames(run,[(0.,measurement()),(.2,measurement(2)),(.4,measurement(3,False))])
    assert analysis['frames']==3 and analysis['frames_with_face']==2
    assert analysis['identity_consensus_observations']==2
    assert analysis['strong_consensus_observations']==2
    assert analysis['models']['lvface']['cosine']['p50']==.56
    assert 'calibration_missing' in analysis['blockers']
    assert 'petal_iqa_unverified' in analysis['blockers']
    assert analysis['calibrated_score_samples']==0
    assert 'accuracy' not in analysis and 'detection_recall' not in analysis


def test_quality_gate_and_runtime_errors_are_not_model_accuracy_or_alignment_failures():
    run=dict(id='unit',scenario='Chính diện 6 m',expected_present=1,person_id='target')
    frame=measurement()
    frame['overlays']=[dict(track_id=1,status='unknown',quality={},model_scores={},
                            face_bbox=[0,0,24,30],reason='Mặt quá nhỏ: 24px trước alignment.')]
    result=analyze_frames(run,[(0.,frame)])
    assert result['quality_rejected_faces']==1 and result['face_pixels']['p50']==24
    assert result['valid_face_observations']==0 and 'quality_gate' in result['blockers']
    result=analyze_frames(run,[(0.,measurement(error='CUDA out of memory'))])
    assert result['inference_failure_faces']==1 and result['alignment_failure_faces']==0


def test_live_accuracy_requires_user_review_and_counts_ranking_separately_from_confirmed():
    run=dict(person_id='target',expected_present=True)
    first=measurement();last=measurement()
    last['overlays'][0]['track_id']=2
    last['overlays'][0]['model_scores']['lvface']['person_id']='other'
    rows=[(0,first),(1,measurement(face=False)),(2,last)]
    assert reviewed_metrics(run,rows,{}) is None
    review=dict(only_target=True,face_visible_throughout=True,boxes_match_target=True)
    result=reviewed_metrics(run,rows,review)
    assert result['detection_recall']==2/3 and result['lost_track_episodes']==1
    assert result['face_track_changes']==1 and result['identity_switches']==0 and result['confirmed_correct_faces']==0
    assert result['models']['lvface']['top1_correct']==1 and result['models']['lvface']['top1_wrong']==1
    assert result['models']['antelopev2']['top1_correct']==2
    last['overlays']*=2
    assert reviewed_metrics(run,rows,review) is None


def test_saved_analysis_is_derived_and_never_modifies_source_frames_or_thresholds(tmp_path):
    st=Store(tmp_path)
    run=Diagnostic(st,'Chính diện 1 m',None,False,True)
    run.finish()
    raw=json.dumps(measurement(error='Crop mặt bị cắt ngoài ảnh quá nhiều.'))
    with st.db() as db:
        db.execute('INSERT INTO diagnostic_frames VALUES (?,?,?,?)',(run.id,1,0.,raw))
    before=st.settings()
    assert next_pending(st)==run.id
    result=analyze_saved(st,run.id)
    assert result['alignment_failure_faces']==1
    assert read_analysis(st,run.id)==result
    assert next_pending(st) is None and st.settings()==before
    with st.db() as db:
        assert db.execute('SELECT metrics FROM diagnostic_frames').fetchone()[0]==raw
        assert db.execute('SELECT count(*) FROM calibrations').fetchone()[0]==0


def test_running_run_is_not_analyzed(tmp_path):
    st=Store(tmp_path);run=Diagnostic(st,'Nhóm đông',None,False,True)
    assert analyze_saved(st,run.id) is None and next_pending(st) is None


def test_completed_labelled_video_counts_as_negative_evidence_without_requesting_extra_live_people(tmp_path):
    st=Store(tmp_path)
    run=Diagnostic(st,'Chính diện 3 m',None,False,True);run.finish()
    with st.db() as db:
        db.execute('INSERT INTO video_ground_truth(source_hash,filename,duration,annotations) VALUES (?,?,?,?)',
                   ('unit-source','unit.mp4',10,json.dumps([dict(kind='outside',start=0,end=5)])))
    assert analyze_saved(st,run.id)['ground_truth_videos']['sources']==0
    with st.db() as db:
        db.execute('INSERT INTO video_evaluations(run_id,source_hash,result) VALUES (?,?,?)',
                   ('unit-video','unit-source',json.dumps(dict(state='complete',frames=20))))
    result=analyze_saved(st,run.id)
    assert result['ground_truth_videos']==dict(sources=1,positive_intervals=0,negative_intervals=1)
    assert not any('Dữ liệu hiện chỉ có mục tiêu' in action for action in result['actions'])
    assert not result.get('calibration_passed')


def test_unknown_ground_truth_is_not_interpreted_as_successful_identity_match():
    run=dict(id='negative',scenario='Người ngoài watchlist',expected_present=0,person_id=None)
    result=analyze_frames(run,[(0.,measurement())])
    assert 'impostor_similarity' in result['blockers']
    assert 'từ chối' in result['title']


def test_background_worker_analyzes_archived_runs_and_api_survives_restart(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main
    st=Store(tmp_path);run=Diagnostic(st,'Người ngoài watchlist',None,False,True);run.finish()
    monkeypatch.setattr(main,'store',st)
    monkeypatch.setattr(main,'diagnostics',{})
    monkeypatch.setattr(main.engine,'load',lambda:None)
    monkeypatch.setattr(st,'reindex',lambda _:[])
    with TestClient(main.app) as client:
        deadline=time.monotonic()+5
        while not read_analysis(st,run.id):
            assert time.monotonic()<deadline
            time.sleep(.02)
        result=client.get('/api/diagnostics/'+run.id)
        assert result.status_code==200 and result.json()['analysis']['frames']==0
        assert client.get('/api/diagnostics').json()['runs'][0]['analysis']
        subsequent=Diagnostic(st,'Nhóm đông',None,False,True);subsequent.finish()
        deadline=time.monotonic()+5
        while not read_analysis(st,subsequent.id):
            assert time.monotonic()<deadline
            time.sleep(.02)
        assert client.get('/api/diagnostics/'+subsequent.id).json()['analysis']['run_id']==subsequent.id


def test_diagnostic_timing_keeps_first_duplicate_event_and_marks_preparation_reset(tmp_path):
    store=Store(tmp_path);store.add('Fixture',[(b'unit',[],{})])
    pid=store.people()[0]['id'];now=[0.]
    run=Diagnostic(store,'Chính diện 1 m',pid,True,True,clock=lambda:now[0])
    run.observe(measurement(),0.)
    now[0]=5.1;run.observe(measurement(),5.1);run.begin()
    for index,at in enumerate((5.2,5.4),start=2):
        now[0]=at;frame=measurement(index)
        frame['measurement_evidence_reset']=index==2
        frame['events']=[dict(id='one-event',person_id=pid,confirmation_seconds=.1)]
        run.observe(frame,at)
    now[0]=35.1;summary=run.finish('complete')['summary']
    assert abs(summary['first_alert_seconds']-.1)<1e-8
    assert summary['preparation_evidence_isolated'] is True
    with store.db() as db:assert db.execute('SELECT count(*) FROM diagnostic_frames').fetchone()[0]==2


def test_shadow_model_is_reported_separately_without_changing_production_consensus():
    run=dict(id='unit',scenario='Chính diện 1 m',expected_present=1,person_id='target')
    frame=measurement();frame['overlays'][0]['model_scores']={'lvface':frame['overlays'][0]['model_scores']['lvface'],
        'expert':dict(person_id='target',score=.99,passed=True,calibrated=True,shadow=True,version='expert:unit')}
    result=analyze_frames(run,[(0.,frame)])
    assert result['identity_consensus_observations']==0 and result['strong_consensus_observations']==0
    assert result['models']['expert']['shadow_samples']==1


def test_late_overlay_counter_excludes_preparation_and_counts_only_observed_increases():
    run=dict(id='unit',scenario='unit',expected_present=False,person_id=None)
    first=measurement();last=measurement(2)
    first['sampling']['late_overlay_results']=26
    last['sampling']['late_overlay_results']=27
    result=analyze_frames(run,[(0.,first),(1.,last)])
    assert result['late_overlay_results']==1
    assert result['late_overlay_results_counter_total']==27
    assert first['sampling']['late_overlay_results']==26


def test_composite_capture_time_phase_and_full_180_seconds(tmp_path):
    from backend.diagnostics import phase_at, protocol_for
    store=Store(tmp_path);clock=[100.]
    run=Diagnostic(store,'Tổng hợp 3 m',None,False,True,clock=lambda:clock[0])
    assert run.view()['protocol']['duration_seconds']==180
    assert run.view()['phase'] is None
    run.observe(measurement(),100.)
    clock[0]=105.;run.observe(measurement(),104.9);run.begin()
    assert run.view()['remaining']==180
    # Inference completes after the boundary, but the face belongs to capture time.
    clock[0]=136.
    run.observe(dict(measurement(10),media_timestamp=29.9,captured_at=129900),134.9)
    run.observe(measurement(11),135.)
    clock[0]=203.;run.observe(measurement(12),203.)  # 98 s, mask transition just over.
    clock[0]=284.9;run.observe(measurement(13),284.9)
    assert run.state=='running' and run.view()['remaining'] > 0
    clock[0]=285.;run.observe(measurement(14),285.)
    assert run.state=='complete' and run.summary['duration_seconds']==180
    with store.db() as db:
        frames=[json.loads(r[0]) for r in db.execute('SELECT metrics FROM diagnostic_frames ORDER BY elapsed')]
    assert len(frames)==4
    assert frames[0]['frame_id']==10 and frames[0]['diagnostic_phase']['stage_id']=='frontal'
    assert frames[0]['source_timestamp']==29.9 and frames[0]['captured_at']==129900
    assert frames[1]['diagnostic_phase']['stage_id']=='profile' and frames[1]['diagnostic_phase']['transition']
    assert frames[2]['diagnostic_phase']['stage_id']=='mask' and not frames[2]['diagnostic_phase']['transition']
    assert frames[3]['diagnostic_phase']['stage_id']=='partial_occlusion'
    assert phase_at(protocol_for('Tổng hợp 12.5 m'),180) is None


def test_composite_analysis_excludes_transitions_preserves_tracked_and_requires_human_labels():
    from backend.diagnostics import protocol_for
    from backend.diagnostic_analysis import analyze_protocol
    protocol=protocol_for('Tổng hợp 1 m')
    run=dict(id='composite-unit',scenario='Tổng hợp 1 m',expected_present=True,person_id='target',context={'protocol':protocol})
    confirmed=measurement(1);confirmed['overlays'][0].update(status='confirmed',person_id='target')
    tracked=measurement(2);tracked['overlays'][0].update(status='tracked',person_id='target')
    unknown=measurement(3);unknown['overlays'][0].update(status='unknown',person_id=None)
    rows=[(29.9,confirmed),(30.,unknown),(33.,tracked),(90.,unknown),(98.,confirmed),(150.,unknown),(158.,unknown)]
    result=analyze_protocol(run,rows,{})
    assert result['transition_frames']==3 and result['evaluated_frames']==4
    assert result['stages'][1]['metrics']['statuses']=={'tracked':1}
    assert result['stages'][1]['reviewed_metrics'] is None
    assert not any(s['action_confirmed'] for s in result['stages'])
    review=dict(only_target=True,face_visible_throughout=True,boxes_match_target=True,stage_actions={'profile':True})
    reviewed=analyze_protocol(run,rows,review)
    assert reviewed['stages'][1]['action_confirmed']
    assert reviewed['stages'][1]['reviewed_metrics']['tracked_correct_faces']==1
    assert reviewed['stages'][1]['reviewed_metrics']['confirmed_correct_faces']==0
    assert reviewed['stages'][5]['error_frame_ids']==[3]
    assert reviewed['stages'][5]['replay_interval']==dict(start=150,end=180)
    assert sum(stage['evaluated_frames']+stage['transition_frames'] for stage in reviewed['stages'])==len(rows)


def test_composite_protocol_boundaries_and_legacy_duration():
    from backend.diagnostics import phase_at, protocol_for
    protocol=protocol_for('Tổng hợp 6 m')
    assert [s['start'] for s in protocol['stages']]==list(range(0,180,20))
    assert [s['group'] for s in protocol['stages']]==list('AAABBBCCC')
    assert protocol_for('Quay trái/phải 3 m')['duration_seconds']==30
    assert phase_at(protocol,-1) is None
    for i,stage in enumerate(protocol['stages']):
        assert phase_at(protocol,stage['start'])['stage_id']==stage['id']
        assert phase_at(protocol,stage['end']-.001)['stage_id']==stage['id']
        assert not phase_at(protocol,stage['start']+stage['transition_seconds'])['transition']


def test_composite_requires_fresh_consent_and_does_not_start_another_run(tmp_path):
    import pytest
    store=Store(tmp_path)
    with pytest.raises(ValueError):Diagnostic(store,'Tổng hợp 1 m',None,False,False)
    run=Diagnostic(store,'Tổng hợp 1 m',None,False,True)
    run.finish('cancelled')
    assert run.view()['phase'] is None and run.summary['frames']==0
    with store.db() as db:assert db.execute('SELECT count(*) FROM diagnostics').fetchone()[0]==1


def test_composite_watchdog_uses_180_seconds_and_stops_only_its_session(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main
    store=Store(tmp_path)
    run=Diagnostic(store,'Tổng hợp 3 m',None,False,True)
    run.state='running';run.started=time.monotonic()-31;run.last_frame=time.monotonic();run.session_id='own-session'
    monkeypatch.setattr(main,'store',store)
    monkeypatch.setattr(main,'diagnostics',{run.id:run})
    session=dict(id='own-session',stopped=False)
    monkeypatch.setattr(main,'active_live',session)
    monkeypatch.setattr(main.engine,'load',lambda:None)
    monkeypatch.setattr(store,'reindex',lambda _:[])
    with TestClient(main.app):
        time.sleep(.2)
        assert run.state=='running' and not session['stopped']
        run.started=time.monotonic()-180
        deadline=time.monotonic()+3
        while run.state=='running':
            assert time.monotonic()<deadline
            time.sleep(.02)
        assert run.state=='complete' and session['stopped']
        assert run.summary['duration_seconds']==180
        assert len(main.diagnostics)==1


def test_capture_timestamp_uses_browser_time_even_when_processing_crosses_action_boundary(tmp_path):
    import pytest
    run=Diagnostic(Store(tmp_path),'Tổng hợp 3 m',None,False,True)
    run.started=100.
    assert run.capture_time({'diagnostic_elapsed':29.95},130.2)==pytest.approx(129.95)
    assert run.capture_time({'diagnostic_elapsed':-.1},100.2)<run.started
    assert run.capture_time({'diagnostic_capture_pending':True},100.2)<run.started
    assert run.capture_time({},130.2)==130.2  # Explicit legacy compatibility.
    for value in (float('nan'),float('inf'),True,'29',31):
        with pytest.raises(ValueError):run.capture_time({'diagnostic_elapsed':value},130.)


def test_composite_volatile_examples_preserve_earlier_failure_pairs_and_remain_bounded(tmp_path):
    run=Diagnostic(Store(tmp_path),'Tổng hợp 3 m',None,False,True,clock=lambda:0.)
    run.state='running';run.started=0.
    expected=[]
    for stage in range(6):
        expected.extend([stage*100,stage*100+1])
        for n in range(90):
            status='confirmed' if n%3==0 else 'unknown'
            frame=dict(frame_id=stage*100+n,width=1280,height=720,overlays=[dict(status=status,person_id='unit' if status=='confirmed' else None,model_scores={})])
            raw=b'original '+str(frame['frame_id']).encode()
            run.retain_example(frame,stage*30+n*.3,raw)
            samples=list(run.examples.values())+([run.previous_named_example] if run.previous_named_example else [])
            assert len({s['frame_id'] for s in samples})<=1024
            assert sum(len(s['raw']) for s in samples)<96*1024**2
    present={s['frame_id'] for s in run.examples.values()}
    assert set(expected)<=present
    assert len(present)==540  # Every actual AI input in this fixture remains replayable.
    assert len({s['stage_id'] for s in run.examples.values()})==6


def test_volatile_native_input_cache_enforces_bytes_and_balances_actions(tmp_path):
    run=Diagnostic(Store(tmp_path),'Tổng hợp 3 m',None,False,True,clock=lambda:0.)
    run.state='running';run.started=0.;raw=b'x'*(2*1024**2)
    for phase in range(6):
        for n in range(12):
            f=dict(frame_id=phase*100+n,width=1920,height=1080,overlays=[])
            run.retain_example(f,phase*30+n,raw)
            assert sum(len(item['raw']) for item in run.examples.values())<=96*1024**2
    assert len({item['stage_id'] for item in run.examples.values()})==6
    assert all(item['width']==1920 and item['height']==1080 and item['raw'] is raw for item in run.examples.values())


def test_selective_movement_retest_has_reference_normal_and_faster_actions_in_30_seconds():
    from backend.diagnostics import protocol_for,phase_at
    p=protocol_for('Giơ tay/di chuyển 3 m')
    assert p['duration_seconds']==30 and p['kind']=='targeted'
    assert phase_at(p,0)['stage_id']=='movement_reference'
    assert phase_at(p,5)['stage_id']=='movement_normal'
    assert phase_at(p,15)['stage_id']=='movement_faster'
    assert phase_at(p,30) is None


def test_track_id_fragmentation_is_not_an_identity_label_swap():
    run=dict(person_id='target',expected_present=True)
    review=dict(only_target=True,face_visible_throughout=True,boxes_match_target=True)
    one=measurement();two=measurement(2)
    one['overlays'][0].update(status='confirmed',person_id='target')
    two['overlays'][0].update(status='confirmed',person_id='target',track_id=2)
    m=reviewed_metrics(run,[(0,one),(.2,two)],review)
    assert m['face_track_changes']==1 and m['identity_switches']==0
    assert m['detection_recall_scope']=='processed_ai_inputs_only' and m['preview_detection_recall'] is None
    two['overlays'][0]['person_id']='other'
    m=reviewed_metrics(run,[(0,one),(.2,two)],review)
    assert m['identity_switches']==1 and m['confirmed_wrong_faces']==1


def test_visual_measurements_deduplicate_frames_and_exclude_preparation():
    from backend.diagnostic_analysis import visual_flow_metrics
    report=dict(anchor_frame_id=8,frames=[dict(frame_id=1,timestamp=.05,faces=1,forward_boxes=1),
        dict(frame_id=3,timestamp=.3,faces=1,forward_boxes=1,tracks=[dict(track_id=1,status='tracked',person_id='target')]),
        dict(frame_id=5,timestamp=.5,faces=0,forward_boxes=0)])
    rows=[(.1,dict(source_timestamp=.1,sampling=dict(visual_tracking_ms=9,visual_tracking_summary=report))),
          (.8,dict(source_timestamp=.8,sampling=dict(visual_tracking_ms=9,visual_tracking_summary=report)))]
    m=visual_flow_metrics(rows)
    assert m['frames']==2 and m['frames_with_box']==1 and m['forward_rescued_boxes']==1
    assert m['tracked_faces']==1 and m['detection_recall'] is None
    assert m['cpu_ms']['p50']==9
    assert visual_flow_metrics([(.1,dict(source_timestamp=.1,sampling={}))]) is None


def test_volatile_preview_failure_samples_share_budget_and_do_not_break_ai_pairs(tmp_path):
    run=Diagnostic(Store(tmp_path),'Giơ tay/di chuyển 3 m',None,False,True,clock=lambda:10.)
    run.state='running';run.started=10.
    first=measurement(1);first['overlays'][0].update(status='confirmed',person_id='unit')
    run.retain_example(first,10.1,b'first real input')
    visual=dict(frame_id=2,width=1280,height=720,media_timestamp=.2,overlays=[])
    run.retain_visual_example(visual,10.2,b'gray preview input',3)
    assert run.previous_named_example['frame_id']==1
    last=measurement(3);last['overlays'][0].update(status='unknown',person_id=None)
    run.retain_example(last,10.3,b'last real input')
    items={item['frame_id']:item for item in run.examples.values()}
    assert items[2]['kind']=='preview' and items[2]['anchor_frame_id']==3
    assert items[1]['severity']==4 and items[3]['severity']==5
    assert items[1]['kind']==items[3]['kind']=='ai'
    before=dict(run.examples)
    run.retain_visual_example(dict(visual,frame_id=4),9.9,b'preparation',3)
    run.state='complete';run.retain_visual_example(dict(visual,frame_id=5),10.4,b'after end',3)
    assert run.examples==before


def test_volatile_preview_pressure_preserves_unnamed_native_ai_inputs(tmp_path):
    run=Diagnostic(Store(tmp_path),'Giơ tay/di chuyển 3 m',None,False,True,clock=lambda:10.)
    run.state='running';run.started=10.
    raw=b'x'*(400*1024)
    failed=[]
    for n in range(20):
        frame=measurement(n*10+1);frame['overlays'][0].update(status='unknown',person_id=None)
        run.retain_example(frame,10.+n,raw);failed.append(frame['frame_id'])
        for k in range(1,5):
            preview=dict(frame_id=n*10+k+1,width=1280,height=720,media_timestamp=n+.01*k,overlays=[])
            run.retain_visual_example(preview,10.+n+.01*k,raw,n*10+10)
        assert sum(len(x['raw']) for x in run.examples.values())<=16*1024**2
    assert set(failed)<=set(run.examples)
    assert all(run.examples[f]['kind']=='ai' for f in failed)


def test_selective_profile_movement_protocol_is_30_seconds_with_explicit_angle_actions():
    from backend.diagnostics import protocol_for,phase_at
    p=protocol_for('Quay nghiêng/di chuyển 3 m')
    assert p['duration_seconds']==30 and p['preparation_seconds']==5
    assert phase_at(p,0)['stage_id']=='profile_reference'
    assert phase_at(p,5)['stage_id']=='profile_moderate' and phase_at(p,5)['transition']
    assert phase_at(p,11)['stage_id']=='profile_left' and not phase_at(p,11)['transition']
    assert phase_at(p,18)['stage_id']=='profile_right'
    assert phase_at(p,25)['stage_id']=='profile_movement'
    assert phase_at(p,30) is None


def test_no_glasses_instructions_cover_profile_mask_and_partial_occlusion_without_changing_clocks():
    from backend.diagnostics import protocol_for
    for scenario in ['Quay nghiêng/di chuyển 3 m','Đeo khẩu trang 3 m','Che một phần mặt 3 m']:
        original=protocol_for(scenario);bare=protocol_for(scenario,eyewear='none')
        assert original['eyewear']=='unchanged' and bare['eyewear']=='none'
        assert bare['duration_seconds']==original['duration_seconds']==30
        assert bare['preparation_seconds']==5
        assert [(s['id'],s['start'],s['end']) for s in bare['stages']]==[(s['id'],s['start'],s['end']) for s in original['stages']]
        assert all(s['instruction'].startswith('KHÔNG ĐEO KÍNH — ') and 'GIỮ NGUYÊN KÍNH' not in s['instruction'] for s in bare['stages'])


def test_eyewear_is_persisted_with_the_run_and_rejects_unknown_modes(tmp_path):
    import pytest,json
    store=Store(tmp_path)
    run=Diagnostic(store,'Quay nghiêng/di chuyển 3 m',None,False,True,eyewear='none')
    assert run.context['eyewear']=='none'
    with store.db() as db:
        saved=json.loads(db.execute('SELECT context FROM diagnostics WHERE id=?',(run.id,)).fetchone()[0])
    assert saved['protocol']['eyewear']=='none'
    run.finish()
    with pytest.raises(ValueError):Diagnostic(store,'Quay nghiêng/di chuyển 3 m',None,False,True,eyewear='invented')


def test_composite_defaults_to_no_glasses_at_all_distances_and_keeps_old_protocols():
    from backend.diagnostics import protocol_for,phase_at
    import copy
    for distance in [3,6,9,12]:
        p=protocol_for(f'Tổng hợp {distance} m')
        assert p['eyewear']=='none' and p['duration_seconds']==180 and p['preparation_seconds']==5
        assert [s['id'] for s in p['stages']]==(['frontal','profile','movement','mask','mask_profile','mask_movement','unmask_reference','unmask_movement','partial_occlusion'] if distance==6 else ['frontal','profile','movement','mask','mask_movement','partial_occlusion'])
        assert all(s['instruction'].startswith('KHÔNG ĐEO KÍNH') for s in p['stages'])
        assert all(s['end']-s['start']==(20 if distance==6 else 30) for s in p['stages'])
        boundary=60 if distance==6 else 90
        assert phase_at(p,boundary)['transition'] and not phase_at(p,boundary+8)['transition']
    explicit=protocol_for('Tổng hợp 3 m',eyewear='unchanged')
    original=copy.deepcopy(explicit)
    protocol_for('Tổng hợp 3 m')
    assert explicit==original and explicit['eyewear']=='unchanged'


def test_native_diagnostic_input_survives_cache_and_runtime_restart(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main
    store=Store(tmp_path)
    run=Diagnostic(store,'Tổng hợp 6 m',None,False,True,clock=lambda:10.)
    run.state='running';run.started=0.
    result=dict(frame_id=91,width=1280,height=720,overlays=[])
    run.retain_example(result,10.,b'original-camera-jpeg-fixture')
    run.examples.clear()
    monkeypatch.setattr(main,'store',store)
    monkeypatch.setattr(main,'diagnostics',{})
    client=TestClient(main.app)
    response=client.get('/api/diagnostics/'+run.id+'/examples')
    assert response.status_code==200 and response.json()['volatile'] is False
    assert response.json()['examples'][0]['stage_id']=='frontal'
    assert client.get('/api/diagnostics/'+run.id+'/examples/91').content==b'original-camera-jpeg-fixture'
    run.started=-60.
    run.sample_group_a_hardware()
    with store.db() as db:
        assert db.execute('SELECT COUNT(*) FROM diagnostic_hardware').fetchone()[0]==0


def test_begin_rejection_keeps_preparation_and_records_reason_until_fresh_ai(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main
    clock=[106.]
    store=Store(tmp_path)
    run=Diagnostic(store,'Tổng hợp 6 m',None,False,True,clock=lambda:clock[0])
    run.state='preparing';run.prepared=100.;run.last_frame=100.
    monkeypatch.setattr(run,'sample_group_a_hardware',lambda:None)
    monkeypatch.setattr(main,'store',store)
    monkeypatch.setattr(main,'diagnostics',{run.id:run})
    client=TestClient(main.app)
    response=client.post('/api/diagnostics/'+run.id+'/begin',json={})
    assert response.status_code==409
    assert run.state=='preparing' and run.started is None
    with store.db() as db:
        context=json.loads(db.execute('SELECT context FROM diagnostics WHERE id=?',(run.id,)).fetchone()[0])
    assert context['last_begin_error']==response.json()['detail']
    run.last_frame=106.
    response=client.post('/api/diagnostics/'+run.id+'/begin',json={})
    assert response.status_code==200 and run.state=='running'
    assert response.json()['remaining']==180
    run.finish('cancelled')
