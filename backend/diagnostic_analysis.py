"""Read-only interpretation of recorded measurements; never changes identity thresholds."""
import json
from collections import Counter, defaultdict
import numpy as np

ANALYSIS_VERSION = 18


def percentiles(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return dict(zip(('p05', 'p50', 'p95'), [round(float(v), 4) for v in np.percentile(values, [5, 50, 95])]))


def visual_flow_metrics(rows):
    timestamps = [f.get('source_timestamp') for _,f in rows if f.get('source_timestamp') is not None]
    if not timestamps:
        return None
    low,high = min(timestamps),max(timestamps)
    observed,anchors = {},{}
    for _,frame in rows:
        sampling=frame.get('sampling') or {};summary=sampling.get('visual_tracking_summary') or {}
        if summary.get('anchor_frame_id') is not None:
            anchors[summary['anchor_frame_id']]=sampling.get('visual_tracking_ms')
        for sample in summary.get('frames',[]):
            timestamp=sample.get('timestamp')
            if isinstance(timestamp,(int,float)) and low<=timestamp<=high and sample.get('frame_id') is not None:
                observed[sample['frame_id']]=sample
    if not observed:
        return None
    return dict(frames=len(observed),frames_with_box=sum((f.get('faces') or 0)>0 for f in observed.values()),
        forward_rescued_boxes=sum(f.get('forward_boxes',0) for f in observed.values()),
        tracked_faces=sum(track.get('status')=='tracked' for f in observed.values() for track in f.get('tracks',[])),
        cpu_ms=percentiles(list(anchors.values())),
        scope='reported_intermediate_frames_between_first_and_last_measured_AI_capture',
        detection_recall=None, caution='Chỉ là các frame trung gian đã được ghi nhận; không phải toàn bộ preview hoặc nhận diện sinh trắc mới.')


def reviewed_metrics(run, rows, review):
    required = ('only_target', 'face_visible_throughout', 'boxes_match_target')
    if not rows or not run.get('person_id') or not run.get('expected_present') or not all(review.get(k) is True for k in required):
        return None
    if any(len(frame.get('overlays', [])) > 1 for _,frame in rows):
        return None  # Multiple faces require individual annotations, not a single-person assertion.
    detections = sum(len(f.get('overlays', [])) == 1 for _,f in rows)
    model_results = {}
    lost, switches, previous = 0, 0, None
    had_face = True
    identity_changes, previous_named = 0, None
    for _, frame in rows:
        faces = frame.get('overlays', [])
        if not faces:
            lost += had_face
            had_face = False
            continue
        face = faces[0]
        if previous is not None and face['track_id'] != previous:
            switches += 1
        previous, had_face = face['track_id'], True
        if face.get('status') in ('confirmed','tracked') and face.get('person_id'):
            identity_changes += previous_named is not None and previous_named != face['person_id']
            previous_named = face['person_id']
        for key, score in face.get('model_scores', {}).items():
            stats = model_results.setdefault(key, dict(samples=0,top1_correct=0,top1_wrong=0))
            if score.get('score') is not None and score.get('person_id'):
                stats['samples'] += 1
                stats['top1_correct' if score['person_id']==run['person_id'] else 'top1_wrong'] += 1
    for stats in model_results.values():
        stats['top1_correct_fraction'] = stats['top1_correct']/stats['samples'] if stats['samples'] else None
        stats['frames_without_valid_score'] = len(rows)-stats['samples']
    confirmed = [o for _,f in rows for o in f.get('overlays', []) if o.get('status')=='confirmed']
    tracked = [o for _,f in rows for o in f.get('overlays', []) if o.get('status')=='tracked']
    return dict(tracked_correct_faces=sum(o.get('person_id')==run['person_id'] for o in tracked),
        tracked_wrong_faces=sum(o.get('person_id')!=run['person_id'] for o in tracked),
        source='user_review_whole_run', detection_recall_scope='processed_ai_inputs_only', preview_detection_recall=None, expected_face_frames=len(rows), detected_face_frames=detections,
        detection_recall=detections/len(rows), lost_track_episodes=lost, face_track_changes=switches, identity_switches=identity_changes,
        confirmed_correct_faces=sum(o.get('person_id')==run['person_id'] for o in confirmed),
        confirmed_wrong_faces=sum(o.get('person_id')!=run['person_id'] for o in confirmed), models=model_results,
        caution='Chỉ tính trên các frame đã chạy AI; không phải recall của mọi frame preview. Nhãn toàn lượt do người dùng xác nhận: chỉ mục tiêu, mặt luôn hiện và box đúng mặt. '
                'Top-1 là xếp hạng ứng viên, không thay cho Confirmed hoặc độ chính xác trên người ngoài watchlist.')


def analyze_frames(run, rows):
    frames = [f for _, f in rows]
    faces = [o for f in frames for o in f.get('overlays', [])]
    statuses = Counter(o.get('status', 'unknown') for o in faces)
    dimensions = Counter(f"{f['width']}×{f['height']}" for f in frames)
    errors, model_samples, model_errors = Counter(), defaultdict(list), Counter()
    consensus = strong_consensus = valid_faces = alignment_failure_faces = 0
    quality_rejected_faces = inference_failure_faces = high_residual_faces = 0
    tracks_changed = 0
    previous_face = None
    for elapsed, frame in rows:
        observed = frame.get('overlays', [])
        if len(observed) == 1:
            face = observed[0]
            if previous_face and elapsed-previous_face[0] < 1.5 and face['track_id'] != previous_face[1]:
                tracks_changed += 1
            previous_face = (elapsed, face['track_id'])
        else:
            previous_face = None
        for face in observed:
            scores = face.get('model_scores', {})
            high_residual_faces += any(v.get('alignment_band') == 'high_residual' for v in scores.values())
            reason = face.get('reason') or ''
            if not scores and reason.startswith(('Mặt quá nhỏ:', 'Mặt mờ:')):
                quality_rejected_faces += 1
                errors[reason] += 1
            proposals, passed, any_valid, face_failed, inference_failed = set(), [], False, False, False
            for key, score in scores.items():
                if score.get('error'):
                    errors[score['error']] += 1
                    model_errors[key] += 1
                    alignment_error = any(part in score['error'] for part in
                        ('Alignment', 'Landmarks', 'landmarks', 'Crop mặt', 'căn chỉnh'))
                    if key == 'lvface' and alignment_error:
                        face_failed = True
                    if not alignment_error:
                        inference_failed = True
                    continue
                value = score.get('score')
                if value is None or not np.isfinite(value):
                    continue
                model_samples[key].append(score)
                any_valid = True
                if score.get('shadow'):
                    continue
                if score.get('person_id'):
                    proposals.add(score['person_id'])
                if score.get('passed'):
                    passed.append(score)
            valid_faces += int(any_valid)
            alignment_failure_faces += int(face_failed)
            inference_failure_faces += int(inference_failed)
            if len(proposals) == 1 and len([v for v in scores.values() if v.get('score') is not None and not v.get('shadow')]) >= 2:
                if not run.get('person_id') or next(iter(proposals)) == run['person_id']:
                    consensus += 1
                    if len(passed) >= 2:
                        strong_consensus += 1
    models = {key: dict(samples=len(samples), cosine=percentiles([s['score'] for s in samples]),
        latency_ms=percentiles([s.get('latency_ms') for s in samples]),
        alignment_error_px=percentiles([s.get('alignment_error') for s in samples]),
        reference_gate_passes=sum(bool(s.get('passed')) for s in samples),
        calibrated_samples=sum(bool(s.get('calibrated')) for s in samples),
        shadow_samples=sum(bool(s.get('shadow')) for s in samples),
        errors=model_errors[key], versions=sorted({s['version'] for s in samples if s.get('version')}))
        for key, samples in model_samples.items()}
    def native_pixels(face):
        value = face.get('quality', {}).get('face_pixels')
        box = face.get('face_bbox', face.get('bbox'))
        return value if value is not None else min(box[2]-box[0], box[3]-box[1]) if box else None
    pixels = percentiles([native_pixels(o) for o in faces])
    sampling = [f.get('sampling') or {} for f in frames]
    late_counters = [sample.get('late_overlay_results',0) or 0 for sample in sampling]
    late_increase = sum(max(0,current-previous) for previous,current in zip(late_counters,late_counters[1:]))
    blockers, findings, actions = [], [], []
    if not frames:
        findings.append('Lượt này chưa có frame trong thời gian đo; không suy luận chất lượng nhận diện.')
    else:
        face_frames = sum(bool(f.get('overlays')) for f in frames)
        findings.append(f'Có detection khuôn mặt trong {face_frames}/{len(frames)} frame đã phân tích; đây không phải detection recall có nhãn từng mặt.')
        if quality_rejected_faces:
            findings.append(f'{quality_rejected_faces} quan sát bị chặn ở ngưỡng kích thước/độ nét trước khi trích xuất embedding; chưa phải lỗi model nhận diện.')
            if not valid_faces:
                blockers.append('quality_gate')
                actions.append('Cần thu số đo nhận diện riêng cho mặt nhỏ với dải hiệu chuẩn riêng; không áp ngưỡng mặt lớn và không tự gán danh tính.')
        if high_residual_faces:
            findings.append(f'{high_residual_faces} quan sát có sai số khớp landmark cao; điểm model thuộc dải hiệu chuẩn alignment riêng.')
        if inference_failure_faces:
            findings.append(f'{inference_failure_faces} quan sát có lỗi thực thi embedding; cần kiểm tra riêng với lỗi crop/alignment.')
        if alignment_failure_faces:
            findings.append(f'{alignment_failure_faces} quan sát mặt bị từ chối ở nhánh chính do lỗi crop/landmarks/alignment.')
            actions.append('Giữ trọn mặt ở giữa khung; tránh để trán/cằm sát mép hình.')
        elif valid_faces:
            findings.append(f'{valid_faces} quan sát mặt có embedding hợp lệ; không ghi nhận lỗi crop/alignment ở nhánh chính.')
        if consensus:
            findings.append(f'{consensus} quan sát có ít nhất hai model cùng gợi ý một danh tính; {strong_consensus} quan sát vượt ngưỡng tham khảo của ít nhất hai nhánh.')
        uncalibrated = any(s.get('score') is not None and not s.get('calibrated') for o in faces for s in o.get('model_scores', {}).values())
        unverified_iqa = any('iqa_threshold_verified' in s and not s['iqa_threshold_verified']
            for o in faces for s in o.get('model_scores', {}).values())
        verified_pair = any(sum(bool(v.get('calibrated')) and (key != 'petalface' or v.get('iqa_threshold_verified', False))
            for key,v in face.get('model_scores', {}).items()) >= 2 for face in faces)
        if statuses.get('confirmed', 0) == 0 and uncalibrated and not verified_pair:
            blockers.append('calibration_missing')
            findings.append('Đã có điểm đối chiếu model. Cần ngưỡng hiệu chuẩn theo model và chất lượng ảnh để xác nhận danh tính.')
        if not run['expected_present'] and strong_consensus:
            blockers.append('impostor_similarity')
            findings.append(f'Lượt đối chứng không có mục tiêu nhưng {strong_consensus} quan sát vượt ngưỡng tham khảo; chưa thể dùng các ngưỡng này để khẳng định chống nhận nhầm.')
        if unverified_iqa:
            blockers.append('petal_iqa_unverified')
            findings.append('Cấu hình IQA của PETALface chưa được kiểm chứng bằng dữ liệu đối chứng.')
        if pixels and pixels['p50'] < 90:
            findings.append(f"Mặt gốc trung vị rộng {pixels['p50']:g} px; resize lên input model không bổ sung chi tiết sinh trắc học.")
            actions.append('Kiểm tra độ phân giải nguồn Camo và vị trí khuôn mặt; giữ nguyên ảnh gốc gửi AI.')
        if tracks_changed:
            findings.append(f'Face track thay đổi {tracks_changed} lần giữa các quan sát một mặt gần nhau; chưa coi đây là identity switch nếu thiếu nhãn từng mặt.')
    preparation_isolated = any(frame.get('measurement_evidence_reset') for frame in frames)
    if frames and not preparation_isolated:
        findings.append('Lượt cũ chưa tách trạng thái xác nhận của 5 giây chuẩn bị; thời điểm cảnh báo đầu tiên không được dùng để kết luận độ trễ xác nhận.')
    calibrated_samples = sum(m['calibrated_samples'] for m in models.values())
    if not run['expected_present'] and statuses.get('confirmed', 0):
        title = 'Có Confirmed trái với nhãn lượt đối chứng; cần kiểm tra nhận nhầm'
    elif 'impostor_similarity' in blockers:
        title = 'Điểm người ngoài watchlist vượt ngưỡng tham khảo; cần hiệu chuẩn khả năng từ chối'
    elif 'calibration_missing' in blockers:
        title = 'Model đã đối chiếu được ảnh; bước xác nhận chưa được hiệu chuẩn'
    elif not frames:
        title = 'Không có dữ liệu đo trong lượt này'
    elif 'quality_gate' in blockers:
        title = 'Mặt được phát hiện nhưng ngưỡng chất lượng đang chặn nhận diện'
    elif alignment_failure_faces:
        title = 'Cần xử lý chất lượng crop/alignment trước khi kết luận danh tính'
    elif statuses.get('confirmed', 0):
        title = 'Đã có danh tính được xác nhận trong lượt đo'
    else:
        title = 'Đã phân tích; chưa đủ bằng chứng xác nhận danh tính'
    return dict(version=ANALYSIS_VERSION, title=title, run_id=run['id'], scenario=run['scenario'],
        expected_present=bool(run['expected_present']), frames=len(frames),
        preparation_evidence_isolated=preparation_isolated,
        frames_with_face=sum(bool(f.get('overlays')) for f in frames), face_observations=len(faces),
        valid_face_observations=valid_faces, alignment_failure_faces=alignment_failure_faces,
        quality_rejected_faces=quality_rejected_faces, inference_failure_faces=inference_failure_faces,
        high_residual_faces=high_residual_faces,
        tracking_methods=dict(Counter(face.get('tracking',{}).get('method','unmeasured') for face in faces)),
        geometry_only_associations=sum(face.get('tracking',{}).get('method') == 'geometry' for face in faces),
        identity_evidence_resets=sum(bool(face.get('tracking',{}).get('identity_evidence_reset')) for face in faces),
        identity_consensus_observations=consensus, strong_consensus_observations=strong_consensus,
        calibrated_score_samples=calibrated_samples, source_sizes=dict(dimensions), face_pixels=pixels,
        processing_ms=percentiles([f.get('processing_ms') for f in frames]),
        detector_ms=percentiles([f.get('detector_ms') for f in frames]),
        capture_fps=percentiles([s.get('preview_fps') for s in sampling]),
        render_fps=percentiles([s.get('render_fps') for s in sampling]),
        display_delay_ms=percentiles([s.get('display_delay_ms') for s in sampling]),
        late_overlay_results=late_increase,
        late_overlay_results_counter_total=max(late_counters,default=0),
        late_overlay_counter_scope='increase_between_recorded_measurement_samples',
        statuses=dict(statuses), errors=dict(errors), models=models, findings=findings, actions=actions,
        visual_flow=visual_flow_metrics(rows),
        pipeline_revisions=dict(Counter(f.get('pipeline_revision') or 'unversioned' for f in frames)),
        timestamp_bases=dict(Counter(f.get('diagnostic_timestamp_basis','server_receive_legacy') for f in frames)),
        blockers=blockers, single_face_track_changes=tracks_changed,
        caution='Cosine không phải phần trăm tin cậy. Nhãn toàn lượt chưa thay thế nhãn từng khuôn mặt.')


def analyze_protocol(run, rows, review, replay_decisions=None):
    """Partition by captured time, not inference completion or current UI instruction."""
    from .diagnostics import phase_at
    context = run.get('context') or {}
    context = json.loads(context) if isinstance(context,str) else context
    protocol = context.get('protocol')
    if not protocol:
        return None  # Historical single-action measurements retain their original meaning.
    stages = []
    for stage in protocol['stages']:
        all_rows = [(t,f) for t,f in rows if stage['start'] <= t < stage['end']]
        scored = [(t,f) for t,f in all_rows if not phase_at(protocol,t)['transition']]
        transitions = [(t,f) for t,f in all_rows if phase_at(protocol,t)['transition']]
        detail = analyze_frames(run,scored)
        verification = reviewed_metrics(run,scored,review)
        decisions = [(t,face,status,emit) for t,face,status,emit in (replay_decisions or [])
                     if stage['start']+stage['transition_seconds'] <= t < stage['end']]
        failure_ids = [f['frame_id'] for _,f in scored if not f.get('overlays') or
                       any(o.get('status') not in ('confirmed','tracked') for o in f.get('overlays',[]))]
        events = {e['id']:e for _,f in scored for e in f.get('events',[]) if e.get('id')}
        from .fusion import identity_evidence
        evidence_counts = Counter()
        for _,frame in scored:
            for face in frame.get('overlays',[]):
                if face.get('status') in ('confirmed','tracked'):
                    continue
                _,eligible,conflict,_ = identity_evidence(face.get('model_scores',{}))
                evidence_counts['credible_conflict' if conflict else 'fusion_temporal_verification' if eligible
                                else 'insufficient_calibrated_recognition'] += 1
        stages.append(dict(**stage, measured_frames=len(all_rows), evaluated_frames=len(scored),
            transition_frames=len(transitions), transition_statuses=dict(Counter(
                o.get('status','unknown') for _,f in transitions for o in f.get('overlays',[]))),
            metrics=detail, reviewed_metrics=verification,
            action_confirmed=review.get('stage_actions',{}).get(stage['id']) is True,
            condition_scope='user_confirmed_action' if review.get('stage_actions',{}).get(stage['id']) is True else 'scheduled_action_not_verified',
            alerts=len(events), first_alert_elapsed=min((t for t,f in scored if f.get('events')),default=None),
            error_frame_ids=failure_ids,
            replay=dict(statuses=dict(Counter(status for _,_,status,_ in decisions)),
                        alerts=sum(bool(emit) for _,_,_,emit in decisions),
                        scope='frozen_scores_and_recorded_tracks_full_run_continuity'),
            fault_observations=dict(detector_missing=len(scored)-detail['frames_with_face'],
                alignment=detail['alignment_failure_faces'], runtime=detail['inference_failure_faces'],
                quality=detail['quality_rejected_faces'], tracking_resets=detail['identity_evidence_resets'],
                recognition_insufficient=evidence_counts['insufficient_calibrated_recognition'],
                fusion_temporal_verification=evidence_counts['fusion_temporal_verification'],
                credible_model_conflicts=evidence_counts['credible_conflict'],
                unconfirmed=sum(o.get('status') not in ('confirmed','tracked') for _,f in scored for o in f.get('overlays',[]))),
            replay_interval=dict(start=stage['start'],end=stage['end']),
            review_status='requires_cause_review' if failure_ids else 'no_unnamed_frame_observed'))
    transition_rows = [(t,f) for t,f in rows if (phase_at(protocol,t) or {}).get('transition')]
    return dict(protocol=protocol, stages=stages,
        transition_frames=len(transition_rows),
        transition_alerts=len({e['id'] for _,f in transition_rows for e in f.get('events',[]) if e.get('id')}),
        evaluated_frames=sum(s['evaluated_frames'] for s in stages),
        interpretation='Confirmed là bằng chứng nhận diện; tracked chỉ duy trì tên, không phải nhận diện mới. '
                       'Chuyển tiếp được ghi riêng và loại khỏi số đo tình huống. Giai đoạn là lịch hướng dẫn; '
                       'chỉ coi hành động đã thực hiện khi người dùng xác nhận.',
        next_live_run='requires_fresh_consent_after_technical_review')


def read_analysis(store, run_id):
    with store.db() as db:
        row = db.execute('SELECT result FROM diagnostic_analysis WHERE run_id=?', (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


def next_pending(store):
    with store.db() as db:
        row = db.execute('''SELECT d.id FROM diagnostics d LEFT JOIN diagnostic_analysis a ON a.run_id=d.id
            WHERE d.state IN ('complete','cancelled','error') AND (a.run_id IS NULL OR a.version<?)
            ORDER BY d.created_at DESC,d.rowid DESC LIMIT 1''', (ANALYSIS_VERSION,)).fetchone()
    return row[0] if row else None


def analyze_saved(store, run_id):
    with store.db() as db:
        row = db.execute('SELECT * FROM diagnostics WHERE id=?', (run_id,)).fetchone()
        if not row or row['state'] not in ('complete', 'cancelled', 'error'):
            return None
        run = dict(row)
        review_row = db.execute('SELECT details FROM diagnostic_ground_truth WHERE run_id=?', (run_id,)).fetchone()
        review = json.loads(review_row[0]) if review_row else {}
        rows = [(r['elapsed'], json.loads(r['metrics'])) for r in db.execute(
            'SELECT elapsed,metrics FROM diagnostic_frames WHERE run_id=? ORDER BY elapsed', (run_id,))]
        coverage = db.execute('''SELECT expected_present,count(*) n FROM diagnostics WHERE state='complete'
            AND EXISTS(SELECT 1 FROM diagnostic_frames f WHERE f.run_id=diagnostics.id)
            GROUP BY expected_present''').fetchall()
        video_truth = []
        for source in db.execute('SELECT source_hash,annotations FROM video_ground_truth'):
            measured = [json.loads(r[0]) for r in db.execute(
                'SELECT result FROM video_evaluations WHERE source_hash=?', (source['source_hash'],))]
            if any(e.get('state') == 'complete' and e.get('frames', 0) > 0 for e in measured):
                video_truth.append(json.loads(source['annotations']))
    result = analyze_frames(run, rows)
    result['user_review'] = review
    result['reviewed_metrics'] = reviewed_metrics(run, rows, review)
    from .calibration import replay_diagnostic
    replay_decisions = []
    result['calibration_replay'] = replay_diagnostic(store,run,rows,review,
        on_decision=lambda elapsed,frame,face,pid,status,emit,visit: replay_decisions.append((elapsed,face,status,emit)))
    result['phase_analysis'] = analyze_protocol(run,rows,review,replay_decisions)
    result['ground_truth_runs'] = {('positive' if r['expected_present'] else 'negative'): r['n'] for r in coverage}
    result['ground_truth_videos'] = dict(sources=len(video_truth),
        positive_intervals=sum(a['kind']=='present' for labels in video_truth for a in labels),
        negative_intervals=sum(a['kind'] in ('absent','outside') for labels in video_truth for a in labels))
    if not result['ground_truth_runs'].get('negative') and not result['ground_truth_videos']['negative_intervals']:
        result['actions'].append('Dữ liệu hiện chỉ có mục tiêu. Cần video hoặc lượt đối chứng người ngoài watchlist trước khi hiệu chuẩn chống nhận nhầm.')
    elif result['ground_truth_videos']['negative_intervals']:
        result['actions'].append('Đã có nhãn vắng mặt trong video. Cần đối chiếu số embedding âm thực sự của từng model và từng dải chất lượng; không coi số frame liên tiếp là số người đối chứng độc lập.')
    with store.db() as db:
        db.execute('''INSERT INTO diagnostic_analysis(run_id,version,result) VALUES (?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET version=excluded.version,result=excluded.result,updated_at=CURRENT_TIMESTAMP''',
            (run_id, ANALYSIS_VERSION, json.dumps(result, ensure_ascii=False)))
    return result
