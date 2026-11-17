"""Empirical model/quality thresholds from human labels, with separate validation frames."""
import json
import math
from collections import defaultdict
import numpy as np
from .video_evaluation import truth_at
from .fusion import quality_band, quality_basis, model_gates, verification_evidence


def collect_samples(store, versions):
    samples = []
    def add_face(face, truth, source, timestamp, validation_only=False, outside=False):
        for key,score in face.get('model_scores',{}).items():
            version = score.get('version')
            if version != versions.get(key) or score.get('score') is None:
                continue
            value = float(score['score'])
            margin = score.get('margin')
            if not math.isfinite(value) or not -1 <= value <= 1:
                continue
            if margin is not None and (not math.isfinite(margin) or margin < 0):
                continue
            pixels = face.get('quality',{}).get('face_pixels')
            if pixels is None:
                box = face.get('face_bbox',face.get('bbox'))
                if not box:
                    continue
                pixels = min(box[2]-box[0],box[3]-box[1])
            alignment = score.get('alignment_band')
            if alignment is None:
                alignment = 'high_residual' if score.get('alignment_error',0) > 112*.09 else 'regular'
            size=face.get('quality',{}).get('original_face_size')
            box=face.get('face_bbox',face.get('bbox'))
            height=size[1] if size else (box[3]-box[1] if box else pixels)
            base = dict(model=key,version=version,band=quality_band(pixels,alignment,height),
                source=source,time=timestamp,subject=truth,
                fold='validation' if validation_only or source.startswith('video:') else 'fit',
                iqa_threshold=score.get('iqa_threshold'))
            if truth:
                correct = score.get('person_id') == truth
                samples.append(dict(base,positive=True,score=value if correct else None,margin=margin,
                                    label_kind='genuine' if correct else 'ranking_miss'))
                if correct and margin is not None:
                    # Ranking is per identity, so the runner-up differs from the human label.
                    samples.append(dict(base,positive=False,score=value-margin,margin=None,label_kind='runner_impostor'))
                elif not correct:
                    samples.append(dict(base,positive=False,score=value,margin=None,label_kind='wrong_top1'))
            elif outside:
                samples.append(dict(base,positive=False,score=value,margin=None,label_kind='declared_absence'))
    with store.db() as db:
        db.execute('BEGIN')
        generation_row = db.execute("SELECT value FROM settings WHERE key='calibration_generation'").fetchone()
        generation = json.loads(generation_row[0]) if generation_row else 0
        runs = db.execute('''SELECT d.*,g.details FROM diagnostics d JOIN diagnostic_ground_truth g ON g.run_id=d.id
            WHERE d.state='complete' AND d.expected_present=1''').fetchall()
        for row in runs:
            review = json.loads(row['details'])
            if not all(review.get(key) is True for key in ('only_target','face_visible_throughout','boxes_match_target')):
                continue
            for measurement in db.execute('SELECT elapsed,metrics FROM diagnostic_frames WHERE run_id=? ORDER BY elapsed',(row['id'],)):
                metrics=json.loads(measurement['metrics'])
                if (metrics.get('diagnostic_phase') or {}).get('transition'):
                    continue
                faces=metrics.get('overlays',[])
                if len(faces)==1:
                    context=json.loads(row['context'] or '{}')
                    add_face(faces[0],row['person_id'],'live:'+row['id'],measurement['elapsed'],
                             validation_only=context.get('calibration_role')=='validation')
        evaluations = db.execute('''SELECT e.*,(SELECT count(*) FROM video_face_ground_truth f WHERE f.run_id=e.run_id) labels
            FROM video_evaluations e ORDER BY labels DESC,e.updated_at DESC,e.rowid DESC''').fetchall()
        used = set()
        for row in evaluations:
            result = json.loads(row['result'])
            source_hash = row['source_hash']
            if result.get('state') != 'complete' or source_hash in used:
                continue
            used.add(source_hash)
            source = db.execute('SELECT annotations FROM video_ground_truth WHERE source_hash=?',(source_hash,)).fetchone()
            annotations = json.loads(source[0]) if source else []
            labels = {(r['frame_id'],r['face_index']):json.loads(r['details']) for r in db.execute(
                'SELECT frame_id,face_index,details FROM video_face_ground_truth WHERE run_id=?',(row['run_id'],))}
            for measurement in db.execute('SELECT frame_id,timestamp,metrics FROM video_measurements WHERE run_id=?',(row['run_id'],)):
                data = json.loads(measurement['metrics'])
                if data.get('enrollment_changed'):
                    continue
                for index,face in enumerate(data.get('overlays',[])):
                    label = labels.get((measurement['frame_id'],index))
                    box = face.get('face_bbox',face.get('bbox'))
                    if label and label.get('bbox') == box:
                        add_face(face,label['person_id'] if label['kind']=='person' else None,
                            'video:'+source_hash,measurement['timestamp'],validation_only=True,
                            outside=label['kind']=='outside')
                    else:
                        # An interval saying someone is present cannot label an individual face.
                        absent = {key:score for key,score in face.get('model_scores',{}).items()
                                  if truth_at(annotations,score.get('person_id'),measurement['timestamp']) is False}
                        add_face(dict(face,model_scores=absent),None,'video:'+source_hash,
                                 measurement['timestamp'],outside=True)
        for probe in db.execute('SELECT * FROM model_probes'):
            result=json.loads(probe['result'])
            if (result.get('state')!='complete' or probe['model'] not in versions.values() or
                    result.get('enrollment_changed') or not result.get('frames') or result.get('frames')!=result.get('total')):
                continue
            truth=db.execute('SELECT annotations FROM video_ground_truth WHERE source_hash=?',(probe['source_hash'],)).fetchone()
            annotations=json.loads(truth[0]) if truth else []
            for row in db.execute('SELECT timestamp,metrics FROM video_measurements WHERE run_id=? ORDER BY frame_id',(probe['run_id'],)):
                for raw_face in json.loads(row['metrics']).get('overlays',[]):
                    face=dict(raw_face,model_scores={key:value for key,value in raw_face.get('model_scores',{}).items()
                                                    if value.get('version')==probe['model']})
                    label=face.get('ground_truth')
                    if label and label.get('bbox')==face.get('face_bbox'):
                        add_face(face,label.get('person_id') if label.get('kind')=='person' else None,
                            'probe:'+probe['source_hash'],row['timestamp'],validation_only=True,outside=label.get('kind')=='outside')
                    else:
                        absent={key:value for key,value in face.get('model_scores',{}).items()
                                if truth_at(annotations,value.get('person_id'),row['timestamp']) is False}
                        add_face(dict(face,model_scores=absent),None,'probe:'+probe['source_hash'],row['timestamp'],validation_only=True,outside=True)
    return generation,samples


def block_values(samples, field):
    blocks = defaultdict(list)
    for sample in samples:
        value = sample.get(field)
        if value is not None:
            blocks[sample['source'],int(sample['time']),sample.get('subject')].append(value)
    return [float(np.median(values)) for values in blocks.values()]


def passes(sample, threshold, margin):
    return sample.get('score') is not None and sample['score'] >= threshold and (
        sample.get('margin') is None or sample['margin'] >= margin)


def historical_identity_gates(store):
    """Read previously deployed gates frozen with reviewed, unchanged enrollment."""
    history = defaultdict(list)
    galleries = defaultdict(list)
    with store.db() as db:
        current_enrollment = sorted(row[0] for row in db.execute('SELECT id FROM images'))
        for row in db.execute('SELECT model,image_id FROM model_embeddings'):
            galleries[row['model']].append(row['image_id'])
        rows = db.execute("""SELECT d.id,d.context,g.details FROM diagnostics d
            JOIN diagnostic_ground_truth g ON g.run_id=d.id WHERE d.state='complete'""")
        for row in rows:
            review = json.loads(row['details'])
            if not all(review.get(key) is True for key in ('only_target','face_visible_throughout','boxes_match_target')):
                continue
            context = json.loads(row['context'] or '{}')
            enrollment_images = sorted(image for person in context.get('enrollment', [])
                                       for image in person.get('image_ids', []))
            for snapshot in context.get('calibration_profile_snapshot', context.get('validation_profile_snapshot', [])):
                version = snapshot.get('version')
                if not enrollment_images or enrollment_images != current_enrollment or enrollment_images != sorted(galleries.get(version, [])):
                    continue
                threshold, margin = snapshot.get('threshold'), snapshot.get('margin')
                if not all(isinstance(value, (int,float)) and math.isfinite(value) for value in (threshold,margin)):
                    continue
                if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
                    continue
                history[version,snapshot['band']].append(dict(snapshot,source_run_id=row['id']))
    return history, {version: sorted(images) for version,images in galleries.items()}


def retain_reviewed_identity_gates(config, values, candidates, gallery_images, stress_negatives):
    """A runner-up for B must not tighten A when A's own negative bounds remain safe.

    Scores from other or unlabelled faces remain conservative bounds, including
    unobserved lower-ranked references in a gallery with more than two people.
    A human-labelled genuine face cannot become a negative for its own identity.
    """
    if not candidates or not gallery_images:
        return
    fitting_positives = [sample for sample in values if sample['positive'] and sample['fold']=='fit']
    negatives = [sample for sample in values if not sample['positive']] + stress_negatives
    gates = {}
    for identity in {sample['subject'] for sample in fitting_positives}:
        positives = [sample for sample in fitting_positives if sample['subject']==identity]
        if len(block_values(positives,'score')) < 3:
            continue
        relevant_negatives = [sample for sample in negatives if not (
            sample.get('subject') == identity and
            sample.get('label_kind') in ('runner_impostor','wrong_top1'))]
        held_out = [sample for sample in relevant_negatives if sample['fold']=='validation']
        if not held_out:
            continue
        for candidate in sorted(candidates,key=lambda item:(item['threshold'],item['margin'])):
            threshold, margin = candidate['threshold'], candidate['margin']
            if threshold >= config['threshold'] and margin >= config['margin']:
                continue
            if candidate.get('quality_basis') != config['quality_basis']:
                continue
            if any(passes(sample,threshold,margin) for sample in relevant_negatives):
                continue
            qualified = [sample for sample in positives if passes(sample,threshold,margin)]
            if len(block_values(qualified,'score')) < 3:
                continue
            verification_gate = dict(threshold=threshold,margin=margin,verification_enabled=True,numeric_guard=.001)
            verification_false = sum(model_gates(sample['score'],sample.get('margin'),verification_gate)[1]
                                     for sample in relevant_negatives)
            gates[identity] = dict(threshold=threshold,margin=margin,
                verification_enabled=verification_false==0,
                verification_false_passes=verification_false,
                validation_negative_observations=len(held_out),validation_false_passes=0,
                fit_positive_blocks=len(block_values(qualified,'score')),
                source_snapshot=candidate['source_run_id'],origin='retained_reviewed_identity_gate')
            break
    if gates:
        config['identity_gates'] = gates
        config['identity_gates_gallery_images'] = gallery_images


def fit_profiles(store, versions):
    generation,samples = collect_samples(store,versions)
    historical_gates,gallery_images = historical_identity_gates(store)
    with store.db() as db:
        previous = {(row['model'],row['quality_band']):json.loads(row['config'])
                    for row in db.execute('SELECT * FROM calibrations')}
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['version'],sample['band']].append(sample)
    profiles,unavailable = [],[]
    for (version,band),values in grouped.items():
        fitting = [sample for sample in values if sample['fold']=='fit']
        positives = [sample for sample in fitting if sample['positive']]
        negatives = [sample for sample in fitting if not sample['positive']]
        positive_scores = block_values(positives,'score')
        if len(positive_scores)<3 or not negatives:
            unavailable.append(dict(model=version,band=band,reason='insufficient_labelled_fit_samples'))
            continue
        negative_max = max(sample['score'] for sample in negatives)
        threshold = max(float(np.percentile(positive_scores,10)),negative_max+.001)
        old = previous.get((version,band))
        all_negatives = [sample for sample in values if not sample['positive']]
        retained = bool(old and old.get('enabled') is not False and
                        old.get('protocol')=='reviewed_live_fit_video_holdout_v2' and
                        (not band.endswith('_high_residual') or old.get('quality_basis')==quality_basis('high_residual')) and
                        math.isfinite(old.get('threshold',float('nan'))) and math.isfinite(old.get('margin',float('nan'))) and
                        not any(passes(sample,old['threshold'],old['margin']) for sample in all_negatives))
        if retained:
            threshold = old['threshold']
        qualified = [sample for sample in positives if sample.get('score') is not None and sample['score']>=threshold]
        if len(block_values(qualified,'score'))<3:
            unavailable.append(dict(model=version,band=band,reason='genuine_impostor_overlap'))
            continue
        margins = block_values(qualified,'margin')
        margin = old['margin'] if retained else (max(.001,float(np.percentile(margins,10))) if margins else 0.)
        validation = [sample for sample in values if sample['fold']=='validation']
        validation_negatives = [sample for sample in validation if not sample['positive']]
        validation_positives = [sample for sample in validation if sample['positive']]
        false_passes = sum(passes(sample,threshold,margin) for sample in validation_negatives)
        if false_passes:
            unavailable.append(dict(model=version,band=band,reason='validation_impostor_passes',count=false_passes))
            continue
        stress_negatives = [sample for sample in samples if band.endswith('_high_residual') and
                            sample['version']==version and sample['fold']=='validation' and not sample['positive'] and
                            sample['band'].endswith('_high_residual')]
        stress_false = sum(passes(sample,threshold,margin) for sample in stress_negatives)
        if stress_false:
            unavailable.append(dict(model=version,band=band,reason='cross_quality_impostor_passes',count=stress_false))
            continue
        verification_negatives = all_negatives + stress_negatives
        verification_gate = dict(threshold=threshold,margin=margin,verification_enabled=True,numeric_guard=.001)
        verification_false = sum(model_gates(sample['score'],sample.get('margin'),verification_gate)[1]
                                 for sample in verification_negatives)
        iqa = {sample['iqa_threshold'] for sample in values if sample['iqa_threshold'] is not None}
        config = dict(enabled=True,threshold=threshold,margin=margin,label_generation=generation,
            verification_enabled=bool(verification_negatives) and verification_false==0,
            verification_negative_observations=len(verification_negatives),
            verification_false_passes=verification_false,
            threshold_origin='retained_after_new_label_revalidation' if retained else 'fitted_from_labelled_blocks',
            quality_basis=quality_basis('high_residual' if band.endswith('_high_residual') else 'regular'),
            negative_max=negative_max,positive_reference=max(threshold+.001,float(np.median(positive_scores))),
            protocol='reviewed_live_fit_video_holdout_v2',positive_quantile=.1,numeric_guard=.001,
            fit_positive_observations=len(positives),fit_negative_observations=len(negatives),
            fit_positive_blocks=len(positive_scores),validation_negative_observations=len(validation_negatives),
            validation_false_passes=0 if validation_negatives else None,validation_positive_observations=len(validation_positives),
            cross_quality_negative_observations=len(stress_negatives),
            cross_quality_false_passes=0 if stress_negatives else None,
            validation_positive_passes=sum(passes(sample,threshold,margin) for sample in validation_positives),
            positive_subjects=sorted({sample['subject'] for sample in values if sample['positive']}),
            sources=sorted({sample['source'] for sample in values}),
            fit_sources=sorted({sample['source'] for sample in fitting}),
            validation_sources=sorted({sample['source'] for sample in validation}),
            fit_positive_subjects=sorted({sample['subject'] for sample in positives}),
            evidence_scope='reviewed_live_fit_independent_video_validation',
            status='preliminary',validation_source_count=len({sample['source'] for sample in validation}),
            limitation='Live positives primarily cover one subject; new people and conditions need independent validation.')
        if len(iqa)==1:
            config['iqa_threshold'] = next(iter(iqa))
        retain_reviewed_identity_gates(config,values,historical_gates.get((version,band),[]),
                                      gallery_images.get(version,[]),stress_negatives)
        profiles.append(dict(model=version,band=band,config=config))
    return dict(label_generation=generation,profiles=profiles,unavailable=unavailable,
                sample_count=len(samples),note='Reviewed Live runs fit thresholds; a separate user-labelled video validates them. Unseen identities and conditions are not certified.')


def apply_profiles(store, result, versions):
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        current = db.execute("SELECT value FROM settings WHERE key='calibration_generation'").fetchone()
        generation = json.loads(current[0]) if current else 0
        if generation != result['label_generation']:
            raise ValueError('Nhãn đối chứng đã thay đổi trong khi hiệu chuẩn. Hãy tính lại.')
        current_versions = set(versions.values())
        if not result['profiles']:
            raise ValueError('Chưa đủ dữ liệu có nhãn để áp dụng ngưỡng.')
        for profile in result['profiles']:
            config = profile['config']
            if (profile['model'] not in current_versions or config.get('label_generation') != generation or
                    not math.isfinite(config['threshold']) or not -1 <= config['threshold'] <= 1 or
                    not math.isfinite(config['margin']) or not 0 <= config['margin'] <= 2 or
                    config.get('validation_false_passes') not in (None,0) or
                    config.get('cross_quality_false_passes') not in (None,0)):
                raise ValueError('Ngưỡng hiệu chuẩn không hợp lệ hoặc chưa vượt kiểm chứng.')
        for profile in result['profiles']:
            config = profile['config']
            if not config.get('identity_gates'):
                continue
            gallery_images = sorted(row[0] for row in db.execute(
                'SELECT image_id FROM model_embeddings WHERE model=?', (profile['model'],)))
            enrollment_images = sorted(row[0] for row in db.execute('SELECT id FROM images'))
            if not gallery_images or gallery_images != config.get('identity_gates_gallery_images') or enrollment_images != gallery_images:
                raise ValueError('Gallery changed while validating identity gates.')
            for identity, gate in config['identity_gates'].items():
                if (not identity or not math.isfinite(gate['threshold']) or not -1 <= gate['threshold'] <= 1 or
                        not math.isfinite(gate['margin']) or not 0 <= gate['margin'] <= 2 or
                        gate.get('validation_false_passes') != 0 or gate.get('validation_negative_observations',0) <= 0 or
                        gate.get('fit_positive_blocks',0) < 3 or
                        gate.get('verification_enabled') and gate.get('verification_false_passes') != 0):
                    raise ValueError('Identity gate lacks valid labelled evidence.')
        revision = db.execute("SELECT value FROM settings WHERE key='calibration_revision'").fetchone()
        db.execute("INSERT OR REPLACE INTO settings VALUES ('calibration_revision',?)",
                   (json.dumps((json.loads(revision[0]) if revision else 0)+1),))
        for row in db.execute('SELECT model,quality_band,config FROM calibrations').fetchall():
            if row['model'] in current_versions:
                config = dict(json.loads(row['config']),enabled=False)
                db.execute('UPDATE calibrations SET config=? WHERE model=? AND quality_band=?',
                           (json.dumps(config),row['model'],row['quality_band']))
        db.execute('UPDATE diagnostic_analysis SET version=0')
        for profile in result['profiles']:
            if profile['model'] not in current_versions:
                raise ValueError('Version model hiệu chuẩn không khớp hệ thống đang chạy.')
            db.execute('INSERT OR REPLACE INTO calibrations VALUES (?,?,?)',
                       (profile['model'],profile['band'],json.dumps(profile['config'])))
    return store.settings(versions)

def replay_diagnostic(store, run, rows, review, *, on_decision=None):
    """Replay frozen model scores/recorded tracks, never claim a fresh camera test."""
    from .fusion import identity_evidence
    from .temporal import Temporal, stable_association
    from .identity_tracking import trusted_identity_link
    from collections import Counter
    settings = store.settings()
    if not settings['calibrated']:
        return None
    versions = {p['model']:p['version'] for p in settings['calibration_profiles']}
    selected = store.selected_ids()
    state = Temporal()
    previously_visible = set()
    previous_faces = []
    known = all(review.get(k) is True for k in ('only_target','face_visible_throughout','boxes_match_target'))
    known = known and bool(run.get('expected_present')) and all(len(f.get('overlays',[]))<=1 for _,f in rows)
    summary = dict(frames=len(rows),eligible_faces=0,confirmed_faces=0,alerts=0,
                   correct_confirmed=0 if known else None,wrong_confirmed=0 if known else None,
                   first_confirmed_elapsed=None,identity_verified=known,
                   calibration_revision=settings.get('calibration_revision',0),
                   used_for_fitting=any('live:'+run['id'] in p.get('fit_sources',p.get('sources',[])) for p in settings['calibration_profiles']),
                   caution='Phát lại điểm model và track đã lưu; không chạy lại detector/embedding, không phải lượt Camo mới.')
    summary.update(tracked_faces=0,correct_tracked=0 if known else None,wrong_tracked=0 if known else None)
    for elapsed,frame in rows:
        clock_time=elapsed
        if frame.get('live_timing_revision')=='camera-media-clock-v1':
            clock_time=frame.get('source_timestamp')
            if not isinstance(clock_time,(int,float)) or not math.isfinite(clock_time):
                raise ValueError('Replay thiếu timestamp camera gốc.')
        entries = []
        for face in frame.get('overlays',[]):
            scores = {}
            for key,original in face.get('model_scores',{}).items():
                score = dict(original)
                pixels = face.get('quality',{}).get('face_pixels',0)
                box=face.get('face_bbox',face.get('bbox'));size=face.get('quality',{}).get('original_face_size')
                height=size[1] if size else (box[3]-box[1] if box else pixels)
                profile = store.calibration(versions.get(key,''),pixels,score.get('alignment_band','regular'),height) if score.get('version') == versions.get(key) else None
                value,margin = score.get('score'),score.get('margin')
                score['calibrated'] = bool(profile)
                identify,verify=model_gates(value,margin,profile,score.get('person_id'),score.get('runner_up_person_id'))
                score['passed'] = bool(profile and identify and not score.get('error'))
                score['verification_passed'] = bool(profile and verify and not score.get('error'))
                if key == 'petalface':
                    score['iqa_threshold_verified'] = bool(profile and profile.get('iqa_threshold')==score.get('iqa_threshold'))
                scores[key] = score
            entries.append((face,identity_evidence(scores),verification_evidence(scores,
                None if face.get('tracking',{}).get('identity_evidence_reset',False) else state.anchor_identity(face['track_id'],clock_time))))
        claims=Counter()
        for face,(pid,eligible,_,_),(vpid,veligible,vconflict,_) in entries:
            reverify=bool(veligible and not vconflict and not face.get('tracking',{}).get('identity_evidence_reset',False) and
                          state.can_reverify(face['track_id'],vpid,clock_time))
            claim=pid if eligible else (vpid if reverify else None)
            if claim in selected:claims[claim]+=1
        for face,(pid,eligible,conflict,info),(vpid,veligible,vconflict,vinfo) in entries:
            track=face['track_id']
            if face.get('tracking',{}).get('identity_evidence_reset',False):state.reset_track(track)
            previous=state.tracks.get(track)
            anchor_pid=state.anchor_identity(track,clock_time)
            trusted_pid=anchor_pid or (previous['pid'] if previous else None)
            held=anchor_pid is not None
            if held and vconflict:conflict=True;eligible=False
            reverify=bool(not eligible and held and not conflict and veligible and state.can_reverify(track,vpid,clock_time))
            if reverify:pid=vpid;eligible=True
            duplicate=eligible and claims[pid]>1
            eligible=eligible and not duplicate and pid in selected and not conflict
            direct_eligible=eligible
            qualified_conflict=bool(trusted_pid and (info['calibrated_support'] and trusted_pid!=pid or
                held and vinfo['calibrated_support'] and vpid and trusted_pid!=vpid))
            summary['reverified_faces']=summary.get('reverified_faces',0)+bool(reverify and eligible)
            bridge = bool(not eligible and pid in selected and info['calibrated_support'] >= 1 and
                          not conflict and not qualified_conflict and not claims[pid] and state.can_bridge(track,pid,clock_time,settings))
            eligible = eligible or bridge
            summary['eligible_faces'] += eligible
            summary['bridged_faces'] = summary.get('bridged_faces',0)+bridge
            resume=bool(direct_eligible and held and pid==anchor_pid and track in previously_visible and
                        stable_association(face.get('tracking',{})))
            fresh_count=vinfo['calibrated_support'] if reverify else info['calibrated_support']
            continuity=bool(not eligible and not conflict and not duplicate and not qualified_conflict and anchor_pid in selected and not claims[anchor_pid] and
                trusted_identity_link(face,previous_faces,frame.get('overlays',[]),settings.get('identity_tracking_profile')))
            status,emit,visit = state.update(track,pid if eligible else None,clock_time,settings,
                reset=bool(conflict or duplicate or qualified_conflict),multimodel=direct_eligible,reverified=resume,fresh_model_count=fresh_count,continuity=continuity)
            if status=='tracked':
                pid=state.label_chains[track];summary['tracked_faces']+=1
                if known:summary['correct_tracked' if pid==run['person_id'] else 'wrong_tracked']+=1
            if on_decision is not None:
                on_decision(elapsed,frame,face,pid if status in ('confirmed','tracked') else None,status,emit,visit)
            if status == 'confirmed':
                summary['confirmed_faces'] += 1
                summary['alerts'] += emit
                if summary['first_confirmed_elapsed'] is None:
                    summary['first_confirmed_elapsed'] = elapsed
                if known:
                    summary['correct_confirmed' if pid == run['person_id'] else 'wrong_confirmed'] += 1
        state.finish_frame(clock_time)
        previous_faces=frame.get('overlays',[])
        previously_visible={face['track_id'] for face in previous_faces}
    return summary
