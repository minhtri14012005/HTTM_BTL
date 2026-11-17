"""User-labelled video intervals; never infer ground truth from model scores."""
import json
import base64
import cv2
import math
import sqlite3
from collections import Counter
from contextlib import closing
import numpy as np


def save_measurement(store, run_id, result):
    metrics = {k: result.get(k) for k in ('frame_id','width','height','processing_ms','detector_ms','pipeline_revision',
                                         'overlays','enrollment_changed','recognition_ms')}
    metrics['events'] = [{k:e.get(k) for k in ('id','person_id','timestamp','track_id','confirmation_seconds')}
                         for e in result.get('events', [])]
    with store.db() as db:
        db.execute('INSERT OR IGNORE INTO video_measurements VALUES (?,?,?,?)',
                   (run_id,result['frame_id'],result['timestamp'],json.dumps(metrics)))


def measurement_rows(job):
    # Current playback cache or retained metrics after the temporary video closes.
    if job.db_path.exists():
        with closing(sqlite3.connect(job.db_path, timeout=20)) as db:
            yield from db.execute('SELECT timestamp,result FROM frames ORDER BY id')
    else:
        with job.store.db() as db:
            yield from db.execute('SELECT timestamp,metrics FROM video_measurements WHERE run_id=? ORDER BY frame_id', (job.id,))


def validate_annotations(items, duration, enrollment):
    if len(items) > 2000:
        raise ValueError('Tối đa 2000 khoảng nhãn cho một video.')
    people = {p['person_id'] for p in enrollment}
    watchlist = {p['person_id'] for p in enrollment if p['selected']}
    clean = []
    for item in items:
        start, end = float(item['start']), float(item['end'])
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= duration + .001:
            raise ValueError('Khoảng thời gian phải nằm trong video, bắt đầu nhỏ hơn kết thúc.')
        kind, pid = item.get('kind'), item.get('person_id')
        if kind not in ('present', 'absent', 'outside') or (kind != 'outside' and pid not in people):
            raise ValueError('Chọn người đã đăng ký hoặc khoảng không có ai trong watchlist.')
        scope = sorted(set(item.get('scope', watchlist)) & people) if kind == 'outside' else [pid]
        if not scope:
            raise ValueError('Khoảng không có mục tiêu cần danh sách người để đối chứng.')
        clean.append(dict(start=start, end=min(end, duration), kind=kind,
                          person_id=pid if kind != 'outside' else None, scope=scope,
                          note=str(item.get('note', ''))[:500],
                          different_source=bool(item.get('different_source', False))))
    for i, a in enumerate(clean):
        for b in clean[:i]:
            if max(a['start'], b['start']) < min(a['end'], b['end']) and set(a['scope']) & set(b['scope']):
                if (a['kind'] == 'present') != (b['kind'] == 'present'):
                    raise ValueError('Nhãn mâu thuẫn: cùng người vừa có mặt vừa vắng mặt trong cùng khoảng.')
    return sorted(clean, key=lambda a: (a['start'], a['end'], a['kind']))


def truth_at(annotations, person_id, timestamp):
    labels = [a['kind'] == 'present' for a in annotations
              if a['start'] <= timestamp < a['end'] and person_id in a['scope']]
    return labels[0] if labels else None


def read_truth(job):
    with job.store.db() as db:
        row = db.execute('SELECT annotations FROM video_ground_truth WHERE source_hash=?', (job.source_hash,)).fetchone()
    return json.loads(row['annotations']) if row else []


def save_truth(job, annotations):
    clean = validate_annotations(annotations, job.duration, job.enrollment)
    with job.store.db() as db:
        previous = db.execute('SELECT annotations FROM video_ground_truth WHERE source_hash=?',(job.source_hash,)).fetchone()
        if previous is None or json.loads(previous[0]) != clean:
            job.store.labels_changed(db)
        db.execute('''INSERT INTO video_ground_truth(source_hash,filename,duration,annotations) VALUES (?,?,?,?)
            ON CONFLICT(source_hash) DO UPDATE SET annotations=excluded.annotations,updated_at=CURRENT_TIMESTAMP''',
            (job.source_hash, job.filename, job.duration, json.dumps(clean)))
    return clean



def frame_measurement(job, frame_id):
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError('Frame ID không hợp lệ.')
    with job.store.db() as db:
        row = db.execute('SELECT timestamp,metrics FROM video_measurements WHERE run_id=? AND frame_id=?',
                         (job.id, frame_id)).fetchone()
    if row is None:
        raise ValueError('Frame chưa được AI phân tích. Hãy chờ hoặc chọn vị trí đã phân tích.')
    return dict(json.loads(row['metrics']), frame_id=frame_id, timestamp=row['timestamp'])


def read_face_truth(job, frame_id=None):
    with job.store.db() as db:
        rows = db.execute('SELECT details FROM video_face_ground_truth WHERE run_id=? ORDER BY frame_id,face_index',
                          (job.id,)).fetchall()
    labels = [json.loads(row['details']) for row in rows]
    return [label for label in labels if frame_id is None or label['frame_id'] == frame_id]


def annotation_faces(job, frame_id):
    measurement = frame_measurement(job, frame_id)
    capture = cv2.VideoCapture(str(job.path))
    try:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
        if not ok or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_id+1:
            raise ValueError('Không đọc được đúng frame video để đối chứng.')
        pts = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
        if abs(pts-measurement['timestamp']) > .5/job.fps+.001:
            raise ValueError('Timestamp ảnh đối chứng lệch frame AI; cần kiểm tra bộ giải mã.')
    finally:
        capture.release()
    if image.shape[:2] != (measurement['height'],measurement['width']):
        raise ValueError('Kích thước ảnh đối chứng khác frame AI.')
    faces = []
    for index, face in enumerate(measurement.get('overlays', [])):
        box = face.get('face_bbox', face.get('bbox'))
        if not box or len(box) != 4 or not np.isfinite(box).all():
            raise ValueError('Box không hợp lệ để gán nhãn.')
        x1,y1,x2,y2 = np.asarray(box, dtype=float)
        x1,y1 = max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
        x2,y2 = min(image.shape[1],int(np.ceil(x2))),min(image.shape[0],int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Crop mặt nằm ngoài frame video.')
        ok, encoded = cv2.imencode('.jpg',image[y1:y2,x1:x2],[cv2.IMWRITE_JPEG_QUALITY,97])
        if not ok:
            raise ValueError('Không tạo được ảnh đối chứng.')
        faces.append(dict(face_index=index,bbox=box,
                          thumbnail='data:image/jpeg;base64,'+base64.b64encode(encoded).decode('ascii')))
    return dict(frame_id=frame_id,timestamp=measurement['timestamp'],faces=faces,
                labels=read_face_truth(job,frame_id),enrollment=job.enrollment)


def save_face_truth(job, frame_id, items):
    measurement = frame_measurement(job,frame_id)
    if measurement.get('enrollment_changed'):
        raise ValueError('Hồ sơ đã đổi trong lượt video; cần một lượt nhất quán để đối chứng danh tính.')
    faces = measurement.get('overlays', [])
    if len(items) > len(faces):
        raise ValueError('Số nhãn vượt số mặt thực sự được phát hiện.')
    people = {person['person_id'] for person in job.enrollment}
    intervals = read_truth(job)
    clean, used = [], set()
    for item in items:
        index = item.get('face_index')
        if isinstance(index,bool) or not isinstance(index,int) or not 0 <= index < len(faces) or index in used:
            raise ValueError('Chọn từng mặt được phát hiện đúng một lần.')
        used.add(index)
        kind,pid = item.get('kind'),item.get('person_id')
        if kind not in ('person','outside') or (kind == 'person' and pid not in people):
            raise ValueError('Chọn danh tính thật hoặc người ngoài danh sách đăng ký.')
        if kind == 'person' and truth_at(intervals,pid,measurement['timestamp']) is False:
            raise ValueError('Nhãn mặt mâu thuẫn nhãn vắng mặt đã lưu. Hãy sửa khoảng thời gian trước.')
        clean.append(dict(frame_id=frame_id,face_index=index,timestamp=measurement['timestamp'],
            bbox=faces[index].get('face_bbox',faces[index].get('bbox')),kind=kind,
            person_id=pid if kind == 'person' else None,scope=sorted(people),
            different_source=bool(item.get('different_source',False)),source='user_face_confirmation'))
    # Replace only the selected frame's annotation set after validation, in one transaction.
    with job.store.db() as db:
        previous = {r['face_index']:json.loads(r['details']) for r in db.execute(
            'SELECT face_index,details FROM video_face_ground_truth WHERE run_id=? AND frame_id=?',(job.id,frame_id))}
        if previous != {label['face_index']:label for label in clean}:
            job.store.labels_changed(db)
        db.execute('DELETE FROM video_face_ground_truth WHERE run_id=? AND frame_id=?',(job.id,frame_id))
        for label in clean:
            db.execute('INSERT INTO video_face_ground_truth(run_id,source_hash,frame_id,face_index,details) VALUES (?,?,?,?,?)',
                (job.id,job.source_hash,frame_id,label['face_index'],json.dumps(label)))
    return clean


def reuse_face_truth(job):
    """Copy human labels only for identical source/frame/PTS/box, never by identity score."""
    people = {person['person_id'] for person in job.enrollment}
    copied = skipped = 0
    with job.store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        candidates = db.execute("SELECT * FROM video_face_ground_truth WHERE source_hash=? AND run_id!=? ORDER BY updated_at DESC",(job.source_hash,job.id)).fetchall()
        groups = {}
        for row in candidates:
            label = json.loads(row['details'])
            key = (row['frame_id'],tuple(label.get('bbox',[])))
            groups.setdefault(key,[]).append((row,label))
        for (frame_id,box),records in groups.items():
            if len({(label.get('kind'),label.get('person_id')) for _,label in records}) != 1:
                skipped += 1
                continue
            old,label = records[0]
            if (label['kind']=='person' and label['person_id'] not in people) or (label['kind']=='outside' and not people.issubset(label.get('scope',[]))):
                skipped += 1
                continue
            current = db.execute('SELECT timestamp,metrics FROM video_measurements WHERE run_id=? AND frame_id=?',(job.id,frame_id)).fetchone()
            original = db.execute('SELECT timestamp,metrics FROM video_measurements WHERE run_id=? AND frame_id=?',(old['run_id'],frame_id)).fetchone()
            if current is None or original is None:
                skipped += 1
                continue
            now,before = json.loads(current['metrics']),json.loads(original['metrics'])
            matches = [index for index,face in enumerate(now.get('overlays',[])) if tuple(face.get('face_bbox',face.get('bbox',[]))) == box]
            if (len(matches)!=1 or abs(current['timestamp']-original['timestamp'])>.001 or
                    now.get('enrollment_changed') or before.get('enrollment_changed') or
                    (now.get('width'),now.get('height')) != (before.get('width'),before.get('height'))):
                skipped += 1
                continue
            index = matches[0]
            details = dict(label,frame_id=frame_id,face_index=index,timestamp=current['timestamp'],
                           source='user_label_reused_exact_frame',source_run_id=old['run_id'])
            cursor = db.execute('INSERT OR IGNORE INTO video_face_ground_truth(run_id,source_hash,frame_id,face_index,details) VALUES (?,?,?,?,?)',
                (job.id,job.source_hash,frame_id,index,json.dumps(details)))
            copied += cursor.rowcount
    return dict(copied=copied,skipped=skipped,labels=read_face_truth(job))


def evaluate(job):
    annotations = read_truth(job)
    frames = face_count = confirmed = unsupported = 0
    latencies, tracks, revisions, models = [], set(), set(), {}
    first = [None] * len(annotations)
    observations = [0] * len(annotations)
    events, event_truth = {}, {}
    through = -1.
    enrollment_changed = False
    stages, face_pixels = Counter(), []
    face_labels = {(label['frame_id'],label['face_index']):label for label in read_face_truth(job)}
    labelled = dict(labeled_faces=0,known_faces=0,outside_faces=0,confirmed_correct=0,false_accepts=0,
                    missed_known_faces=0,rejected_outsiders=0,different_source_known_faces=0,models={})
    for timestamp, raw in measurement_rows(job):
        result = json.loads(raw)
        enrollment_changed |= bool(result.get('enrollment_changed'))
        through = max(through, timestamp)
        frames += 1
        latencies.append(result['processing_ms'])
        revisions.add(result.get('pipeline_revision', 'unknown'))
        faces = result.get('overlays', [])
        face_count += len(faces)
        for idx, a in enumerate(annotations):
            if a['start'] <= timestamp < a['end']:
                observations[idx] += 1
                if a['kind'] == 'present' and first[idx] is None and any(
                        o.get('status') == 'confirmed' and o.get('person_id') == a['person_id'] for o in faces):
                    first[idx] = timestamp
        labels_by_track = {}
        for face_index, face in enumerate(faces):
            label = face_labels.get((result.get('frame_id'),face_index))
            box = face.get('face_bbox', face.get('bbox'))
            if box:
                face_pixels.append(min(box[2]-box[0],box[3]-box[1]))
            if label and (box is None or label['bbox'] != box):
                label = None  # Labels are bound to the measured face, never transferred by track ID.
            if label:
                labels_by_track[face.get('track_id')] = label
                known = label['kind'] == 'person'
                confirmed_face = face.get('status') == 'confirmed'
                correct = confirmed_face and known and face.get('person_id') == label['person_id']
                labelled['labeled_faces'] += 1
                labelled['known_faces' if known else 'outside_faces'] += 1
                labelled['confirmed_correct'] += bool(correct)
                labelled['false_accepts'] += bool(confirmed_face and not correct)
                labelled['missed_known_faces'] += bool(known and not correct)
                labelled['rejected_outsiders'] += bool(not known and not confirmed_face)
                labelled['different_source_known_faces'] += bool(known and label['different_source'])
            reason = face.get('reason') or ''
            scores = face.get('model_scores', {})
            if not scores and reason.startswith(('Mặt quá nhỏ:', 'Mặt mờ:')):
                stages['quality_rejected_faces'] += 1
            errors = [s['error'] for s in scores.values() if s.get('error')]
            alignment_errors = [e for e in errors if any(p in e for p in
                ('Alignment','Landmarks','landmarks','Crop mặt','căn chỉnh'))]
            stages['alignment_rejected_faces'] += bool(alignment_errors)
            stages['inference_error_faces'] += any(e not in alignment_errors for e in errors)
            if face.get('track_id') is not None:
                tracks.add(face['track_id'])
            if face.get('status') == 'confirmed':
                confirmed += 1
                unsupported += truth_at(annotations, face.get('person_id'), timestamp) is False
            for key, score in scores.items():
                info = models.setdefault(key, dict(samples=0, errors=0, scores=[], versions=set(), calibrated=0,
                    declared_absence_scores=[], presence_interval_samples=0, unlabeled_samples=0))
                info['errors'] += bool(score.get('error'))
                if score.get('score') is not None:
                    info['samples'] += 1
                    info['scores'].append(score['score'])
                    info['calibrated'] += bool(score.get('calibrated'))
                    truth = truth_at(annotations, score.get('person_id'), timestamp)
                    if truth is False:
                        info['declared_absence_scores'].append(score['score'])
                    elif truth is True:
                        info['presence_interval_samples'] += 1  # Not face-level identity ground truth.
                    else:
                        info['unlabeled_samples'] += 1
                if label:
                    measured = labelled['models'].setdefault(key,dict(samples=0,known_samples=0,
                        known_top1_correct=0,known_top1_wrong=0,outside_samples=0,outside_reference_passes=0,
                        outside_calibrated_passes=0))
                    if score.get('score') is not None:
                        measured['samples'] += 1
                        if label['kind'] == 'person':
                            measured['known_samples'] += 1
                            measured['known_top1_correct' if score.get('person_id') == label['person_id'] else 'known_top1_wrong'] += 1
                        else:
                            measured['outside_samples'] += 1
                            measured['outside_reference_passes'] += bool(score.get('passed'))
                            measured['outside_calibrated_passes'] += bool(score.get('passed') and score.get('calibrated') and
                                (key != 'petalface' or score.get('iqa_threshold_verified')))
                if score.get('version'):
                    info['versions'].add(score['version'])
        for event in result.get('events', []):
            events[event['id']] = event
            label = labels_by_track.get(event.get('track_id')) if event.get('track_id') is not None else None
            if truth_at(annotations,event['person_id'],timestamp) is False:
                event_truth[event['id']] = 'wrong'
            elif label and label['kind']=='person':
                event_truth[event['id']] = 'correct' if label['person_id']==event['person_id'] else 'wrong'
            elif label and event['person_id'] in label.get('scope',[]):
                event_truth[event['id']] = 'wrong'
            else:
                event_truth[event['id']] = 'unverified'
    identity_alerts = dict(Counter(event_truth.values()))
    identity_alerts = {key:identity_alerts.get(key,0) for key in ('correct','wrong','unverified')}
    counts = dict(consistent_with_interval=0, false_alerts=0, unverified=0)
    for event in events.values():
        truth = truth_at(annotations, event['person_id'], event['timestamp'])
        counts['unverified' if truth is None else 'consistent_with_interval' if truth else 'false_alerts'] += 1
    intervals = []
    for idx, a in enumerate(annotations):
        if a['kind'] != 'present':
            continue
        covered = observations[idx] > 0 and (job.state == 'complete' or through >= a['end'])
        intervals.append(dict(**a, analyzed_frames=observations[idx], fully_analyzed=covered,
                              confirmation_seen=first[idx] is not None,
                              no_confirmation=covered and first[idx] is None,
                              time_to_confirmation_seconds=first[idx]-a['start'] if first[idx] is not None else None))
    for info in models.values():
        scores = info.pop('scores')
        negatives = info.pop('declared_absence_scores')
        info['declared_absence_samples'] = len(negatives)
        info['declared_absence_cosine_p95'] = float(np.percentile(negatives,95)) if negatives else None
        info['cosine_p50'] = float(np.median(scores)) if scores else None
        info['versions'] = sorted(info['versions'])
        info['identity_accuracy'] = None
    result = dict(run_id=job.id, source_hash=job.source_hash, filename=job.filename, duration=job.duration,
        state=job.state, frames=frames, face_observations=face_count, analyzed_through_seconds=through,
        enrollment=job.enrollment, pipeline_revisions=sorted(revisions), models=models,
        enrollment_changed=enrollment_changed, one_photo_protocol_valid=not enrollment_changed,
        stage_rejections=dict(stages), face_pixels_p50=float(np.median(face_pixels)) if face_pixels else None,
        annotations=annotations, intervals=intervals, alerts=counts, alert_identity_verification=identity_alerts,
        confirmed_face_observations=confirmed,
        face_ground_truth=labelled, face_labels=list(face_labels.values()),
        confirmed_during_declared_absence=unsupported, distinct_track_ids=len(tracks),
        detection_recall=None, identity_switches=None, face_identity_accuracy=None,
        latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else None,
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
        caution='Nhãn thời gian do người dùng xác nhận. Có mặt cùng thời điểm chưa chứng minh AI gán đúng khuôn mặt. '
                'Recall detector, đổi track và độ chính xác từng mặt chưa kiểm chứng khi thiếu nhãn từng mặt. '
                'Cosine chỉ là số đo, không phải độ chính xác.')
    with job.store.db() as db:
        db.execute('''INSERT INTO video_evaluations(run_id,source_hash,result) VALUES (?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET result=excluded.result,updated_at=CURRENT_TIMESTAMP''',
            (job.id, job.source_hash, json.dumps(result)))
    return result
