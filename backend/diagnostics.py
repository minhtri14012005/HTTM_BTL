"""Explicitly consented staged diagnostics; persist metrics and consented replay inputs."""
import json
import math
import re
import time
import uuid
import os
import subprocess
import threading
import numpy as np

LEGACY_SCENARIOS = ('Chính diện 1 m', 'Quay trái/phải 1 m', 'Che miệng/khẩu trang 1 m',
             'Giơ tay/di chuyển 1 m', 'Khoảng cách 3 m', 'Khoảng cách 6 m',
             'Khoảng cách 6–10 m', 'Nhóm đông', 'Người ngoài watchlist')

COMPOSITE_CONDITION = 'Tổng hợp'
CONDITIONS = ('Chính diện', 'Quay trái/phải', 'Giơ tay/di chuyển', 'Che một phần mặt', 'Đeo khẩu trang')
TARGETED_CONDITIONS = ('Quay nghiêng/di chuyển',)
LIVE_CASES = tuple(dict(scenario=f'{condition} {distance} m', distance_m=distance,
                        condition=condition) for distance in (1, 3, 6, 9)
                   for condition in ((COMPOSITE_CONDITION, *CONDITIONS) if distance!=1 else CONDITIONS))
SCENARIOS = tuple(dict.fromkeys([c['scenario'] for c in LIVE_CASES] + list(LEGACY_SCENARIOS)))


def scenario_details(scenario):
    match = re.fullmatch('('+ '|'.join(re.escape(value) for value in (COMPOSITE_CONDITION, *CONDITIONS, *TARGETED_CONDITIONS)) + r') ([0-9]+(?:\.[0-9]+)?) m', scenario)
    if not match:
        return {}
    distance = float(match.group(2))
    if not math.isfinite(distance) or distance <= 0:
        return {}
    return dict(scenario=scenario,distance_m=distance,condition=match.group(1))


def protocol_for(scenario,eyewear=None):
    if eyewear is None:
        eyewear='none' if scenario_details(scenario).get('condition')==COMPOSITE_CONDITION else 'unchanged'
    if eyewear not in ('unchanged','none'):
        raise ValueError('Chọn điều kiện kính hợp lệ.')
    case = scenario_details(scenario)
    combined = case.get('condition') == COMPOSITE_CONDITION
    instructions = (
        ('frontal', 'Chính diện, quay đầu nhẹ', 'NHÌN THẲNG — QUAY ĐẦU NHẸ', 0),
        ('profile', 'Quay trái/phải 45–90°', 'QUAY TRÁI/PHẢI CHẬM 45–90° — GIỮ MỖI BÊN 2–3 GIÂY', 3),
        ('movement', 'Di chuyển, giơ tay, thay đổi tư thế', 'ĐI LẠI NHẸ — GIƠ TAY — THAY ĐỔI TƯ THẾ', 3),
        ('mask', 'Khẩu trang, đứng yên và quay đầu', 'ĐEO KHẨU TRANG — ĐỨNG YÊN VÀ QUAY ĐẦU', 8),
        ('mask_movement', 'Khẩu trang kết hợp di chuyển', 'GIỮ KHẨU TRANG — DI CHUYỂN, QUAY ĐẦU VÀ ĐỔI TƯ THẾ', 3),
        ('partial_occlusion', 'Che một phần mặt, quay đầu và di chuyển', 'THÁO KHẨU TRANG — CHE MỘT PHẦN MẶT BẰNG TAY — QUAY ĐẦU VÀ DI CHUYỂN', 8),
    ) if combined else (('targeted', case.get('condition', scenario), {
        'Chính diện':'NHÌN THẲNG — QUAY ĐẦU NHẸ',
        'Quay trái/phải':'QUAY TRÁI/PHẢI CHẬM TỚI KHOẢNG 90° — GIỮ MỖI BÊN 2–3 GIÂY',
        'Giơ tay/di chuyển':'GIƠ TAY — DI CHUYỂN NHẸ — THAY ĐỔI TƯ THẾ',
        'Che một phần mặt':'CHE MỘT PHẦN MẶT BẰNG TAY — GIỮ MẶT TRONG KHUNG',
        'Đeo khẩu trang':'ĐEO KHẨU TRANG — NHÌN VỀ CAMERA',
    }.get(case.get('condition'),scenario.upper()), 0),)
    stages = [dict(id=key, label=label, instruction=instruction, start=i*30, end=(i+1)*30,
                   transition_seconds=transition) for i,(key,label,instruction,transition) in enumerate(instructions)]
    if combined and case.get('distance_m') == 6:
        actions = (
            ('frontal', 'A', 'Chính diện và quay nhẹ', 'NHÌN THẲNG — QUAY TRÁI/PHẢI NHẸ', 0),
            ('profile', 'A', 'Quay góc lớn', 'QUAY TRÁI/PHẢI CHẬM TỚI GẦN 90°', 2),
            ('movement', 'A', 'Di chuyển và đổi tư thế', 'ĐI LẠI — GIƠ TAY — QUAY ĐẦU', 2),
            ('mask', 'B', 'Đeo khẩu trang và quay nhẹ', 'ĐEO KHẨU TRANG — NHÌN THẲNG VÀ QUAY NHẸ', 8),
            ('mask_profile', 'B', 'Khẩu trang và quay góc lớn', 'GIỮ KHẨU TRANG — QUAY TRÁI/PHẢI', 2),
            ('mask_movement', 'B', 'Khẩu trang và di chuyển', 'GIỮ KHẨU TRANG — ĐI LẠI, GIƠ TAY VÀ QUAY ĐẦU', 2),
            ('unmask_reference', 'C', 'Tháo khẩu trang và xác minh lại', 'THÁO KHẨU TRANG — NHÌN THẲNG', 8),
            ('unmask_movement', 'C', 'Sau tháo khẩu trang, quay và di chuyển', 'KHÔNG KHẨU TRANG — QUAY ĐẦU VÀ ĐI LẠI', 2),
            ('partial_occlusion', 'C', 'Che mặt và di chuyển', 'CHE MỘT PHẦN MẶT BẰNG TAY — QUAY ĐẦU VÀ ĐI LẠI', 3),
        )
        stages = [dict(id=key, group=group, label='NHÓM '+group+' — '+label,
                       instruction='NHÓM '+group+' — '+instruction,
                       start=i*20, end=(i+1)*20, transition_seconds=transition)
                  for i,(key,group,label,instruction,transition) in enumerate(actions)]
    if not combined and case.get('condition')=='Giơ tay/di chuyển':
        stages = [dict(id=key,label=label,instruction=instruction,start=start,end=end,transition_seconds=0)
            for key,label,instruction,start,end in (
                ('movement_reference','Nhìn thẳng, giữ nguyên kính','GIỮ NGUYÊN KÍNH — NHÌN THẲNG — KHÔNG CHE MẶT',0,5),
                ('movement_normal','Di chuyển bình thường','ĐI NGANG TRÁI/PHẢI BÌNH THƯỜNG — GIỮ MẶT TRONG KHUNG',5,15),
                ('movement_faster','Di chuyển nhanh hơn','ĐI NGANG NHANH HƠN VỪA PHẢI — GIỮ MẶT TRONG KHUNG',15,30))]
    if not combined and case.get('condition')=='Quay nghiêng/di chuyển':
        stages=[dict(id=key,label=label,instruction=instruction,start=start,end=end,transition_seconds=transition)
            for key,label,instruction,start,end,transition in (
                ('profile_reference','Nhìn thẳng, giữ kính','NHÌN THẲNG — GIỮ NGUYÊN KÍNH — KHÔNG CHE MẶT',0,5,0),
                ('profile_moderate','Quay nhẹ 45–70°','QUAY TRÁI/PHẢI CHẬM 45–70°',5,10,1),
                ('profile_left','Nghiêng trái khoảng 90°','QUAY TRÁI KHOẢNG 90° — GIỮ MẶT TRONG HÌNH',10,17,1),
                ('profile_right','Nghiêng phải khoảng 90°','QUAY PHẢI KHOẢNG 90° — GIỮ MẶT TRONG HÌNH',17,24,1),
                ('profile_movement','Quay nghiêng kết hợp di chuyển','ĐI NGANG NHẸ VÀ QUAY MẶT — GIỮ MẶT TRONG HÌNH',24,30,1))]
    if eyewear=='none':
        for stage in stages:
            instruction=stage['instruction'].replace('GIỮ NGUYÊN KÍNH — ','').replace(' — GIỮ NGUYÊN KÍNH','')
            stage['instruction']='KHÔNG ĐEO KÍNH — '+instruction
            stage['label']=stage['label'].replace('giữ nguyên kính','không kính').replace('giữ kính','không kính')
    return dict(version=2 if combined and case.get('distance_m') == 6 else 1, kind='composite' if combined else 'targeted', duration_seconds=max(stage['end'] for stage in stages),
                preparation_seconds=5, eyewear=eyewear, stages=stages, transition_policy='scheduled_excluded_from_condition_metrics')


def phase_at(protocol, elapsed):
    if elapsed < 0 or elapsed >= protocol['duration_seconds']:
        return None
    stage = next(stage for stage in protocol['stages'] if stage['start'] <= elapsed < stage['end'])
    transition_end = stage['start']+stage['transition_seconds']
    return dict(stage_id=stage['id'], group=stage.get('group'), label=stage['label'], instruction=stage['instruction'],
                transition=elapsed < transition_end,
                remaining_seconds=max(0., stage['end']-elapsed),
                transition_remaining_seconds=max(0., transition_end-elapsed))


def enrollment_snapshot(store):
    return [dict(person_id=p['id'], name=p['name'], selected=p['selected'],
                 image_ids=[i['id'] for i in p['images']], photo_count=len(p['images']))
            for p in store.people()]


def live_coverage(store):
    with store.db() as db:
        runs = [dict(r) for r in db.execute('SELECT id,scenario,person_id,state,summary,context FROM diagnostics ORDER BY rowid DESC')]
    cases = []
    for case in LIVE_CASES:
        measured = []
        for run in runs:
            if run['scenario'] != case['scenario'] or run['state'] != 'complete':
                continue
            summary = json.loads(run['summary'] or '{}')
            if summary.get('frames', 0) <= 0:
                continue
            context = json.loads(run['context'] or '{}')
            target = next((p for p in context.get('enrollment', []) if p['person_id'] == run['person_id']), None)
            measured.append(dict(id=run['id'], person_id=run['person_id'], frames=summary['frames'],
                                 one_photo=target['photo_count'] == 1 if target else None))
        cases.append(dict(**case, runs=measured, measured=bool(measured), accuracy_verified=False))
    return cases


class Diagnostic:
    def __init__(self, store, scenario, person_id, expected_present, ready, clock=time.monotonic, purpose="validation", eyewear=None):
        if ready is not True:
            raise ValueError('Cần xác nhận sẵn sàng cho từng lượt kiểm thử.')
        case = scenario_details(scenario)
        if (not case and scenario not in LEGACY_SCENARIOS) or purpose not in ('validation','calibration') or (expected_present and not person_id):
            raise ValueError('Chọn tình huống và danh tính mục tiêu dự kiến.')
        if person_id and not store.exists(person_id):
            raise ValueError('Danh tính mục tiêu không tồn tại.')
        self.store, self.clock = store, clock
        self.id, self.state = uuid.uuid4().hex, 'opening'
        self.created = self.clock()
        self.scenario, self.person_id, self.expected_present = scenario, person_id, expected_present
        self.protocol = protocol_for(scenario,eyewear)
        self.duration = self.protocol['duration_seconds']
        calibration = store.settings()
        self.context = dict(**case, enrollment=enrollment_snapshot(store), single_person=bool(case),
                            natural_movement_allowed=bool(case),
                            calibration_revision=calibration.get('calibration_revision',0),
                            calibration_role='fit' if purpose=='calibration' or not calibration['calibrated'] else 'validation',
                            purpose=purpose, eyewear=self.protocol['eyewear'], protocol=self.protocol,
                            hardware_profile=dict(sample_group='A', start_seconds=0, end_seconds=60,
                                deep_profile='offline_replay_group_A_only') if self.protocol['version']==2 else None,
                            identity_tracking_profile=calibration.get('identity_tracking_profile'))
        profile_fields = ('model','version','band','threshold','margin','quality_basis',
                          'verification_enabled','numeric_guard','label_generation','identity_gates')
        self.context['calibration_profile_snapshot'] = [
            {key: profile[key] for key in profile_fields if key in profile}
            for profile in calibration['calibration_profiles']]
        self.prepared = self.started = self.last_frame = None
        self.session_id, self.summary = None, None
        self.examples = {}  # Bounded preview cache; native AI inputs are also persisted below.
        self.previous_named_example = None
        self.hardware_stop = threading.Event()
        with store.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS diagnostic_hardware (
                run_id TEXT NOT NULL, elapsed REAL NOT NULL, metrics TEXT NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS diagnostic_inputs (
                run_id TEXT NOT NULL, frame_id INTEGER NOT NULL, metadata TEXT NOT NULL,
                jpeg BLOB NOT NULL, PRIMARY KEY(run_id,frame_id))''')
            db.execute('INSERT INTO diagnostics(id,scenario,person_id,expected_present,state,context) VALUES (?,?,?,?,?,?)',
                (self.id, scenario, person_id, int(expected_present), self.state, json.dumps(self.context)))

    def view(self):
        now = self.clock()
        remaining = max(0., self.duration-(now-self.started)) if self.started is not None else None
        prepare = max(0., 5-(now-self.prepared)) if self.prepared is not None else None
        return dict(id=self.id, state=self.state, remaining=remaining, prepare_remaining=prepare,
                    scenario=self.scenario, summary=self.summary, context=self.context, protocol=self.protocol,
                    phase=phase_at(self.protocol, now-self.started) if self.state=='running' else None)

    def retain_example(self, result, captured, raw):
        if self.state != 'running' or self.started is None or not self.started <= captured < self.started+self.duration:
            return
        if not isinstance(raw,bytes) or not raw or len(raw)>2*1024**2:
            return
        elapsed = captured-self.started
        faces = result.get('overlays',[])
        severity = max((3 if any(v.get('error') for v in face.get('model_scores',{}).values()) else
                        2 if any(v.get('alignment_band')=='high_residual' for v in face.get('model_scores',{}).values()) else
                        1 if face.get('status')!='confirmed' else 0 for face in faces),default=3)
        captured_phase = phase_at(self.protocol,elapsed)
        current = dict(frame_id=result['frame_id'],elapsed=elapsed,severity=severity,stage_id=captured_phase['stage_id'],kind='ai',
                       width=result['width'],height=result['height'],raw=raw)
        # Commit each original AI input before acknowledging its result. This also
        # survives cancellation and process exit, without retaining an unbounded RAM cache.
        with self.store.db() as db:
            db.execute('INSERT OR IGNORE INTO diagnostic_inputs VALUES (?,?,?,?)',
                       (self.id,result['frame_id'],json.dumps({k:v for k,v in current.items() if k!='raw'}),raw))
        named = any(face.get('status') in ('confirmed','tracked') and face.get('person_id') for face in faces)
        # Unnamed AI inputs are needed to distinguish recognition failures from
        # missing visual overlays. Keep them ahead of secondary preview samples.
        if not named:
            current['severity'] = 5
        previous_named = self.previous_named_example
        self.previous_named_example = current if named else None
        # Keep actual AI inputs, not the full 30-fps video. Bounded native JPEGs
        # allow detector/alignment/association replay, rather than scores only.
        self.examples[result['frame_id']] = current
        if previous_named is not None and not named:
            for sample in (previous_named,current):
                self.examples[sample['frame_id']] = dict(sample,severity=max(4,sample['severity']))
        self._trim_examples()

    def retain_visual_example(self,frame,captured,raw,anchor_frame_id):
        if self.state!='running' or self.started is None or not self.started<=captured<self.started+self.duration:
            return
        if not isinstance(raw,bytes) or not raw or len(raw)>2*1024**2:
            return
        elapsed=captured-self.started;phase=phase_at(self.protocol,elapsed)
        self.examples[frame['frame_id']]=dict(frame_id=frame['frame_id'],elapsed=elapsed,stage_id=phase['stage_id'],
            severity=3,kind='preview',anchor_frame_id=anchor_frame_id,width=frame['width'],height=frame['height'],
            source_timestamp=frame['media_timestamp'],overlays=frame['overlays'],raw=raw)
        self._trim_examples()

    def _trim_examples(self):
        composite = self.protocol['kind']=='composite'
        max_examples = 1024 if composite else 256
        max_bytes = (96 if composite else 16)*1024**2
        extra = self.previous_named_example
        while self.examples:
            extra_bytes = len(extra['raw']) if extra and all(item['frame_id']!=extra['frame_id'] for item in self.examples.values()) else 0
            count = len(self.examples)+bool(extra_bytes)
            size = sum(len(item['raw']) for item in self.examples.values())+extra_bytes
            if count <= max_examples and size <= max_bytes:
                break
            # Under pressure, evict from the most represented action first;
            # keep its first error pair before routine or redundant samples.
            counts = {stage['id']:sum(item.get('stage_id')==stage['id'] for item in self.examples.values()) for stage in self.protocol['stages']}
            largest = max(counts,key=counts.get)
            candidates=[key for key,item in self.examples.items() if item.get('stage_id')==largest]
            protected=sorted((key for key in candidates if self.examples[key]['severity']==4),key=lambda key:self.examples[key]['elapsed'])[:2]
            candidates=[key for key in candidates if key not in protected] or candidates
            discard=min(candidates,key=lambda key:(self.examples[key]['severity'],self.examples[key]['elapsed']))
            self.examples.pop(discard)

    def capture_time(self, header, received):
        if self.started is not None and header.get('diagnostic_capture_pending') is True:
            return self.started-1.  # A preparation image encoded before BEGIN is never measured.
        elapsed = header.get('diagnostic_elapsed')
        if elapsed is None or self.started is None:
            return received
        if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed):
            raise ValueError('Timestamp kiểm thử không hợp lệ.')
        captured = self.started+elapsed
        if captured > received:
            raise ValueError('Timestamp capture vượt thời điểm nhận frame.')
        return captured

    def observe(self, result, captured):
        now = self.clock()
        self.last_frame = now
        if self.state == 'opening':
            self.state, self.prepared = 'preparing', now
        if self.state != 'running':
            return
        if now >= self.started+self.duration:
            self.finish('complete')
            return
        if captured < self.started or captured >= self.started+self.duration:
            return  # Discard all inference started during preparation.
        metrics = {key: result.get(key) for key in ('width','height','processing_ms','ai_fps','ai_throughput_fps','detector_ms','gpu','backlog','overlays','sampling','pipeline_revision','recognition_ms','visual_tracking_ms','measurement_evidence_reset','shadow_models','live_timing_revision')}
        metrics['frame_id'] = result['frame_id']
        metrics['diagnostic_phase'] = phase_at(self.protocol, captured-self.started)
        metrics['source_timestamp'] = result.get('media_timestamp')
        metrics['captured_at'] = result.get('captured_at')
        metrics['diagnostic_timestamp_basis'] = result.get('diagnostic_timestamp_basis','server_receive_legacy')
        metrics['events'] = [{k:e.get(k) for k in ('id','person_id','timestamp','track_id','confirmation_seconds')} for e in result['events']]
        with self.store.db() as db:
            db.execute('INSERT OR IGNORE INTO diagnostic_frames VALUES (?,?,?,?)',
                (self.id, result['frame_id'], captured-self.started, json.dumps(metrics)))

    def sample_group_a_hardware(self):
        # Sampling runs outside the inference thread and ends at 60 seconds.
        previous = os.times()
        previous_time = time.monotonic()
        while not self.hardware_stop.is_set():
            elapsed = self.clock()-self.started
            if elapsed >= 60 or self.state != 'running':
                return
            now = time.monotonic()
            cpu = os.times()
            metrics = dict(pid=os.getpid(), group='A',
                           process_cpu_percent=100*((cpu.user+cpu.system)-(previous.user+previous.system))/max(.001,now-previous_time),
                           cpu_percent_basis='one_logical_core_100_percent',
                           gpu_scope='device_total_includes_other_processes')
            previous,previous_time = cpu,now
            try:
                result = subprocess.run(['nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm',
                    '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=1.5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                metrics['nvidia_smi'] = result.stdout.strip()
                if result.returncode:
                    metrics['gpu_error'] = result.stderr.strip()
            except (OSError,subprocess.TimeoutExpired) as exc:
                metrics['gpu_error'] = str(exc)
            with self.store.db() as db:
                db.execute('INSERT INTO diagnostic_hardware VALUES (?,?,?)',
                           (self.id,elapsed,json.dumps(metrics)))
            if self.hardware_stop.wait(2):
                return

    def begin(self):
        now = self.clock()
        if self.state != 'preparing' or now-self.prepared < 5:
            raise ValueError('Chưa hoàn tất 5 giây chuẩn bị với camera/backend sẵn sàng.')
        if self.last_frame is None or now-self.last_frame > 3:
            raise ValueError('Không có frame AI mới. Chờ backend xử lý hoặc kiểm tra kết nối camera.')
        self.state, self.started = 'running', now
        if self.protocol.get('version') == 2:
            threading.Thread(target=self.sample_group_a_hardware,daemon=True,
                             name='diagnostic-group-a-hardware').start()
        with self.store.db() as db:
            db.execute('UPDATE diagnostics SET state=? WHERE id=?', (self.state,self.id))
        return self.view()

    def finish(self, state='cancelled'):
        self.hardware_stop.set()
        self.previous_named_example = None
        self.runtime_models = {}  # Active pipeline owns its own references until inference finishes.
        if self.state in ('complete', 'cancelled', 'error'):
            return self.view()
        duration = min(float(self.duration), max(0., self.clock()-self.started)) if self.started is not None else 0.
        with self.store.db() as db:
            rows = [(r['elapsed'],json.loads(r['metrics'])) for r in db.execute(
                'SELECT * FROM diagnostic_frames WHERE run_id=? ORDER BY elapsed', (self.id,))]
        events = {}
        for elapsed,frame in rows:
            for event in frame['events']:
                events.setdefault(event['id'],(elapsed,event))
        false = [event for _,event in events.values() if not self.expected_present or event['person_id'] != self.person_id]
        latencies = [frame['processing_ms'] for _,frame in rows]
        confirm_times = [event['confirmation_seconds'] for _,event in events.values() if event.get('confirmation_seconds') is not None]
        confirmed_frames = sum(any(o['status']=='confirmed' and o['person_id']==self.person_id for o in frame['overlays']) for _,frame in rows)
        self.summary = dict(frames=len(rows), duration_seconds=duration, confirmed_frames=confirmed_frames,
            tracked_frames=sum(any(o['status']=='tracked' and o.get('person_id')==self.person_id for o in frame['overlays']) for _,frame in rows),
            preparation_evidence_isolated=any(f.get('measurement_evidence_reset') for _,f in rows),
            ai_throughput_fps=len(rows)/duration if duration else None,
            target_frame_recall_proxy=confirmed_frames/len(rows) if rows and self.expected_present else None,
            false_alerts=len(false), false_alerts_per_hour=len(false)*3600/duration if duration else None,
            latency_p50_ms=float(np.percentile(latencies,50)) if latencies else None,
            latency_p95_ms=float(np.percentile(latencies,95)) if latencies else None,
            time_to_confirm_p50_seconds=float(np.percentile(confirm_times,50)) if confirm_times else None,
            time_to_confirm_p95_seconds=float(np.percentile(confirm_times,95)) if confirm_times else None,
            first_alert_seconds=min((elapsed for elapsed,e in events.values() if e['person_id']==self.person_id),default=None),
            ground_truth='Người dùng khai báo cho toàn lượt; chưa có nhãn từng frame/face.',
            detection_recall=None, identity_switches=None,
            verification='Chưa kiểm chứng recall detection và identity switches nếu thiếu nhãn từng khuôn mặt.')
        self.state = state
        with self.store.db() as db:
            db.execute('UPDATE diagnostics SET state=?,summary=? WHERE id=?', (state,json.dumps(self.summary),self.id))
        return self.view()
