import asyncio
import base64
import hashlib
from datetime import datetime, timezone
import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import closing

import cv2
from .ai import Pipeline, jpeg
from .diagnostics import enrollment_snapshot
from .video_evaluation import save_measurement


def camera_timestamp(value, previous=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Timestamp camera không hợp lệ.')
    if previous is not None and value <= previous:
        raise ValueError('Timestamp camera không tăng theo frame.')
    return float(value)


def camera_matches(kind, label):
    label = label.casefold().strip()
    if kind == 'camo':
        return 'camo' in label
    if kind != 'laptop' or not label:
        return False
    return not any(x in label for x in ('iriun', 'camo', 'obs', 'virtual', 'screen', 'droidcam', 'iphone', 'continuity'))


def event_with_evidence(event, image, source, session_id):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = event.get('face_bbox', event['bbox'])
    margin = max(x2-x1, y2-y1)*.35
    x1, y1 = max(0, int(x1-margin)), max(0, int(y1-margin))
    x2, y2 = min(w, int(x2+margin)), min(h, int(y2+margin))
    crop = image[y1:y2, x1:x2]
    event.update(id=f"{session_id}:{event['person_id']}:{event['visit']}", source=source,
                 session_id=session_id, occurred_at=datetime.now(timezone.utc).isoformat(),
                 evidence='data:image/jpeg;base64,'+base64.b64encode(jpeg(crop)).decode())
    return event


class Broker:
    def __init__(self):
        self.clients = set()
        self.loop = None

    async def attach(self, ws):
        await ws.accept()
        self.clients.add(ws)

    async def publish(self, message):
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=2)
            except Exception:
                self.clients.discard(ws)

    def send(self, message):
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.publish(message), self.loop)


class VideoJob:
    def __init__(self, path, root, engine, store, broker, source_hash=None, filename=None):
        self.id = uuid.uuid4().hex
        self.path = path
        self.db_path = root/f'{self.id}.sqlite3'
        self.engine, self.store, self.broker = engine, store, broker
        if not source_hash:
            with path.open('rb') as source:
                source_hash = hashlib.file_digest(source, 'sha256').hexdigest()
        self.source_hash = source_hash
        self.filename = filename or path.name
        self.enrollment = enrollment_snapshot(store)
        self.enrollment_changed = False
        self.capture = cv2.VideoCapture(str(path))
        self.capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, first = self.capture.read()
        if not ok:
            self.capture.release()
            raise ValueError('Không đọc được video. Hãy chọn MP4 H.264 hoặc WebM hợp lệ.')
        self.first = first
        self.width, self.height = first.shape[1], first.shape[0]
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 < self.fps <= 240 or self.total <= 0 or self.width*self.height > 16_000_000:
            self.capture.release()
            raise ValueError('Metadata video không hợp lệ hoặc độ phân giải quá lớn.')
        self.duration = self.total/self.fps
        self.state, self.error = 'ready', None
        self.processed, self.analyzed_until, self.ai_fps = 0, -1., 0.
        self.processing_seconds = 0.
        self.events = []
        self.generation = 0
        self.gate = threading.Event()
        self.cancelled = threading.Event()
        self.thread = None
        self.pipeline = None
        self.control_lock = threading.RLock()
        self.playback_timestamp = 0.
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE frames (id INTEGER PRIMARY KEY, timestamp REAL UNIQUE, result TEXT NOT NULL)')

    def status(self):
        return dict(id=self.id, session_id=self.id, source='Video Upload', state=self.state, error=self.error,
                    fps=self.fps, total=self.total, duration=self.duration, width=self.width, height=self.height,
                    processed=self.processed, progress=min(1., self.processed/self.total),
                    analyzed_until=self.analyzed_until, ai_fps=self.ai_fps, generation=self.generation,
                    media_url=f'/api/videos/{self.id}/media', events=self.events, source_hash=self.source_hash,
                    filename=self.filename)

    def control(self, action, timestamp=0.):
        if self.cancelled.is_set():
            raise ValueError('Tác vụ đã đóng.')
        if action not in ('pause', 'play', 'seek'):
            raise ValueError('Lệnh không hợp lệ.')
        if not 0 <= timestamp <= self.duration:
            raise ValueError('Vị trí phát nằm ngoài video.')
        self.generation += 1
        self.playback_timestamp = timestamp
        if action == 'pause':
            self.gate.clear()
            if self.state not in ('complete', 'error'):
                self.state = 'paused'
        elif action in ('play', 'seek'):
            if self.state not in ('complete', 'error'):
                self.state = 'analyzing'
                self.gate.set()
                if self.thread is None:
                    self.thread = threading.Thread(target=self.run, daemon=True, name=f'video-{self.id}')
                    self.thread.start()
        else:
            raise ValueError('Lệnh không hợp lệ.')
        return self.status()

    def run(self):
        try:
            self.pipeline = Pipeline(self.engine, self.store, self.fps)
            frame, frame_id, last_timestamp = self.first, 0, -1.
            self.first = None
            while not self.cancelled.is_set():
                if not self.gate.wait(.1):
                    continue
                work_started = time.perf_counter()
                if frame is None:
                    ok, frame = self.capture.read()
                    if not ok:
                        self.state = 'complete'
                        self.analyzed_until = self.duration
                        break
                timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                if timestamp <= last_timestamp:
                    timestamp = max(last_timestamp+1/self.fps, frame_id/self.fps)
                generation = self.generation
                before = enrollment_snapshot(self.store)
                result = self.pipeline.run(frame, frame_id, timestamp)
                self.enrollment_changed |= before != self.enrollment or enrollment_snapshot(self.store) != self.enrollment
                result['enrollment_changed'] = self.enrollment_changed
                result.update(source='Video Upload', session_id=self.id)
                for event in result['events']:
                    event_with_evidence(event, frame, 'Video Upload', self.id)
                    self.events.append(event)
                if self.cancelled.is_set():
                    break
                # Write before advancing watermark. Readers never consume an incomplete frame.
                with closing(sqlite3.connect(self.db_path, timeout=20)) as db:
                    db.execute('INSERT INTO frames VALUES (?,?,?)', (frame_id, timestamp, json.dumps(result)))
                    db.commit()
                save_measurement(self.store, self.id, result)
                self.processed = frame_id+1
                self.analyzed_until = timestamp
                self.processing_seconds += time.perf_counter()-work_started
                self.ai_fps = self.processed/self.processing_seconds
                if generation == self.generation and self.gate.is_set():
                    self.broker.send(dict(type='video_frame', session_id=self.id, frame_id=frame_id,
                        generation=generation, timestamp=timestamp, events=result['events'], status=self.status_without_events()))
                frame_id += 1
                last_timestamp = timestamp
                frame = None
            self.broker.send(dict(type='video_status', session_id=self.id, status=self.status_without_events()))
        except Exception as exc:
            self.error, self.state = str(exc), 'error'
            self.broker.send(dict(type='video_status', session_id=self.id, status=self.status_without_events()))
        finally:
            self.capture.release()

    def status_without_events(self):
        result = self.status()
        result.pop('events')
        return result

    def frames(self, start, end):
        with closing(sqlite3.connect(self.db_path, timeout=20)) as db:
            rows = db.execute('''SELECT result FROM frames WHERE
                 (timestamp>=? AND timestamp<=?) OR id=(SELECT id FROM frames WHERE timestamp<? ORDER BY timestamp DESC LIMIT 1)
                 ORDER BY timestamp''', (max(0, start), max(start, min(end, start+10)), max(0, start))).fetchall()
        return [dict(json.loads(r[0]), generation=self.generation) for r in rows]

    def close(self):
        self.cancelled.set(); self.gate.set()
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError('AI chưa dừng trong thời gian cho phép; chưa xóa dữ liệu phiên.')
        else:
            self.capture.release()

    def dispose(self):
        self.close()
        # A browser may still hold a range response open on Windows. Shutdown will
        # remove any such files once the response releases its handle.
        for path in (self.path, self.db_path):
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass

