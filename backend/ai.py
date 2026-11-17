"""Hybrid face-only pipeline; all inference stays on the local machine."""
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .config import MODELS
from .models import ONNXRecognizer, RetinaR50, YoloFace, PetalFace
from .temporal import Temporal, stable_association
from .identity_tracking import trusted_identity_link
from .fusion import identity_evidence, verification_evidence, quality_band, model_gates


def jpeg(image, quality=90):
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('Không mã hóa được ảnh.')
    return buf.tobytes()


def decode(raw):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width*image.height > 40_000_000:
                raise ValueError('Ảnh quá lớn (tối đa 40 megapixel).')
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert('RGB')), cv2.COLOR_RGB2BGR)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Không đọc được ảnh. Hãy dùng JPEG, PNG hoặc WebP hợp lệ.') from exc


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    w, h = max(0, min(a[2], b[2])-x), max(0, min(a[3], b[3])-y)
    area = w*h
    return area/max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def landmarks_order_valid(points):
    points=np.asarray(points)
    return bool(points.shape==(5,2) and np.isfinite(points).all() and
                points[1,0]>points[0,0] and points[4,0]>points[3,0])


def predicted_face_box(track, timestamp):
    """Predict association geometry only. The displayed box is always a detection."""
    elapsed = min(.5, max(0., timestamp-track['time']))
    velocity = np.asarray(track.get('velocity', (0., 0.)), dtype=np.float32)
    shift = velocity*elapsed
    predicted = track['box'].copy()
    predicted[:4] += np.tile(shift, 2)
    return predicted


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.ready, self.error, self.device = False, None, 'cpu'
        self.models, self.status, self.versions = {}, {}, {}
        self.detector = self.fallback = None
        self.root = MODELS/'hybrid'
        self.preprocess_pool = None
        self.preprocess_workers = min(4,max(1,(os.cpu_count() or 2)//2))

    def load(self):
        with self.lock:
            if self.ready:
                return
            try:
                import torch
                self.torch = torch
                torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4)//2)))
                requested = os.environ.get('FACEWATCH_DEVICE', 'auto')
                if requested not in ('auto', 'cpu', 'cuda'):
                    raise ValueError('FACEWATCH_DEVICE phải là auto, cpu hoặc cuda.')
                self.device = 'cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu'
                if requested == 'cuda' and self.device != 'cuda':
                    raise RuntimeError('CUDA không khả dụng.')
                from scripts.setup_models import sha
                manifest_path = self.root/'manifest.json'
                if not manifest_path.exists():
                    raise RuntimeError('Chưa cài Hybrid model. Chạy .\\setup.ps1.')
                manifest = json.loads(manifest_path.read_text(encoding='utf8'))
                bad = set()
                for relative, expected in manifest['files'].items():
                    path = (self.root/relative).resolve()
                    if not path.is_relative_to(self.root.resolve()) or not path.is_file() or sha(path) != expected:
                        bad.add(relative)
                paths = {'lvface': 'lvface.onnx', 'antelopev2': 'antelopev2.onnx', 'petalface': 'petalface.pt',
                         'retinaface': 'Resnet50_Final.pth', 'yolov5m_face': 'yolov5m-face.onnx'}
                for key, filename in paths.items():
                    try:
                        if filename not in manifest['files'] or filename in bad:
                            raise RuntimeError(f'Thiếu hoặc sai checksum: {filename}')
                        if key == 'retinaface':
                            if any(p.startswith('retina/') for p in bad):
                                raise RuntimeError('RetinaFace source checksum không đúng.')
                            self.detector = RetinaR50(self.root, self.device)
                            provider = self.device
                        elif key == 'yolov5m_face':
                            self.fallback = YoloFace(self.root/filename, self.device)
                            provider = self.fallback.provider
                        elif key == 'petalface':
                            if 'cnniqa.pth' in bad or any(p.startswith('petal/') for p in bad):
                                raise RuntimeError('PETALface source/IQA checksum không đúng.')
                            self.status[key] = dict(state='lazy', device=self.device,
                                note='Quality-adaptive LoRA + CNNIQA; threshold IQA mặc định tác giả chưa kiểm chứng trên checkpoint.')
                            self.versions[key] = f'{key}:{manifest["files"][filename][:20]}:rgb120-iqa05-flip-v1'
                            continue
                        else:
                            self.models[key] = ONNXRecognizer(self.root/filename, self.device, key)
                            provider = self.models[key].provider
                        self.versions[key] = f'{key}:{manifest["files"][filename][:20]}:align-rgb-v1'
                        self.status[key] = dict(state='ready', device=provider, requested_device=self.device,
                            fallback_reason='CUDA provider unavailable; ONNX Runtime selected CPU'
                            if self.device == 'cuda' and provider == 'CPUExecutionProvider' else None)
                    except Exception as exc:
                        self.status[key] = dict(state='unavailable', error=str(exc))
                self.ready = 'lvface' in self.models and bool(self.detector or self.fallback)
                self.detector_provider = self.status.get('retinaface', {}).get('device') or self.status.get('yolov5m_face', {}).get('device')
                if not self.ready:
                    raise RuntimeError('Hybrid AI chưa sẵn sàng: '+json.dumps(self.status, ensure_ascii=False))
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                raise

    def model(self, key):
        if key == 'petalface' and key not in self.models:
            if self.status.get(key, {}).get('state') == 'unavailable':
                raise RuntimeError(self.status[key]['error'])
            device = self.device
            fallback_reason = None
            if device == 'cuda' and self.torch.cuda.mem_get_info()[0] < 1700*1024**2:
                device = 'cpu'
                fallback_reason = 'Insufficient free CUDA memory for PETALface (requires 1700 MiB reserve)'
            try:
                self.models[key] = PetalFace(self.root, device)
                self.status[key].update(state='ready', device=device, requested_device=self.device,
                                        fallback_reason=fallback_reason)
            except Exception as exc:
                self.status[key].update(state='unavailable', error=str(exc))
                raise
        return self.models[key]

    def embed(self, image, points, key='lvface', strict_alignment=True):
        started = time.perf_counter()
        embedding, info = self.model(key).embed(image, points, strict_alignment=strict_alignment)
        info['latency_ms'] = (time.perf_counter()-started)*1000
        info['version'] = self.versions[key]
        return embedding, info

    def embed_many(self, image, landmarks, key='lvface', strict_alignment=True):
        recognizer = self.model(key)
        if self.preprocess_pool is None and len(landmarks) > 1:
            self.preprocess_pool = ThreadPoolExecutor(max_workers=self.preprocess_workers,
                                                      thread_name_prefix='face-preprocess')
        results = recognizer.embed_many(image,landmarks,strict_alignment,self.preprocess_pool)
        self.status[key]['batch_limit'] = recognizer.batch_size
        self.status[key]['execution'] = 'fp32_batch' if recognizer.batch_size > 1 else 'fp32_single'
        for result in results:
            if not isinstance(result, Exception):
                result[1]['version'] = self.versions[key]
        return results

    def detect(self, image):
        h, w = image.shape[:2]
        found = []
        # Native resolution tiles with overlap; full image is never squashed to 640.
        tile, stride = 1280, 1024
        xs = sorted(set(range(0, max(1, w-tile+1), stride)) | {max(0, w-tile)})
        ys = sorted(set(range(0, max(1, h-tile+1), stride)) | {max(0, h-tile)})
        primary = self.detector or self.fallback
        # Global pass keeps large faces intact before detail tiles are merged.
        if max(h, w) > tile:
            if self.detector and h*w <= 3_500_000:
                found.extend(self.detector.detect(image))
            else:
                ratio = min(1., 1600/max(h, w))
                global_image = cv2.resize(image, (round(w*ratio), round(h*ratio)))
                for box, points, name in primary.detect(global_image):
                    box[:4] /= ratio
                    points /= ratio
                    found.append((box, points, name))
        for y in ys:
            for x in xs:
                crop = image[y:y+tile, x:x+tile]
                detections = primary.detect(crop)
                # Conditional rescue for no detections and periodic difficult small-face tiles.
                difficult = not detections or any(min(b[2]-b[0], b[3]-b[1]) < 90 or not landmarks_order_valid(points)
                                                   for b,points,_ in detections)
                if self.detector and self.fallback and difficult:
                    detections += self.fallback.detect(crop)
                for box, points, detector in detections:
                    box, points = box.copy(), points.copy()
                    ch, cw = crop.shape[:2]
                    # A tile-edge fragment is not another face. The global pass covers large faces.
                    if ((x > 0 and box[0] < 3) or (y > 0 and box[1] < 3) or
                        (x+cw < w and box[2] > cw-3) or (y+ch < h and box[3] > ch-3)):
                        continue
                    if ((points[:, 0] < box[0]-3).any() or (points[:, 0] > box[2]+3).any() or
                        (points[:, 1] < box[1]-3).any() or (points[:, 1] > box[3]+3).any()):
                        continue
                    box[:4] += [x, y, x, y]
                    points += [x, y]
                    box[:4] = np.clip(box[:4], [0, 0, 0, 0], [w, h, w, h])
                    if box[2] > box[0] and box[3] > box[1]:
                        found.append((box, points, detector))
        found.sort(key=lambda f: float(f[0][4]), reverse=True)
        kept = []
        for face in found:
            duplicate=next((index for index,other in enumerate(kept) if iou(face[0],other[0])>.4),None)
            if duplicate is None:
                kept.append(face)
            elif (iou(face[0],kept[duplicate][0])>=.6 and not landmarks_order_valid(kept[duplicate][1]) and
                  landmarks_order_valid(face[1])):
                # Rescue the same detected face using valid geometry. No identity scores participate.
                kept[duplicate]=face
        return kept

    def quality(self, image, box, settings, inference=False):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = np.clip(box[:4], [0, 0, 0, 0], [w, h, w, h]).astype(int)
        pixels = min(x2-x1, y2-y1)
        minimum = settings.get('min_inference_face_pixels', 14) if inference else settings['min_face_pixels']
        if pixels < minimum:
            raise ValueError(f'Mặt quá nhỏ: {pixels}px trước alignment.')
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        native_sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Laplacian variance at native size penalizes a large/upscaled face even
        # when the recognizer receives the same useful detail after alignment.
        # Normalize only the quality measurement; inference keeps the source.
        quality_gray = gray
        if pixels > 112:
            ratio = 112/pixels
            quality_gray = cv2.resize(gray, (round(gray.shape[1]*ratio), round(gray.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(quality_gray, cv2.CV_64F).var())
        if sharpness < settings['min_sharpness']:
            raise ValueError(f'Mặt mờ: sharpness={sharpness:.1f}.')
        return dict(face_pixels=int(pixels), original_face_size=[int(x2-x1), int(y2-y1)],
                    sharpness=sharpness, native_sharpness=native_sharpness,
                    quality_basis='min112-downsample-only-v1',
                    quality_band='tiny' if pixels < 28 else 'small' if pixels < 90 else 'normal',
                    brightness=float(gray.mean()), detector_score=float(box[4]))

    def register(self, raw, settings):
        with self.lock:
            self.load()
            image = decode(raw)
            faces = self.detect(image)
            if len(faces) != 1:
                raise ValueError(f'Ảnh phải có đúng 1 khuôn mặt hợp lệ; tìm thấy {len(faces)}.')
            box, points, detector = faces[0]
            quality = self.quality(image, box, settings)
            quality.update(detector=detector, source_size=[image.shape[1], image.shape[0]], models={})
            embeddings = {}
            for key in self.versions:
                if key not in ('lvface', 'antelopev2', 'petalface'):
                    continue
                try:
                    embedding, info = self.embed(image, points, key)
                    embeddings[self.versions[key]] = embedding
                    quality['models'][key] = info
                except Exception as exc:
                    quality['models'][key] = {'error': str(exc)}
                    if key == 'lvface':
                        raise
            # Preserve exact uploaded bytes, including EXIF. Decode has already validated it.
            return raw, embeddings, quality

    def gpu(self):
        if self.device != 'cuda':
            return dict(device='cpu',cpu_preprocessing_workers=self.preprocess_workers if self.preprocess_pool else 1)
        free, total = self.torch.cuda.mem_get_info()
        return dict(device=self.torch.cuda.get_device_name(), total_mb=round(total/1024**2),
                    used_mb=round((total-free)/1024**2),cpu_preprocessing_workers=self.preprocess_workers if self.preprocess_pool else 1, torch_allocated_mb=round(self.torch.cuda.memory_allocated()/1024**2))


class Pipeline:
    def __init__(self, engine, store, fps=30, *, model_overrides=None, shadow_models=None, disabled_models=()):
        self.engine, self.store = engine, store
        self.model_overrides = dict(model_overrides or {})
        self.shadow_models = set(self.model_overrides if shadow_models is None else shadow_models)
        self.shadow_models.update(key for key,adapter in self.model_overrides.items() if getattr(adapter,'shadow_only',False))
        self.disabled_models = set(disabled_models)
        self.temporal = Temporal()
        self.face_tracks, self.next_face = {}, 1
        self.visible_tracks = set()
        self.previous_overlays = []
        self.source_shape = None
        self.evidence_generation = None

    def run(self, image, frame_id, timestamp, *, reset_temporal=False, diagnostic_details=False):
        started = time.perf_counter()
        engine = self.engine
        with engine.lock:
            engine.load()
            if reset_temporal:
                self.temporal = Temporal()
            if self.source_shape != image.shape[:2]:
                self.face_tracks.clear()
                self.visible_tracks.clear()
                self.previous_overlays = []
                self.temporal.reset_identity_evidence()
                self.source_shape = image.shape[:2]
            settings = self.store.settings(engine.versions)
            selected = self.store.selected_ids()
            evidence_generation = (settings.get('calibration_generation',0),settings.get('calibration_revision',0),
                                   settings['confirm_frames'],tuple(sorted(selected)),json.dumps(settings.get('identity_tracking_profile'),sort_keys=True))
            if evidence_generation != self.evidence_generation:
                self.temporal.reset_identity_evidence()
                self.evidence_generation = evidence_generation
            versions = {key:version for key,version in engine.versions.items() if key in ('lvface','antelopev2','petalface')}
            versions.update({key:adapter.version for key,adapter in self.model_overrides.items()})
            versions = {key:version for key,version in versions.items() if key not in self.disabled_models}
            keys = tuple(versions)
            anchor_key = 'lvface' if 'lvface' in versions else next(iter(versions),None)
            galleries = {key:self.store.gallery_model(version) for key,version in versions.items()}
            person_names = {person:name for gallery in galleries.values() for person,name,_ in gallery}
            detector_start = time.perf_counter()
            faces = engine.detect(image)
            detector_ms = (time.perf_counter()-detector_start)*1000
            processed, valid = [], []
            for box, points, detector in faces:
                item = dict(box=box, points=points,detector=detector,
                            quality={'face_pixels':int(min(box[2]-box[0],box[3]-box[1]))},
                            scores={},vectors={},error=None)
                try:
                    item['quality'] = engine.quality(image,box,settings,inference=True)
                    valid.append(item)
                except ValueError as exc:
                    item['error'] = str(exc)
                processed.append(item)
            recognition_ms = {}
            for key in keys:
                adapter = self.model_overrides.get(key)
                if adapter is None and engine.status[key]['state'] == 'unavailable':
                    continue
                candidates = []
                for item in valid:
                    native_key = getattr(adapter,'fallback_for',None) if adapter is not None else None
                    native_score = item['scores'].get(native_key,{})
                    if native_score.get('calibrated') and native_score.get('passed') and not native_score.get('error'):
                        if diagnostic_details:
                            item.setdefault('routing',{})[key]=dict(inference_performed=False,
                                reason='native_model_has_calibrated_evidence',native_model=native_key)
                        continue
                    allowed_bands=getattr(adapter,'allowed_bands',None) if adapter is not None else None
                    if allowed_bands is not None:
                        alignment=getattr(adapter,'routing_alignment',None) or item['scores'].get(anchor_key,{}).get('alignment_band')
                        height=item['quality'].get('original_face_size',[0,item['box'][3]-item['box'][1]])[1]
                        band=quality_band(item['quality']['face_pixels'],alignment,height) if alignment is not None else None
                        if band not in allowed_bands:
                            if diagnostic_details:
                                item.setdefault('routing',{})[key]=dict(inference_performed=False,
                                    reason='outside_validated_quality_band',band=band,allowed_bands=sorted(allowed_bands))
                            continue  # Only the expert abstains; capture, detection and other models still run.
                    if key == 'petalface':
                        core = [item['scores'].get(model,{}) for model in ('lvface','antelopev2')]
                        clear_agreement = all(score.get('passed') and
                            score.get('alignment_band','regular') == 'regular' for score in core) and (
                            core[0].get('person_id') == core[1].get('person_id'))
                        if item['quality']['face_pixels'] >= 90 and item['quality']['sharpness'] >= 35 and clear_agreement:
                            continue
                    candidates.append(item)
                if not candidates:
                    continue
                model_started = time.perf_counter()
                if adapter is not None:
                    try:
                        outcomes = adapter.embed_many(image,[item['points'] for item in candidates],strict_alignment=False,executor=getattr(engine,'preprocess_pool',None))
                    except Exception as exc:
                        outcomes = [exc]*len(candidates)
                elif hasattr(engine,'embed_many'):
                    try:
                        outcomes = engine.embed_many(image,[item['points'] for item in candidates],key,strict_alignment=False)
                    except Exception as exc:
                        outcomes = [exc]*len(candidates)
                else:
                    outcomes = []
                    for item in candidates:
                        try:
                            outcomes.append(engine.embed(image,item['points'],key,strict_alignment=False))
                        except Exception as exc:
                            outcomes.append(exc)
                for item, outcome in zip(candidates,outcomes):
                    try:
                        if isinstance(outcome,Exception):
                            raise outcome
                        vector, info = outcome
                        item['vectors'][key] = vector
                        gallery = galleries[key]
                        ranked = {}
                        for pid, name, reference in gallery:
                            score = float(np.clip(np.dot(vector, reference), -1, 1))
                            if pid not in ranked or score > ranked[pid][0]:
                                ranked[pid] = (score, name)
                        ranking = sorted(ranked.items(), key=lambda v: v[1][0], reverse=True)
                        height=item['quality'].get('original_face_size',[0,item['box'][3]-item['box'][1]])[1]
                        alignment=info.get('alignment_band','regular')
                        profile = self.store.calibration(versions[key], item['quality']['face_pixels'],
                            alignment_band=alignment,face_height=height)
                        if ranking:
                            pid, (score, name) = ranking[0]
                            margin = score-ranking[1][1][0] if len(ranking)>1 else None
                            # An absolute gate is still required for a one-person watchlist.
                            accepted,verification = model_gates(score,margin,profile,pid,ranking[1][0] if len(ranking)>1 else None)
                            item['scores'][key] = dict(person_id=pid, name=name, score=score, margin=margin,
                                runner_up_person_id=ranking[1][0] if len(ranking)>1 else None,
                                passed=accepted,verification_passed=verification, calibrated=bool(profile),
                                calibration_band=quality_band(item['quality']['face_pixels'],alignment,height),
                                **info)
                            if key == 'petalface':
                                item['scores'][key]['iqa_threshold_verified'] = bool(profile and
                                    profile.get('iqa_threshold') == info.get('iqa_threshold'))
                        else:
                            item['scores'][key] = dict(score=None, passed=False, calibrated=False, **info)
                    except Exception as exc:
                        item['scores'][key] = dict(error=str(exc),passed=False,calibrated=False)
                        if key == anchor_key:
                            item['error'] = str(exc)
                if key in self.shadow_models:
                    for item in candidates:
                        item['scores'][key]['shadow'] = True
                recognition_ms[key] = (time.perf_counter()-model_started)*1000
            # Mutual, unique face-to-face association. Crossing/ambiguous assignments reset evidence.
            used, overlays, events = set(), [], []
            old = {k:v for k,v in self.face_tracks.items() if 0 <= timestamp-v['time'] < 1.5}
            choices, geometric, association_diagnostics, geometry_links = [], [], [], []
            for item in processed:
                v = item['vectors'].get(anchor_key)
                nearby, options, observed_candidates, geometry_options, overlap_options = [], [], [], [], []
                primary_score = item['scores'].get('lvface', {})
                weak_appearance = (item['quality'].get('face_pixels', 0) < 90 or
                    not primary_score.get('passed') or primary_score.get('alignment_band') == 'high_residual')
                center = (item['box'][:2]+item['box'][2:4])*.5
                for key, previous in old.items():
                    predicted = predicted_face_box(previous, timestamp)
                    overlap = max(iou(item['box'], previous['box']), iou(item['box'], predicted))
                    old_vector = previous['vector']
                    similarity = float(np.dot(v, old_vector)) if v is not None and old_vector is not None else None
                    old_center = (previous['box'][:2]+previous['box'][2:4])*.5
                    predicted_center = (predicted[:2]+predicted[2:4])*.5
                    scale = max(np.linalg.norm(item['box'][2:4]-item['box'][:2]),
                                np.linalg.norm(previous['box'][2:4]-previous['box'][:2]))
                    displacement = min(np.linalg.norm(center-old_center), np.linalg.norm(center-predicted_center))
                    appearance_recovery = similarity is not None and similarity > .4 and displacement <= 1.5*scale
                    if key in self.visible_tracks:
                        observed_candidates.append(dict(track_id=key, overlap=float(overlap),
                            appearance_similarity=similarity, displacement_diagonals=float(displacement/max(scale, 1)),
                            geometry_passed=bool(overlap > .25 or appearance_recovery),
                            appearance_passed=bool(similarity is None or similarity > .4)))
                    if overlap > .25 or appearance_recovery:
                        nearby.append(key)
                        if similarity is None or similarity > .4:
                            options.append(key)
                            if key in self.visible_tracks and overlap >= .5:
                                overlap_options.append(key)
                        elif weak_appearance and key in self.visible_tracks and overlap >= .5:
                            # Preserve an unambiguous location through unreliable appearance.
                            # This association supplies no biometric identity evidence.
                            geometry_options.append(key)
                geometric.append(nearby)
                association_diagnostics.append(min(observed_candidates,
                    key=lambda candidate:candidate['displacement_diagonals'], default=None))
                # A recently visible track takes precedence over stale fragments.
                active_options = [key for key in options if key in self.visible_tracks]
                choices.append(overlap_options or active_options or geometry_options or options)
                geometry_links.append(set(geometry_options) if not active_options else set())
            assignments = [options[0] if len(options)==1 and sum(options[0] in c for c in choices)==1 else None
                           for options in choices]
            # An uncertain region must retire its old hypotheses. Keeping those clones
            # alive makes the next stable face ambiguous forever, producing a new ID each frame.
            retired = set()
            while True:
                expanded = retired | {k for index, nearby in enumerate(geometric)
                                      if assignments[index] is None or assignments[index] in retired for k in nearby}
                if expanded == retired:
                    break
                retired = expanded
            for track in retired:
                self.face_tracks.pop(track, None)
                self.temporal.reset_track(track)
            frame_evidence = [identity_evidence(item['scores']) for item in processed]
            verification_frames = [verification_evidence(item['scores'],self.temporal.anchor_identity(assignments[index],timestamp))
                                   for index,item in enumerate(processed)]
            reverify_candidates = []
            for index,(pid,eligible,conflict,_) in enumerate(verification_frames):
                track=assignments[index]
                prior_face=old.get(track)
                geometry_reset=bool(prior_face and track in geometry_links[index] and not prior_face.get('geometry_only',False))
                reverify_candidates.append(bool(eligible and not conflict and not geometry_reset and
                    self.temporal.can_reverify(track,pid,timestamp)))
            identity_claims = Counter()
            for index,(proposed,eligible,_,_) in enumerate(frame_evidence):
                claim=proposed if eligible else (verification_frames[index][0] if reverify_candidates[index] else None)
                if claim in selected:identity_claims[claim]+=1
            current_regions = [dict(track_id=assignments[i],bbox=item['box'][:4].tolist()) for i,item in enumerate(processed)]
            for index, item in enumerate(processed):
                track = assignments[index]
                if track in retired:
                    track = None
                if track is None or track in used:
                    track = self.next_face
                    self.next_face += 1
                used.add(track)
                previous = old.get(track)
                velocity = np.zeros(2, dtype=np.float32)
                tracking = dict(method='new', previous_iou=None, predicted_iou=None, appearance_similarity=None)
                if previous is not None:
                    elapsed = timestamp-previous['time']
                    current_center = (item['box'][:2]+item['box'][2:4])*.5
                    old_center = (previous['box'][:2]+previous['box'][2:4])*.5
                    if elapsed > 0:
                        velocity = (current_center-old_center)/elapsed
                    observed_iou = iou(item['box'], previous['box'])
                    predicted_iou = iou(item['box'], predicted_face_box(previous, timestamp))
                    current_vector, old_vector = item['vectors'].get(anchor_key), previous['vector']
                    tracking = dict(method=('appearance' if predicted_iou <= .25 else 'motion') if observed_iou <= .25 else 'overlap',
                        previous_iou=float(observed_iou), predicted_iou=float(predicted_iou),
                        appearance_similarity=float(np.dot(current_vector, old_vector))
                        if current_vector is not None and old_vector is not None else None,
                        appearance_age_seconds=float(timestamp-previous.get('vector_time',previous['time'])))
                geometry_only = previous is not None and track in geometry_links[index]
                if geometry_only:
                    tracking['method'] = 'geometry'
                    tracking['identity_evidence_reset'] = not previous.get('geometry_only', False)
                    if tracking['identity_evidence_reset']:
                        self.temporal.reset_track(track)
                current_vector = item['vectors'].get(anchor_key)
                vector_time=timestamp if current_vector is not None else (previous.get('vector_time',previous['time']) if previous else timestamp)
                if current_vector is None and previous is not None:
                    current_vector = previous['vector']
                self.face_tracks[track] = dict(box=item['box'], vector=current_vector,
                                              time=timestamp,vector_time=vector_time, velocity=velocity, geometry_only=geometry_only,
                                              )
                tracking['candidate_count'] = len(choices[index])
                if tracking['method'] == 'new' and association_diagnostics[index] is not None:
                    tracking['nearest_previous'] = association_diagnostics[index]
                proposed, eligible, disagreement, fusion = frame_evidence[index]
                verify_pid,verify_eligible,verify_conflict,verify_info=verification_frames[index]
                previous_identity=self.temporal.tracks.get(track)
                anchor_pid=self.temporal.anchor_identity(track,timestamp)
                trusted_pid=anchor_pid or (previous_identity['pid'] if previous_identity else None)
                held=anchor_pid is not None
                if held and verify_conflict:disagreement=True;eligible=False
                reverify=bool(not eligible and held and not disagreement and verify_eligible and
                    self.temporal.can_reverify(track,verify_pid,timestamp))
                if reverify:proposed=verify_pid;eligible=True
                duplicate_candidate = eligible and identity_claims[proposed] > 1
                fusion['conflicting_face_claims'] = identity_claims[proposed] if eligible else 0
                eligible = eligible and not duplicate_candidate and not disagreement
                direct_eligible = eligible
                pid = proposed if proposed in selected and not disagreement else None
                qualified_conflict = bool(trusted_pid and (
                    fusion['calibrated_support'] and trusted_pid != proposed or
                    held and verify_info['calibrated_support'] and verify_pid and trusted_pid != verify_pid))
                fusion['track_reverification']=bool(reverify and eligible)
                fusion['verification_roles']=verify_info['model_roles']
                bridge = bool(not eligible and pid and fusion['calibrated_support'] >= 1 and
                    not disagreement and not qualified_conflict and not identity_claims[proposed] and
                    self.temporal.can_bridge(track,pid,timestamp,settings))
                eligible = eligible or bridge
                fusion['temporal_bridge'] = bridge
                resume = bool(direct_eligible and held and pid == anchor_pid and track in self.visible_tracks and
                              stable_association(tracking))
                fusion['track_resumed'] = resume
                fresh_count = verify_info['calibrated_support'] if reverify else fusion['calibrated_support']
                continuity = bool(not eligible and not disagreement and not duplicate_candidate and not qualified_conflict and
                    anchor_pid in selected and not identity_claims[anchor_pid] and trusted_identity_link(dict(track_id=track,bbox=item['box'][:4].tolist(),
                        tracking=tracking,model_scores=item['scores']),self.previous_overlays,current_regions,settings.get('identity_tracking_profile')))
                status, emit, visit = self.temporal.update(track, pid if eligible else None, timestamp, settings,
                    reset=bool(disagreement or duplicate_candidate or qualified_conflict), frame_id=frame_id,
                    multimodel=direct_eligible, reverified=resume, fresh_model_count=fresh_count, continuity=continuity)
                if status == 'tracked':pid=self.temporal.label_chains[track]
                fusion['tracking_continuity'] = status == 'tracked'
                if pid and not eligible and status != 'tracked':
                    status = 'candidate'
                reason = ('Duy trì danh tính đã xác nhận qua liên kết ảnh liên tục; chưa phải xác nhận sinh trắc mới' if status=='tracked' else None) or ('Bằng chứng sinh trắc đã hiệu chuẩn xác minh lại danh tính trên track ổn định' if reverify and eligible else None) or ('Một nhánh đã hiệu chuẩn xác minh lại cùng danh tính; bằng chứng nhiều model không quá 0,5 giây' if bridge else None) or item['error'] or ('Nhiều mặt cùng gợi ý một danh tính; cần xác minh lại' if duplicate_candidate else 'Các model bất đồng' if disagreement else
                    'Đủ bằng chứng đã hiệu chuẩn' if eligible else 'Chưa đủ bằng chứng/hiệu chuẩn theo từng model')
                box = [float(v) for v in item['box'][:4]]
                overlay = dict(track_id=track, bbox=box, face_bbox=box, status=status,
                    person_id=pid if status in ('confirmed','tracked') else None,
                    identity_source='track_continuity' if status=='tracked' else 'biometric' if status=='confirmed' else None,
                    name=person_names.get(pid) if status in ('confirmed','tracked') else None,
                    reason=reason, detector=item['detector'], quality=item['quality'], model_scores=item['scores'], fusion=fusion, tracking=tracking)
                if diagnostic_details:
                    overlay['landmarks'] = np.asarray(item['points']).tolist() if item['points'] is not None else None
                    overlay['model_routing'] = item.get('routing',{})
                overlays.append(overlay)
                if emit:
                    events.append(dict(**overlay, visit=visit, frame_id=frame_id, timestamp=timestamp,
                        confirmation_seconds=timestamp-self.temporal.tracks[track]['first_qualified']))
            self.previous_overlays = overlays
            self.visible_tracks = used
            self.temporal.finish_frame(timestamp)
            self.face_tracks = {k:v for k,v in self.face_tracks.items() if timestamp-v['time'] < 1.5}
        ms = (time.perf_counter()-started)*1000
        return dict(frame_id=frame_id, timestamp=timestamp, width=image.shape[1], height=image.shape[0],
            pipeline_revision='hybrid-contextual-fusion-v21'+('-shadow' if self.shadow_models else ''),
            shadow_models=sorted(self.shadow_models),
            measurement_evidence_reset=reset_temporal,
            overlays=overlays, events=events, processing_ms=ms, ai_fps=1000/ms if ms else 0,
            recognition_ms=recognition_ms, detector_ms=detector_ms, detector_fps=1000/detector_ms if detector_ms else 0,
            backlog=0, gpu=engine.gpu())


engine = Engine()

