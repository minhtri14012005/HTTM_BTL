"""Eye-aligned LVFace experiment with a separate embedding space and shadow-only decisions."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .models import ONNXRecognizer, REFERENCE, batched_embeddings, rgb_tensor


def align_eyes(image, points):
    points = np.asarray(points, np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('Invalid eye landmarks')
    height, width = image.shape[:2]
    eyes = points[:2]
    if (eyes < 0).any() or (eyes[:, 0] >= width).any() or (eyes[:, 1] >= height).any():
        raise ValueError('Eye landmarks outside image')
    baseline = eyes[1] - eyes[0]
    squared_length = float(baseline @ baseline)
    if squared_length < 1 or baseline[0] <= 0:
        raise ValueError('Degenerate or reversed eye baseline')
    target = REFERENCE[1] - REFERENCE[0]
    real = float(target @ baseline) / squared_length
    imaginary = float(baseline[0] * target[1] - baseline[1] * target[0]) / squared_length
    rotation = np.array([[real, -imaginary], [imaginary, real]], np.float32)
    transform = np.column_stack((rotation, REFERENCE[0] - rotation @ eyes[0]))
    coverage = cv2.warpAffine(np.ones((height, width), np.uint8), transform, (112, 112))
    if coverage.mean() < .85:
        raise ValueError('Eye-aligned crop extends outside image')
    crop = cv2.warpAffine(image, transform, (112, 112))
    projected = points @ rotation.T + transform[:, 2]
    residual = float(np.sqrt(np.mean(np.sum((projected - REFERENCE) ** 2, axis=1))))
    return crop, residual


class LVFaceEyeExpert:
    shadow_only = True
    recognizer_family = 'lvface'
    fallback_for = 'lvface'
    routing_alignment = 'regular'

    def __init__(self, root, device='cpu', recognizer=None):
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
        checkpoint = root / 'lvface.onnx'
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest['files']['lvface.onnx']:
            raise ValueError('LVFace checkpoint checksum mismatch')
        if recognizer is not None and recognizer.key != 'lvface':
            raise ValueError('Eye adapter requires the LVFace recognizer')
        self.model = recognizer if recognizer is not None else ONNXRecognizer(checkpoint, device, 'lvface')
        self.provider = self.model.provider
        self.batch_size = self.model.batch_size
        self.version = f'lvface_eye:{digest[:20]}:eyes112-rgb-fp32-v1'

    def prepare(self, image, points, strict_alignment=True):
        crop, residual = align_eyes(image, points)
        return rgb_tensor(crop), dict(version=self.version, recognizer_family=self.recognizer_family,
            alignment_method='eyes_similarity_v1', alignment_band='regular',
            five_point_residual=residual, input_size=112, numeric_precision='fp32')

    def forward_batch(self, tensors):
        return self.model.forward_batch(tensors)

    def embed_many(self, image, landmarks, strict_alignment=True, executor=None):
        return batched_embeddings(self, image, landmarks, strict_alignment, executor)

    def embed(self, image, points, strict_alignment=True):
        result = self.embed_many(image, [points], strict_alignment)[0]
        if isinstance(result, Exception):
            raise result
        return result
