"""Alignment geometry and shadow isolation, with real checkpoint portability checks."""
from pathlib import Path

import numpy as np
import pytest

from backend.eye_expert import LVFaceEyeExpert, align_eyes
from backend.models import REFERENCE


def test_eye_alignment_preserves_native_pixels_and_ignores_occluded_lower_landmarks():
    image = np.random.default_rng(17).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    original = image.copy()
    crop, error = align_eyes(image, REFERENCE)
    np.testing.assert_array_equal(crop, image)
    assert error < 1e-4
    covered = REFERENCE.copy()
    covered[2:] += [5, -20]
    other, residual = align_eyes(image, covered)
    np.testing.assert_array_equal(crop, other)
    np.testing.assert_array_equal(image, original)
    assert residual > error


@pytest.mark.parametrize('condition', ['collapsed', 'reversed', 'outside', 'nonfinite', 'clipped'])
def test_eye_alignment_rejects_unsafe_geometry(condition):
    points = REFERENCE.copy()
    if condition == 'collapsed':
        points[1] = points[0]
    elif condition == 'reversed':
        points[:2] = points[1::-1]
    elif condition == 'outside':
        points[0, 0] = -1
    elif condition == 'nonfinite':
        points[0, 0] = np.nan
    else:
        points[:2] -= [35, 45]
    with pytest.raises(ValueError):
        align_eyes(np.zeros((112, 112, 3), np.uint8), points)


def test_eye_adapter_cannot_be_promoted_by_pipeline_options(tmp_path):
    from backend.ai import Pipeline
    from backend.store import Store
    adapter = object.__new__(LVFaceEyeExpert)
    pipeline = Pipeline(object(), Store(tmp_path), model_overrides={'lvface_eye': adapter}, shadow_models=[])
    assert pipeline.shadow_models == {'lvface_eye'}


def test_eye_adapter_real_checkpoint_cpu_cuda_and_separate_space():
    import cv2
    import torch
    from skimage import data
    root = Path(__file__).resolve().parents[2] / 'models/hybrid'
    if not (root / 'lvface.onnx').exists():
        pytest.skip('LVFace checkpoint is unavailable')
    image = cv2.cvtColor(data.astronaut()[60:172, 170:282], cv2.COLOR_RGB2BGR)
    cpu = LVFaceEyeExpert(root, 'cpu')
    tensor, info = cpu.prepare(image, REFERENCE)
    assert tensor.shape == (1, 3, 112, 112) and tensor.min() >= -1 and tensor.max() <= 1
    shared = LVFaceEyeExpert(root, 'cpu', recognizer=cpu.model)
    assert shared.model is cpu.model
    vector, metadata = cpu.embed(image, REFERENCE)
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)
    assert metadata['recognizer_family'] == 'lvface'
    assert cpu.version.startswith('lvface_eye:') and cpu.version.endswith(':eyes112-rgb-fp32-v1')
    assert cpu.shadow_only and info['alignment_method'] == 'eyes_similarity_v1'
    if torch.cuda.is_available():
        gpu = LVFaceEyeExpert(root, 'cuda')
        other, _ = gpu.embed(image, REFERENCE)
        assert vector @ other > .9999
    outcomes = cpu.embed_many(image, [REFERENCE, np.zeros((5, 2), np.float32), REFERENCE])
    assert isinstance(outcomes[1], ValueError)
    np.testing.assert_allclose(outcomes[0][0], outcomes[2][0], atol=1e-6)
