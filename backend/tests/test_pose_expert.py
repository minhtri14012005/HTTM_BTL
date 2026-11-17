"""Candidate integrity and numeric portability, never biometric accuracy from fixtures."""
from pathlib import Path
import json
import numpy as np
import pytest
from backend.pose_expert import KPRPEExpert
from backend.models import REFERENCE


def test_candidate_rejects_missing_or_changed_source_before_model_loading(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files={'required.py':'abc'})))
    with pytest.raises(ValueError,match='Thiếu file'):
        KPRPEExpert(tmp_path)
    (tmp_path/'required.py').write_text('changed')
    with pytest.raises(ValueError,match='Sai checksum'):
        KPRPEExpert(tmp_path)


def test_official_candidate_loads_strict_weights_and_matches_cpu_cuda():
    import torch
    root=Path(__file__).resolve().parents[2]/'models/candidates/kprpe12m'
    if not (root/'model.safetensors').is_file():pytest.skip('Optional candidate checkpoint is not installed.')
    candidate=KPRPEExpert(root,'cpu')
    image=np.full((160,160,3),120,np.uint8)
    cpu,info=candidate.embed(image,REFERENCE)
    assert cpu.shape==(512,) and np.isfinite(cpu).all() and abs(np.linalg.norm(cpu)-1)<1e-5
    assert info['version']==candidate.version and info['rpe_indexing']=='official_pytorch_fallback'
    if torch.cuda.is_available():
        candidate.model.cuda();candidate.aligner.cuda();candidate.device='cuda'
        cuda,_=candidate.embed(image,REFERENCE)
        np.testing.assert_allclose(cpu,cuda,rtol=1e-4,atol=2e-5)
    outputs=candidate.embed_many(image,[REFERENCE,np.full((5,2),-1,np.float32),REFERENCE])
    assert isinstance(outputs[1],ValueError) and not isinstance(outputs[0],Exception) and not isinstance(outputs[2],Exception)
    np.testing.assert_allclose(outputs[0][0],outputs[2][0],atol=1e-6)
    from backend.models import normalize
    alternate=image.copy();alternate[:,:,0]=30;alternate[:,:,2]=220
    tensors=[candidate.prepare(source,REFERENCE)[0] for source in (image,alternate)]
    expected=[normalize(candidate.forward_batch([tensor])[0][0]) for tensor in tensors]
    actual=[normalize(vector) for vector,_ in candidate.forward_batch(tensors)]
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=2e-5)
