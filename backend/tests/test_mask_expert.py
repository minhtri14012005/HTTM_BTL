"""Real optional checkpoint contract; isolated fixtures, no physical camera."""
from pathlib import Path
import numpy as np
import pytest
from backend.models import REFERENCE
from backend.mask_expert import MaskInvExpert

ROOT=Path(__file__).resolve().parents[2]/'models/candidates/maskinv_hg'

@pytest.mark.skipif(not (ROOT/'manifest.json').exists(),reason='Optional MaskInv-HG is not installed')
def test_maskinv_real_cpu_gpu_preprocessing_flip_and_normalization():
    import torch,cv2
    from skimage import data
    torch.set_num_threads(2)
    image=cv2.cvtColor(data.astronaut()[60:172,170:282],cv2.COLOR_RGB2BGR)
    cpu=MaskInvExpert(ROOT,'cpu')
    tensor,info=cpu.prepare(image,REFERENCE)
    assert tensor.shape==(1,3,112,112) and np.isfinite(tensor).all()
    assert tensor.min()>=-1 and tensor.max()<=1 and info['input_size']==112
    vector,meta=cpu.embed(image,REFERENCE)
    assert vector.shape==(512,) and np.isfinite(vector).all() and np.linalg.norm(vector)==pytest.approx(1,abs=1e-5)
    assert cpu.shadow_only and meta['flip_fusion']=='author_raw_sum_then_l2'
    x=torch.from_numpy(tensor)
    with torch.inference_mode():expected=(cpu.model(x)+cpu.model(x.flip([-1]))).numpy()[0]
    expected/=np.linalg.norm(expected)
    assert np.allclose(vector,expected,atol=1e-5)
    if torch.cuda.is_available():
        gpu=MaskInvExpert(ROOT,'cuda');other,_=gpu.embed(image,REFERENCE)
        assert float(vector@other)>.9999


@pytest.mark.parametrize('expert', ['maskinv_hg', 'lvface_eye'])
def test_maskinv_shadow_only_and_fresh_consent_api(tmp_path,monkeypatch,expert):
    import backend.main as main
    import backend.experts as experts
    from backend.store import Store
    from fastapi.testclient import TestClient
    store=Store(tmp_path);calls=[]
    class Adapter:version=expert+':unit';shadow_only=True
    monkeypatch.setattr(main,'store',store);monkeypatch.setattr(main,'diagnostics',{})
    monkeypatch.setattr(main,'active_live',None);monkeypatch.setattr(main,'model_loading',False)
    monkeypatch.setattr(main.engine,'ready',True);monkeypatch.setattr(main.engine,'load',lambda:None)
    monkeypatch.setattr(store,'reindex',lambda _:[])
    monkeypatch.setattr(experts,'build_expert',lambda *args:(calls.append(args) or Adapter()))
    body=dict(ready=False,scenario='Đeo khẩu trang 3 m',expected_present=False,expert=expert)
    with TestClient(main.app) as client:
        assert client.post('/api/diagnostics',json=body).status_code==422 and not calls
        assert client.post('/api/diagnostics',json=dict(body,ready=True,expert_mode='calibrated')).status_code==400
        assert len(calls)==(1 if expert=='lvface_eye' else 0)
        calls.clear()
        response=client.post('/api/diagnostics',json=dict(body,ready=True))
        assert response.status_code==201 and len(calls)==1
        run=main.diagnostics[response.json()['id']]
        assert run.context['shadow_models']=={expert:expert+':unit'}
        assert client.post('/api/diagnostics/'+run.id+'/stop').status_code==200
        assert not run.runtime_models and main.active_live is None


def test_shadow_only_adapter_cannot_be_promoted_by_pipeline_options(tmp_path):
    from backend.ai import Pipeline
    from backend.store import Store
    class Adapter:shadow_only=True
    pipe=Pipeline(object(),Store(tmp_path),model_overrides={'maskinv_hg':Adapter()},shadow_models=[])
    assert 'maskinv_hg' in pipe.shadow_models


def test_validated_eye_and_pose_combination_requires_profiles_and_fresh_consent(tmp_path, monkeypatch):
    import backend.main as main
    import backend.experts as experts
    from backend.store import Store
    from fastapi.testclient import TestClient
    store = Store(tmp_path)
    calls = []
    class Adapter:
        shadow_only = True
        def __init__(self, key): self.version = key + ':unit'
    monkeypatch.setattr(main, 'store', store)
    monkeypatch.setattr(main, 'diagnostics', {})
    monkeypatch.setattr(main, 'active_live', None)
    monkeypatch.setattr(main, 'model_loading', False)
    monkeypatch.setattr(main.engine, 'ready', True)
    monkeypatch.setattr(main.engine, 'load', lambda: None)
    monkeypatch.setattr(store, 'reindex', lambda _: [])
    monkeypatch.setattr(experts, 'build_expert', lambda key,*args: (calls.append(key) or Adapter(key)))
    body = dict(ready=True,scenario='Tổng hợp 6 m',expected_present=False,
        expert='kprpe12m',additional_experts=['lvface_eye'],expert_mode='calibrated',purpose='validation')
    with TestClient(main.app) as client:
        assert client.post('/api/diagnostics',json=dict(body,ready=False)).status_code == 422
        assert not calls
        assert client.post('/api/diagnostics',json=dict(body,purpose='calibration')).status_code == 400
        assert not calls
        assert client.post('/api/diagnostics',json=body).status_code == 400
        store.calibrate('kprpe12m:unit','tiny',dict(threshold=.8,margin=.1,validation_negative_observations=10,validation_false_passes=0))
        assert client.post('/api/diagnostics',json=body).status_code == 400
        store.calibrate('lvface_eye:unit','tiny',dict(threshold=.8,margin=.1,validation_negative_observations=10,validation_false_passes=0))
        response = client.post('/api/diagnostics',json=body)
        assert response.status_code == 201, response.text
        run = main.diagnostics[response.json()['id']]
        assert run.runtime_shadow_models == []
        assert run.runtime_models['lvface_eye'].shadow_only is False
        assert run.runtime_models['lvface_eye'].allowed_bands == {'tiny'}
        assert set(run.context['fusion_experts']) == {'kprpe12m','lvface_eye'}
        assert main.active_live is None
        assert client.post('/api/diagnostics/'+run.id+'/stop').status_code == 200
        assert not run.runtime_models
