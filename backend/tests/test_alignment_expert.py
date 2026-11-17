"""Real optional ARoFace contract, isolated API fixtures; no physical camera."""
from pathlib import Path
import pytest,numpy as np
from backend.models import REFERENCE,normalize
ROOT=Path(__file__).resolve().parents[2]/'models/candidates/aroface'
@pytest.mark.skipif(not (ROOT/'manifest.json').exists(),reason='Optional ARoFace not installed')
def test_aroface_real_cpu_gpu_author_single_forward_and_preprocessing():
    import torch,cv2
    from skimage import data
    from backend.alignment_expert import ARoFaceExpert
    torch.set_num_threads(2)
    image=cv2.cvtColor(data.astronaut()[60:172,170:282],cv2.COLOR_RGB2BGR)
    cpu=ARoFaceExpert(ROOT,'cpu');tensor,info=cpu.prepare(image,REFERENCE)
    assert tensor.shape==(1,3,112,112) and tensor.min()>=-1 and tensor.max()<=1
    vector,meta=cpu.embed(image,REFERENCE)
    with torch.inference_mode():expected=normalize(cpu.model(torch.from_numpy(tensor)).numpy()[0])
    assert np.allclose(vector,expected,atol=1e-5) and cpu.shadow_only
    assert meta['flip_fusion']=='none_author_single_forward' and vector.shape==(512,)
    if torch.cuda.is_available():
        gpu=ARoFaceExpert(ROOT,'cuda');other,_=gpu.embed(image,REFERENCE)
        assert float(vector@other)>.9999


def test_aroface_needs_fresh_consent_and_cannot_enter_calibrated_fusion(tmp_path,monkeypatch):
    import backend.main as main
    import backend.experts as experts
    from backend.store import Store
    from fastapi.testclient import TestClient
    store=Store(tmp_path);calls=[]
    class Adapter:version='aroface:unit';shadow_only=True
    monkeypatch.setattr(main,'store',store);monkeypatch.setattr(main,'diagnostics',{})
    monkeypatch.setattr(main,'active_live',None);monkeypatch.setattr(main,'model_loading',False)
    monkeypatch.setattr(main.engine,'ready',True);monkeypatch.setattr(main.engine,'load',lambda:None)
    monkeypatch.setattr(store,'reindex',lambda _:[])
    monkeypatch.setattr(experts,'build_expert',lambda *args:(calls.append(args) or Adapter()))
    body=dict(ready=False,scenario='Quay nghiêng/di chuyển 3 m',expected_present=False,expert='aroface',eyewear='none')
    with TestClient(main.app) as client:
        assert client.post('/api/diagnostics',json=body).status_code==422 and not calls
        assert client.post('/api/diagnostics',json=dict(body,ready=True,expert_mode='calibrated')).status_code==400 and not calls
        response=client.post('/api/diagnostics',json=dict(body,ready=True))
        assert response.status_code==201 and len(calls)==1
        run=main.diagnostics[response.json()['id']]
        assert run.context['shadow_models']=={'aroface':'aroface:unit'}
        assert run.context['eyewear']=='none' and run.protocol['stages'][0]['instruction'].startswith('KHÔNG ĐEO KÍNH')
        assert client.post('/api/diagnostics/'+run.id+'/stop').status_code==200
        assert not run.runtime_models and main.active_live is None


def test_aroface_factory_uses_the_actual_adapter_contract(tmp_path,monkeypatch):
    import backend.alignment_expert as adapter_module
    from backend.experts import build_expert
    sentinel=object();calls=[]
    def create(root,device):
        calls.append((root,device));return sentinel
    monkeypatch.setattr(adapter_module,'ARoFaceExpert',create)
    assert build_expert('aroface',tmp_path,'cpu') is sentinel
    assert calls==[(tmp_path/'aroface','cpu')]


@pytest.mark.parametrize('core_strong',[True,False])
def test_shadow_expert_wrong_high_score_neither_alerts_nor_vetoes_verified_core(tmp_path,core_strong):
    import threading
    from backend.ai import Pipeline
    from backend.store import Store
    first,wrong,noise=np.eye(512,dtype=np.float32)[:3]
    versions=dict(lvface='lvface:isolated',antelopev2='antelopev2:isolated',aroface='aroface:isolated')
    store=Store(tmp_path)
    pid=store.add('Isolated A',[(b'fixture',{v:first for v in versions.values()},{})])
    store.add('Isolated B',[(b'fixture',{v:wrong for v in versions.values()},{})])
    for v in versions.values():store.calibrate(v,'normal',dict(threshold=.8,margin=.08))
    class LocalEngine:
        lock=threading.RLock()
        status={k:dict(state='ready') for k in ('lvface','antelopev2')}
        def load(self):pass
        def detect(self,image):return [(np.array([20,20,132,132,.99]),None,'isolated')]
        def quality(self,*a,**kw):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kw):return (first if core_strong else noise),dict(version=self.versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    class Expert:
        version=versions['aroface'];shadow_only=True
        def embed_many(self,image,points,**kw):return [(wrong,dict(version=self.version,alignment_band='regular')) for _ in points]
    engine=LocalEngine();engine.versions={k:v for k,v in versions.items() if k!='aroface'}
    pipeline=Pipeline(engine,store,model_overrides={'aroface':Expert()},shadow_models=[])
    outputs=[pipeline.run(np.zeros((160,160,3),np.uint8),i,i*.2) for i in range(5)]
    assert all(o['overlays'][0]['model_scores']['aroface']['score']>.99 for o in outputs)
    assert all(o['overlays'][0]['fusion']['model_roles']['aroface']=='shadow_only' for o in outputs)
    alerts=[e for o in outputs for e in o['events']]
    if core_strong:
        assert outputs[-1]['overlays'][0]['status']=='confirmed'
        assert alerts and all(e['person_id']==pid for e in alerts)
    else:
        assert not alerts and all(o['overlays'][0]['person_id'] is None for o in outputs)
