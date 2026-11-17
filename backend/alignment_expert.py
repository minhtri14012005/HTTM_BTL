"""ARoFace pretrained alignment specialist; shadow-only until independent validation."""
import hashlib,json,importlib.util
from pathlib import Path
import numpy as np
from .models import align,rgb_tensor,batched_embeddings

class ARoFaceExpert:
    shadow_only=True
    def __init__(self,root,device='cpu'):
        import torch
        self.root,self.device,self.batch_size=Path(root),device,1
        if device not in ('cpu','cuda') or device=='cuda' and not torch.cuda.is_available():
            raise ValueError('Thiết bị ARoFace không khả dụng.')
        manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        for name,digest in manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()):raise ValueError('Đường dẫn candidate không hợp lệ.')
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=digest:raise ValueError('Sai checksum ARoFace: '+name)
        spec=importlib.util.spec_from_file_location('facewatch_aroface_iresnet',self.root/'iresnet.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.model=module.iresnet100(num_features=512,fp16=False)
        weights=torch.load(self.root/'adaface_aroface_webface12m.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(weights,strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.version='aroface:'+manifest['files']['adaface_aroface_webface12m.pt'][:20]+':arc112-rgb-single-v1'
    def prepare(self,image,points,strict_alignment=True):
        crop,error=align(image,points,strict=strict_alignment)
        return rgb_tensor(crop),dict(version=self.version,input_size=112,numeric_precision='fp32',
            alignment_error=error,alignment_band='regular' if error<=112*.09 else 'high_residual')
    def forward_batch(self,tensors):
        import torch
        x=torch.from_numpy(np.concatenate(tensors)).to(self.device)
        with torch.inference_mode(),torch.backends.cudnn.flags(allow_tf32=False):
            vectors=self.model(x).float().cpu().numpy()
        return [(vector,dict(execution=self.device+'_fp32',flip_fusion='none_author_single_forward')) for vector in vectors]
    def embed_many(self,image,landmarks,strict_alignment=True,executor=None):
        return batched_embeddings(self,image,landmarks,strict_alignment,executor)
    def embed(self,image,points,strict_alignment=True):
        result=self.embed_many(image,[points],strict_alignment)[0]
        if isinstance(result,Exception):raise result
        return result
