"""Opt-in KP-RPE adapter; evaluated as a shadow branch before identity voting."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from .models import align, rgb_tensor, namespace, module, batched_embeddings


class KPRPEExpert:
    def __init__(self, root, device='cpu'):
        import torch
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        self.root, self.device, self.batch_size = Path(root), device, 1
        if device not in ('cpu','cuda') or device=='cuda' and not torch.cuda.is_available():
            raise ValueError('Thiết bị KP-RPE không khả dụng.')
        manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf8'))
        for name,expected in manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                raise ValueError('Thiếu file candidate KP-RPE: '+name)
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:raise ValueError('Sai checksum candidate KP-RPE: '+name)
        self.version='kprpe12m:'+manifest['files']['model.safetensors'][:20]+':arc112-dfa-fp32-v2'
        prefix='facewatch_kprpe_candidate'
        if prefix not in sys.modules:
            namespace(prefix,self.root)
            namespace(prefix+'.models',self.root/'models')
            namespace(prefix+'.models.vit_kprpe',self.root/'models/vit_kprpe')
            rpe_path=self.root/'models/vit_kprpe/RPE/__init__.py'
            source=rpe_path.read_text(encoding='utf8')
            # Upstream attempts a C++ compiler install during import. Use its
            # existing pure PyTorch indexing fallback, identically on CPU/GPU.
            portable='from .KPRPE import kprpe_shared\n'+source[source.index('def build_rpe('):]
            module(prefix+'.models.vit_kprpe.RPE',rpe_path,{source:portable})
            module(prefix+'.models.vit_kprpe.vit',self.root/'models/vit_kprpe/vit.py')
            namespace(prefix+'.aligners',self.root/'aligners')
            module(prefix+'.aligners.differentiable_face_aligner',self.root/'aligners/differentiable_face_aligner/__init__.py')
        config=OmegaConf.load(self.root/'pretrained_model/model.yaml')
        architecture=sys.modules[prefix+'.models.vit_kprpe.vit'].VisionTransformerWithKPRPE
        self.model=architecture(img_size=112,patch_size=8,num_classes=512,embed_dim=512,depth=24,
            mlp_ratio=3,num_heads=16,drop_path_rate=.1,norm_layer='ln',mask_ratio=config.mask_ratio,rpe_config=config.rpe_config)
        weights=load_file(str(self.root/'model.safetensors'),device='cpu')
        self.model.load_state_dict({key.removeprefix('model.net.'):value for key,value in weights.items()},strict=True)
        aligner_class=sys.modules[prefix+'.aligners.differentiable_face_aligner'].DifferentiableFaceAligner
        self.aligner=aligner_class.from_config(OmegaConf.load(self.root/'pretrained_model/aligner.yaml'))
        self.aligner.load_state_dict(torch.load(self.root/'pretrained_model/aligner.pt',map_location='cpu',weights_only=True),strict=True)
        self.model.eval().to(device);self.aligner.eval().to(device)
        self.model.requires_grad_(False);self.aligner.requires_grad_(False)

    def prepare(self,image,points,strict_alignment=True):
        crop,error=align(image,points,strict=strict_alignment)
        return rgb_tensor(crop),dict(alignment_error=error,input_size=112,numeric_precision='fp32',
            alignment_band='regular' if error<=112*.09 else 'high_residual',version=self.version)

    def forward_batch(self,tensors):
        import torch
        x=torch.from_numpy(np.concatenate(tensors)).to(self.device)
        with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
            # Official infer_aligner_keypoint_model_pipeline feeds the original
            # square input and DFA's normalized original landmarks, not the DFA warp.
            _,points,_,confidence,_,_=self.aligner(x)
            if points is None or not torch.isfinite(points).all():
                raise ValueError('DFA không trả landmarks hữu hạn cho input vuông.')
            vectors=self.model(x,points).float().cpu().numpy()
            confidence=confidence.flatten().cpu().numpy()
        return [(vector,dict(dfa_score=float(score),execution=self.device+'_fp32',
                             rpe_indexing='official_pytorch_fallback')) for vector,score in zip(vectors,confidence)]

    def embed_many(self,image,landmarks,strict_alignment=True,executor=None):
        return batched_embeddings(self,image,landmarks,strict_alignment,executor)

    def embed(self,image,points,strict_alignment=True):
        result=self.embed_many(image,[points],strict_alignment)[0]
        if isinstance(result,Exception):raise result
        return result
