"""Optional adapters share the same embedding and version contract."""
from pathlib import Path

def build_expert(key,root,device,base_recognizer=None):
    root=Path(root)
    if key=='kprpe12m':
        from .pose_expert import KPRPEExpert
        return KPRPEExpert(root/'kprpe12m',device)
    if key=='maskinv_hg':
        from .mask_expert import MaskInvExpert
        return MaskInvExpert(root/'maskinv_hg',device)
    if key=='lvface_eye':
        from .eye_expert import LVFaceEyeExpert
        return LVFaceEyeExpert(root.parent/'hybrid',device,base_recognizer)
    if key=='aroface':
        from .alignment_expert import ARoFaceExpert
        return ARoFaceExpert(root/'aroface',device)
    raise ValueError('Model chuyên gia chưa được hỗ trợ.')
