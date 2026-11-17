"""Local adapters. Model spaces, alignment and preprocessing stay independent."""
import importlib.util
import sys
import types
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

REFERENCE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                      [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def module(name, path, replace=None):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    if replace:
        code = path.read_text(encoding='utf8')
        for before, after in replace.items():
            code = code.replace(before, after)
        exec(compile(code, str(path), 'exec'), obj.__dict__)
    else:
        spec.loader.exec_module(obj)
    return obj


def namespace(name, path):
    obj = types.ModuleType(name)
    obj.__path__ = [str(path)]
    sys.modules[name] = obj


def normalize(embedding):
    embedding = np.asarray(embedding, np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if embedding.size != 512 or not np.isfinite(embedding).all() or norm < 1e-8:
        raise ValueError('Embedding không hợp lệ.')
    return embedding / norm


def align(image, points, size=112, petal=False, strict=True):
    from skimage.transform import SimilarityTransform
    points = np.asarray(points, np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('Landmarks không hợp lệ.')
    h, w = image.shape[:2]
    if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError('Landmarks nằm ngoài ảnh.')
    eye_dx=float(points[1,0]-points[0,0]);mouth_dx=float(points[4,0]-points[3,0])
    # In a full profile, projected corners may overlap or cross by less than one
    # native pixel. Keep their original semantic indices; never swap or synthesize.
    tolerance=0. if strict else 1.
    if eye_dx < -tolerance or mouth_dx < -tolerance or (strict and (eye_dx<=0 or mouth_dx<=0)):
        raise ValueError('Sai thứ tự landmarks.')
    # PETAL official IJB evaluator warps to 112, then resizes to 120.
    warp_size = 112 if petal else size
    target = REFERENCE * (warp_size / 112)
    transform = SimilarityTransform()
    if not transform.estimate(points, target):
        raise ValueError('Không căn chỉnh được khuôn mặt.')
    error = float(np.sqrt(np.mean(np.sum((transform(points)-target)**2, axis=1))))
    if strict and error > warp_size * .09:
        raise ValueError(f'Alignment sai lệch {error:.1f}px.')
    crop = cv2.warpAffine(image, transform.params[:2], (warp_size, warp_size), borderValue=0)
    mask = cv2.warpAffine(np.ones((h, w), np.uint8), transform.params[:2], (warp_size, warp_size))
    if mask.mean() < .85:
        raise ValueError('Crop mặt bị cắt ngoài ảnh quá nhiều.')
    return cv2.resize(crop, (size, size)) if size != warp_size else crop, error


def rgb_tensor(crop):
    return np.ascontiguousarray((crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255 - .5) / .5)[None]


def ort_session(path, device, precise_matmul=False):
    import torch
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ['CPUExecutionProvider']
    if device == 'cuda':
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        providers = [('CUDAExecutionProvider', {'device_id': torch.cuda.current_device(),
                      'gpu_mem_limit': 1200 * 1024**2, 'arena_extend_strategy': 'kSameAsRequested',
                      'cudnn_conv_algo_search': 'HEURISTIC', 'use_tf32': '0' if precise_matmul else '1'}), 'CPUExecutionProvider']
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    return session



def batched_embeddings(recognizer, image, landmarks, strict_alignment=True, executor=None):
    """CPU preparation may overlap bounded GPU batches; preserve each face's order and errors."""
    results = [None] * len(landmarks)
    def prepare(points):
        started = time.perf_counter()
        try:
            tensor, info = recognizer.prepare(image, points, strict_alignment)
            info['preprocessing_ms'] = (time.perf_counter()-started)*1000
            return tensor, info
        except Exception as exc:
            return exc
    prepared = executor.map(prepare, landmarks) if executor is not None else map(prepare, landmarks)
    remaining = iter(enumerate(prepared))
    pending, finished = [], False
    while not finished or pending:
        while not finished and len(pending) < recognizer.batch_size:
            try:
                index, value = next(remaining)
            except StopIteration:
                finished = True
                break
            if isinstance(value, Exception):
                results[index] = value
            else:
                pending.append((index, *value))
        if not pending:
            continue
        chunk = pending[:recognizer.batch_size]
        started = time.perf_counter()
        try:
            outputs = recognizer.forward_batch([value[1] for value in chunk])
            if len(outputs) != len(chunk):
                raise ValueError('Model trả sai số embedding trong batch.')
        except Exception as exc:
            if len(chunk) > 1:
                recognizer.batch_size = max(1,len(chunk)//2)
                continue  # Retry the same faces in smaller chunks, never skip them.
            results[chunk[0][0]] = exc
            pending = pending[1:]
            continue
        elapsed = (time.perf_counter()-started)*1000
        for (index, _, info), (embedding, metadata) in zip(chunk,outputs):
            try:
                results[index] = (normalize(embedding),dict(info,**metadata,
                    latency_ms=info['preprocessing_ms']+elapsed/len(chunk),
                    latency_basis='batch_amortized',batch_size=len(chunk),batch_latency_ms=elapsed))
            except Exception as exc:
                results[index] = exc
        pending = pending[len(chunk):]
    return results


class ONNXRecognizer:
    def __init__(self, path, device, key):
        self.key = key
        self.session = ort_session(path, device, precise_matmul=True)
        self.provider = self.session.get_providers()[0]
        self.mean, self.std = 127.5, 127.5
        if key == 'antelopev2':
            import onnx
            graph = onnx.load(str(path)).graph
            nodes = [node.name for node in graph.node[:8]]
            if any(n.startswith(('Sub', '_minus')) for n in nodes) and any(n.startswith(('Mul', '_mul')) for n in nodes):
                self.mean, self.std = 0., 1.
        shape = self.session.get_inputs()[0].shape
        assert shape[-2:] == [112, 112]
        self.batch_size = 4 if key == 'lvface' and self.provider == 'CUDAExecutionProvider' and not isinstance(shape[0],int) else 1

    def prepare(self, image, points, strict_alignment=True):
        crop, error = align(image, points, strict=strict_alignment)
        if self.key == 'lvface':
            tensor = rgb_tensor(crop)
        else:
            tensor = cv2.dnn.blobFromImage(crop, 1/self.std, (112, 112), (self.mean,)*3, swapRB=True)
        return tensor, dict(alignment_error=error,input_size=112,numeric_precision='fp32',
                            alignment_band='regular' if error <= 112*.09 else 'high_residual')

    def forward_batch(self, tensors):
        output = self.session.run(None, {self.session.get_inputs()[0].name: np.concatenate(tensors)})[0]
        return [(vector,{}) for vector in output]

    def embed_many(self, image, landmarks, strict_alignment=True, executor=None):
        return batched_embeddings(self,image,landmarks,strict_alignment,executor)

    def embed(self, image, points, strict_alignment=True):
        tensor, info = self.prepare(image,points,strict_alignment)
        output = self.forward_batch([tensor])[0][0]
        return normalize(output), info



class RetinaR50:
    def __init__(self, root, device):
        import torch
        namespace('httm_retina', root/'retina/models')
        module('httm_retina.net', root/'retina/models/net.py')
        code = module('httm_retina.retinaface', root/'retina/models/retinaface.py', {'from models.net import': 'from .net import'})
        cfg = dict(name='Resnet50', pretrain=False, return_layers={'layer2': 1, 'layer3': 2, 'layer4': 3},
                   in_channel=256, out_channel=256)
        self.model = code.RetinaFace(cfg, phase='test')
        state = torch.load(root/'Resnet50_Final.pth', map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.model.eval().to(device)
        self.device = device
        self.prior_cache = {}
        self.mean = np.array((104,117,123),np.float32)

    def detect(self, image):
        import torch
        h, w = image.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray((image.astype(np.float32)-self.mean).transpose(2,0,1)))[None].to(self.device)
        with torch.inference_mode():
            loc, conf, landmarks = self.model(x)
            if (h, w) not in self.prior_cache:
                priors = [[(j+.5)*step/w, (i+.5)*step/h, size/w, size/h]
                          for step, sizes in zip((8, 16, 32), ((16, 32), (64, 128), (256, 512)))
                          for i in range((h+step-1)//step) for j in range((w+step-1)//step) for size in sizes]
                if len(self.prior_cache) >= 8:
                    self.prior_cache.clear()
                self.prior_cache[h, w] = torch.tensor(priors, device=self.device)
            priors = self.prior_cache[h, w]
            keep = conf[0, :, 1] >= .6
            p, loc, lm = priors[keep], loc[0, keep], landmarks[0, keep].reshape(-1, 5, 2)
            centers, sizes = p[:, :2]+loc[:, :2]*.1*p[:, 2:], p[:, 2:]*torch.exp(loc[:, 2:]*.2)
            boxes = torch.cat((centers-sizes/2, centers+sizes/2), 1) * torch.tensor([w, h, w, h], device=self.device)
            lm = (p[:, None, :2]+lm*.1*p[:, None, 2:]) * torch.tensor([w, h], device=self.device)
            scores = conf[0, keep, 1]
        return [(np.r_[b, s], k, 'RetinaFace R50') for b, s, k in zip(boxes.cpu().numpy(), scores.cpu().numpy(), lm.cpu().numpy())]


class YoloFace:
    def __init__(self, path, device):
        self.session = ort_session(path, device)
        self.provider = self.session.get_providers()[0]

    def detect(self, image):
        h, w = image.shape[:2]
        shape = self.session.get_inputs()[0].shape
        th, tw = [v if isinstance(v, int) else 640 for v in shape[-2:]]
        scale = min(tw/w, th/h)
        nw, nh = round(w*scale), round(h*scale)
        left, top = (tw-nw)//2, (th-nh)//2
        padded = np.full((th, tw, 3), 114, np.uint8)
        padded[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        rows = outputs[0][0]
        if rows.ndim != 2 or rows.shape[1] != 16:
            raise ValueError(f'YOLOv5-Face output không đúng định dạng decoded Nx16: {rows.shape}')
        result = []
        for row in rows:
            score = float(row[4] * row[15])
            if score < .6:
                continue
            cx, cy, bw, bh = row[:4]
            box = (np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]) - [left, top, left, top])/scale
            points = (row[5:15].reshape(5, 2) - [left, top])/scale
            result.append((np.r_[box, score], points, 'YOLOv5m-Face'))
        return result


class PetalFace:
    def __init__(self, root, device, threshold=.5):
        import torch
        from torch import nn
        namespace('httm_petal', root/'petal/backbones')
        code = module('httm_petal.swin_models_iqa', root/'petal/backbones/swin_models_iqa.py')
        self.model = code.SwinTransformer(lora_rank=8, lora_scale=1, img_size=120, patch_size=6,
            in_chans=3, num_classes=512, embed_dim=384, depths=[2, 18, 2], num_heads=[8, 16, 16],
            window_size=5, use_lora=True, reso=120)
        state = torch.load(root/'petalface.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        # CNNIQA implementation from IQA-PyTorch (MIT), same parameter names and forward.
        class CNNIQA(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 50, 7)
                self.fc1 = nn.Linear(100, 800)
                self.fc2 = nn.Linear(800, 800)
                self.fc3 = nn.Linear(800, 1)
                self.dropout = nn.Dropout()

            def forward(self, x):
                import torch.nn.functional as f
                h = self.conv1(x)
                h = torch.cat((h.amax((-2, -1)), h.amin((-2, -1))), 1)
                return self.fc3(f.relu(self.fc2(self.dropout(f.relu(self.fc1(h))))))
        self.iqa = CNNIQA()
        state = torch.load(root/'cnniqa.pth', map_location='cpu', weights_only=True)
        self.iqa.load_state_dict(state.get('params', state), strict=True)
        self.model.eval().to(device)
        self.iqa.eval().to(device)
        self.device, self.threshold = device, threshold
        self.batch_size = 4 if device == 'cuda' else 1

    def prepare(self, image, points, strict_alignment=True):
        crop, error = align(image,points,120,petal=True,strict=strict_alignment)
        return rgb_tensor(crop),dict(alignment_error=error,input_size=120,
            alignment_band='regular' if error <= 112*.09 else 'high_residual')

    def quality_forward(self, x):
        import torch
        count = x.shape[0]
        augmented = torch.cat((x,torch.flip(x,[3])),dim=0)
        scores = self.iqa(augmented)
        alphas = .5+scores-self.threshold
        embeddings = torch.nn.functional.normalize(self.model(augmented,alphas),dim=1)
        return embeddings[:count]+embeddings[count:],scores[:count],alphas[:count]

    def forward_batch(self, tensors):
        import torch
        x = torch.from_numpy(np.concatenate(tensors)).to(self.device)
        count = len(tensors)
        with torch.inference_mode():
            combined,scores,alphas = self.quality_forward(x)
            vectors = combined.cpu().numpy()
            quality = scores.flatten().cpu().numpy()
            blend = alphas.flatten().cpu().numpy()
        return [(vector,dict(iqa_score=float(score),alpha=float(alpha),iqa_threshold=self.threshold,
                              iqa_threshold_verified=False,execution='cuda_fp32_batch' if self.device == 'cuda' else 'cpu_fp32'))
                for vector,score,alpha in zip(vectors,quality,blend)]

    def embed_many(self, image, landmarks, strict_alignment=True, executor=None):
        return batched_embeddings(self,image,landmarks,strict_alignment,executor)

    def embed(self, image, points, strict_alignment=True):
        import torch
        crop, error = align(image, points, 120, petal=True, strict=strict_alignment)
        x = torch.from_numpy(rgb_tensor(crop)).to(self.device)
        with torch.inference_mode():
            # Official generate_alpha uses normalized RGB and deliberately does not clamp alpha.
            score = self.iqa(x)
            alpha = .5 + score - self.threshold
            output = self.model(x, alpha)
            flipped = torch.flip(x, [3])
            flip_alpha = .5 + self.iqa(flipped) - self.threshold
            output = torch.nn.functional.normalize(output, dim=1) + torch.nn.functional.normalize(self.model(flipped, flip_alpha), dim=1)
        return normalize(output.cpu().numpy()), dict(alignment_error=error, input_size=120,
            alignment_band='regular' if error <= 112*.09 else 'high_residual',
            iqa_score=float(score.item()), alpha=float(alpha.item()), iqa_threshold=self.threshold,
            iqa_threshold_verified=False)
