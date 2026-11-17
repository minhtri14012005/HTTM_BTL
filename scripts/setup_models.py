"""Install official hybrid checkpoints locally; never touches registration data."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models' / 'hybrid'
REVISIONS = {
    'biubug6/Pytorch_Retinaface': 'b984b4b775b2c4dced95c1eadd195a5c7d32a60b',
    'Kartik-3004/PETALface': '2715c33ed6b4ea39e054592bd6a4ead8315ca7fb',
    'bytedance/LVFace': '80bd20c8bb78e56eb708941180d800a27c396a6b',
    'clibdev/yolov5-face': 'd3246815d8747d26a91bf980afef2485bc29f839',
    'chaofengc/IQA-PyTorch': '2cd388c92ccb6f72eeddad07cbb1503a99a7ffbb',
    'bytedance-research/LVFace': 'b12702ab1f5c721748e054a66dc90e1edd1f0724',
    'kartiknarayan/PETALface': '90bb0959328bbc0dfec0de915331b392193cbb3a',
    'chaofengc/IQA-PyTorch-Weights': 'f480f1d8f73dfa374670097cb4098d4a13704165',
    'deepinsight/insightface': 'c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'HTTMV2-Installer'}), timeout=120)


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (not expected or sha(path) == expected):
        return
    temp = path.with_suffix(path.suffix + '.part')
    with request(url) as source, temp.open('wb') as target:
        while chunk := source.read(4 * 1024**2):
            target.write(chunk)
    if expected and sha(temp) != expected:
        raise RuntimeError(f'SHA256 mismatch: {path.name}')
    temp.replace(path)
    print(f'Ready: {path.relative_to(MODELS)}', flush=True)


def source(repo, branch, folder, paths):
    revision = REVISIONS.get(repo) or json.load(request(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
    for path in paths:
        download(f'https://raw.githubusercontent.com/{repo}/{revision}/{path}', MODELS/folder/path)
    return revision


def hf(repo, filename, target):
    revision = REVISIONS.get(repo) or json.load(request(f'https://huggingface.co/api/models/{repo}'))['sha']
    tree = json.load(request(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true'))
    item = next(item for item in tree if item['path'] == filename)
    download(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', MODELS/target, item['lfs']['oid'])
    return revision


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = {'architecture': 'hybrid-v1', 'sources': {}, 'errors': {}, 'files': {}}
    tasks = {
        'lvface': lambda: hf('bytedance-research/LVFace', 'LVFace-B_Glint360K/LVFace-B_Glint360K.onnx', 'lvface.onnx'),
        'petalface': lambda: hf('kartiknarayan/PETALface', 'swin_cosface_webface12m_briar/model.pt', 'petalface.pt'),
        'cnniqa': lambda: hf('chaofengc/IQA-PyTorch-Weights', 'CNNIQA_koniq10k-e6f14c91.pth', 'cnniqa.pth'),
        'antelope': lambda: download('https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip', MODELS/'antelopev2.zip', '8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5'),
        'yolo': lambda: download('https://github.com/clibdev/yolov5-face/releases/download/1.0.0/yolov5m-face.onnx', MODELS/'yolov5m-face.onnx', 'c7ea51072e5f5c1ead34be14b3f4a23f44477448c271bc161b99d122fa0d8f10'),
        'retina_weights': lambda: download('https://drive.usercontent.google.com/download?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW&export=download&confirm=t', MODELS/'Resnet50_Final.pth', '6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d'),
        'retina_source': lambda: source('biubug6/Pytorch_Retinaface', 'master', 'retina', ['LICENSE.MIT', 'models/__init__.py', 'models/net.py', 'models/retinaface.py']),
        'petal_source': lambda: source('Kartik-3004/PETALface', 'main', 'petal', ['LICENSE', 'backbones/__init__.py', 'backbones/iresnet.py', 'backbones/mobilefacenet.py', 'backbones/lora_layers.py', 'backbones/swin_models_iqa.py', 'validation_hq/validate_hq_iqa.py', 'validation_ijb/eval_ijb_iqa.py']),
        'lv_notice': lambda: source('bytedance/LVFace', 'main', 'lvface', ['LICENSE_CODE.txt', 'inference_onnx.py']),
        'insight_notice': lambda: source('deepinsight/insightface', 'master', 'insightface', ['server/backend/insightface_server/licensing/defaults/antelopev2/MODEL.LICENSE']),
        'yolo_notice': lambda: source('clibdev/yolov5-face', 'master', 'yolo', ['LICENSE', 'export.py', 'detect_face.py']),
        'iqa_notice': lambda: source('chaofengc/IQA-PyTorch', 'main', 'iqa', ['LICENSE', 'pyiqa/archs/cnniqa_arch.py']),
    }
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        pending = {pool.submit(task): key for key, task in tasks.items()}
        for future in concurrent.futures.as_completed(pending):
            key = pending[future]
            try:
                manifest['sources'][key] = future.result()
            except Exception as exc:
                manifest['errors'][key] = str(exc)
                print(f'UNAVAILABLE {key}: {exc}', flush=True)
    archive = MODELS/'antelopev2.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as files:
            name = next(name for name in files.namelist() if name.endswith('glintr100.onnx'))
            (MODELS/'antelopev2.onnx').write_bytes(files.read(name))
    for path in MODELS.rglob('*'):
        if path.is_file() and path.name != 'manifest.json' and path.suffix != '.part' and '__pycache__' not in path.parts:
            manifest['files'][path.relative_to(MODELS).as_posix()] = sha(path)
    (MODELS/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('Hybrid installer complete; unavailable artifacts are listed explicitly. Research use only.', flush=True)


if __name__ == '__main__':
    main()
