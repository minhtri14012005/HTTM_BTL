"""Acquire the optional author ARoFace checkpoint for local shadow evaluation."""
import json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'models/candidates/aroface'
MANIFEST=json.loads('{\n  "name": "AdaFace+ARoFace WebFace12M R100",\n  "source": "https://github.com/msed-Ebrahimi/ARoFace",\n  "commit": "5f87f83b7f02f8400ba04bef25d34def459f3950",\n  "checkpoint": "https://drive.google.com/file/d/1tHqcQBY5s10uxNdfGtIYmYDEKypNtOuQ/view",\n  "checkpoint_license": "Not specified in the author repository; no commercial-use rights asserted.",\n  "architecture_source": "https://github.com/fdbtrs/Masked-Face-Recognition-KD/blob/1225983494484d2391ac0595305274d7e1558835/backbones/iresnet.py",\n  "architecture_license": "CC-BY-NC-4.0",\n  "architecture_copyright": "Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt",\n  "citation": "Ebrahimi Saadabadi et al. ARoFace: Alignment Robustness to Improve Low-Quality Face Recognition. ECCV 2024. arXiv:2407.14972",\n  "changes": "Unmodified pretrained checkpoint. Existing licensed IResNet100 architecture; validated against author architecture. Single forward RGB [-1,1] as official inference.py; no flip fusion.",\n  "files": {\n    "adaface_aroface_webface12m.pt": "4d0805dae2510f5b4e14c3bceb8ee990e71c7dd5fc3122d662b87378018a1ad5",\n    "iresnet.py": "b0abc34b2aa477ae862a1dec85433e1d581824c9af66e63184d5126ef4b1eb7f"\n  }\n}')
def setup():
    ROOT.mkdir(parents=True,exist_ok=True)
    weight=ROOT/'adaface_aroface_webface12m.pt'
    if not weight.exists():
        import gdown
        gdown.download(id='1tHqcQBY5s10uxNdfGtIYmYDEKypNtOuQ',output=str(weight),use_cookies=False,quiet=False,timeout=30)
    architecture=ROOT/'iresnet.py'
    raw=urllib.request.urlopen(MANIFEST['architecture_source'].replace('github.com/','raw.githubusercontent.com/').replace('/blob/','/'),timeout=30).read()
    if architecture.exists() and architecture.read_bytes()!=raw:raise ValueError('Source candidate đã thay đổi; không ghi đè.')
    architecture.write_bytes(raw)
    for name,expected in MANIFEST['files'].items():
        with (ROOT/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Checkpoint/source ARoFace khác bản đã kiểm tra: '+name)
    (ROOT/'manifest.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')
    print(json.dumps(dict(installed=True,files=MANIFEST['files'])))
if __name__=='__main__':setup()
