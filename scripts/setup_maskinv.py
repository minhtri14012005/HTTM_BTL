"""Acquire the author's optional MaskInv-HG checkpoint; no enrollment changes."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'models/candidates/maskinv_hg'
COMMIT='1225983494484d2391ac0595305274d7e1558835'
WEIGHT_SHA='98b9d24341b25c954408da7a2038fec51491284215c69c2a88f689ce5ca5c8e1'
def setup():
    ROOT.mkdir(parents=True,exist_ok=True)
    weight=ROOT/'MaskInvHG_backbone.pth'
    if not weight.exists():
        import gdown
        gdown.download(id='1fYJDn0MD7h5EpXPjWVT-rumzcLQjhqIA',output=str(weight),use_cookies=False,quiet=False,timeout=30)
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if sha(weight)!=WEIGHT_SHA:raise ValueError('Checkpoint MaskInv-HG khác bản đã kiểm tra.')
    source=ROOT/'iresnet.py'
    url=f'https://raw.githubusercontent.com/fdbtrs/Masked-Face-Recognition-KD/{COMMIT}/backbones/iresnet.py'
    raw=urllib.request.urlopen(url,timeout=30).read()
    if source.exists() and source.read_bytes()!=raw:raise ValueError('Source candidate đã thay đổi; không ghi đè.')
    source.write_bytes(raw)
    manifest=dict(name='MaskInv-HG',source='https://github.com/fdbtrs/Masked-Face-Recognition-KD',commit=COMMIT,
        checkpoint='https://drive.google.com/file/d/1fYJDn0MD7h5EpXPjWVT-rumzcLQjhqIA/view',
        license='CC-BY-NC-4.0',license_url='https://creativecommons.org/licenses/by-nc/4.0/',
        copyright='Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt',
        citation='Huber, Boutros, Kirchbuchner, Damer. Mask-invariant Face Recognition through Template-level Knowledge Distillation. FG2021. doi:10.1109/FG52635.2021.9667081',
        changes='Unmodified author architecture and weights; local RGB/ArcFace alignment and flip-sum inference wrapper.',
        files={'MaskInvHG_backbone.pth':WEIGHT_SHA,'iresnet.py':sha(source)})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(installed=True,files=manifest['files'])))
if __name__=='__main__':setup()
