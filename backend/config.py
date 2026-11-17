import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FACEWATCH_DATA', str(ROOT / 'data'))).resolve()
MODELS = ROOT / 'models'
for folder in ('people', 'runtime'):
    (DATA / folder).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(DATA/'runtime/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(DATA/'runtime/matplotlib'))
MODEL_ID = 'hybrid-lvface-antelopev2-petalface-v1'
