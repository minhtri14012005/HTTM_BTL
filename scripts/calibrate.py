"""Calibrate each model on LOCAL labeled open-set gallery/probes, without training.
Input JSON: {gallery:[{person_id,images:[path]}], probes:[{image,person_id|null,
 split:'calibration'|'validation'}]}. All paths are relative to the input file.
Print measured held-out results; --apply stores only independently validated profiles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.ai import engine
from backend.store import Store


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--max-far',type=float,default=0.)
    args=parser.parse_args()
    if not 0<=args.max_far<1:raise ValueError('max-far phải thuộc [0,1).')
    dataset=json.loads(args.dataset.read_text(encoding='utf8'))
    store=Store();engine.load();cache={};split_hashes={'gallery':set(),'calibration':set(),'validation':set()}
    def embed(path,split):
        path=(args.dataset.parent/path).resolve()
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();split_hashes[split].add(digest)
        if digest not in cache:cache[digest]=engine.register(raw,store.settings())
        return cache[digest]
    galleries={}
    for person in dataset['gallery']:
        for image in person['images']:
            _,vectors,_=embed(image,'gallery')
            for version,vector in vectors.items():galleries.setdefault(version,[]).append((person['person_id'],vector))
    measurements={}
    for probe in dataset['probes']:
        split=probe['split']
        if split not in ('calibration','validation'):raise ValueError('Thiếu split calibration/validation.')
        _,vectors,quality=embed(probe['image'],split)
        band='small' if quality['face_pixels']<90 else 'normal'
        for version,vector in vectors.items():
            scores={}
            for pid,reference in galleries.get(version,[]):scores[pid]=max(scores.get(pid,-1.),float(np.dot(vector,reference)))
            ranking=sorted(scores.items(),key=lambda v:v[1],reverse=True)
            if not ranking:continue
            pid,score=ranking[0];margin=score-ranking[1][1] if len(ranking)>1 else 2.
            measurements.setdefault((version,band),[]).append(dict(score=score,margin=margin,correct=pid==probe['person_id'],known=probe['person_id'] is not None,split=split))
    if any(split_hashes[a]&split_hashes[b] for a,b in [('gallery','calibration'),('gallery','validation'),('calibration','validation')]):
        raise ValueError('Ảnh trùng giữa gallery/calibration/validation; không được hiệu chuẩn trên chính ảnh đăng ký.')
    for (version,band),rows in measurements.items():
        train=[r for r in rows if r['split']=='calibration'];valid=[r for r in rows if r['split']=='validation']
        if any(sum(not r['known'] for r in group)<30 or not any(r['correct'] for r in group) for group in (train,valid)):
            print(version,band,'CHƯA ÁP DỤNG: cần ít nhất 30 probe Unknown và probe đúng trong mỗi split.');continue
        candidates=[]
        for threshold in sorted({max(.1,r['score']+1e-6) for r in train}|{.5,.6,.7,.8,.9}):
            for margin in sorted({max(.01,r['margin']+1e-6) for r in train if r['margin']<1}|{.04,.08,.12}):
                accepted=lambda r:r['score']>=threshold and r['margin']>=margin
                false=sum(accepted(r) and not r['correct'] for r in train)/sum(not r['correct'] for r in train)
                recall=sum(accepted(r) and r['correct'] for r in train)/sum(r['known'] for r in train)
                if false<=args.max_far:candidates.append((recall,threshold,margin))
        if not candidates:print(version,band,'Không có ngưỡng đạt ràng buộc.');continue
        _,threshold,margin=max(candidates)
        accepted=lambda r:r['score']>=threshold and r['margin']>=margin
        false=sum(accepted(r) and not r['correct'] for r in valid)/sum(not r['correct'] for r in valid)
        recall=sum(accepted(r) and r['correct'] for r in valid)/sum(r['known'] for r in valid)
        config=dict(threshold=threshold,margin=margin,heldout_false_rate=false,heldout_recall=recall,
                    iqa_threshold=.5 if version.startswith('petalface:') else None,
                    calibration_count=len(train),validation_count=len(valid),dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    quality_band=band,verification='Empirical held-out probes; not a classroom accuracy guarantee.')
        print(json.dumps(dict(model=version,**config),ensure_ascii=False))
        if args.apply and false<=args.max_far and recall>0:store.calibrate(version,band,config)


if __name__=='__main__':main()
