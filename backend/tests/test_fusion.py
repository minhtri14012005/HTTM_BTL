import threading
import numpy as np
import pytest
from backend.fusion import identity_evidence
from backend.ai import Engine, Pipeline
from backend.store import Store


def vote(pid='a', passed=True, calibrated=True, **extra):
    return dict(person_id=pid,score=.9,passed=passed,calibrated=calibrated,**extra)


def test_unqualified_branch_abstains_and_qualified_conflict_blocks():
    scores=dict(lvface=vote('b',passed=False),antelopev2=vote(),
                petalface=vote(iqa_threshold_verified=True))
    pid,eligible,conflict,info=identity_evidence(scores)
    assert (pid,eligible,conflict)==('a',True,False)
    assert info['model_roles']['lvface']=='below_calibrated_gate'
    scores['lvface']=vote('b')
    assert identity_evidence(scores)[1:3]==(False,True)


def test_single_high_score_uncalibrated_or_invalid_iqa_never_certifies_identity():
    assert not identity_evidence(dict(lvface=vote()))[1]
    scores=dict(lvface=vote(calibrated=False),antelopev2=vote(calibrated=False))
    assert not identity_evidence(scores)[1]
    scores=dict(lvface=vote(),petalface=vote(iqa_threshold_verified=False))
    assert not identity_evidence(scores)[1]
    scores['antelopev2']=vote()
    assert identity_evidence(scores)[1]  # Invalid PETAL IQA does not veto two verified branches.
    scores['lvface']['score']=float('nan')
    assert not identity_evidence(scores)[1]


def test_tiny_faces_are_measured_by_each_model_without_reusing_larger_face_profiles(tmp_path):
    store=Store(tmp_path)
    reference=np.eye(512,dtype=np.float32)[0]
    versions={k:'unit-'+k for k in ('lvface','antelopev2','petalface')}
    store.add('Unit fixture',[(b'unit image',{v:reference for v in versions.values()},{})])
    for v in versions.values():store.calibrate(v,'small',dict(threshold=.2,margin=.01))
    image=np.random.default_rng(3).integers(0,256,(96,96,3),dtype=np.uint8)
    before=image.copy();calls=[]
    class MeasuredEngine:
        lock=threading.RLock()
        status={k:dict(state='ready') for k in versions}
        def load(self):pass
        def detect(self,image):return [(np.array([20,20,44,44,.99]),None,'unit detector')]
        def quality(self,*args,**kwargs):return Engine().quality(*args,**kwargs)
        def embed(self,source,points,key,**kwargs):
            assert np.array_equal(source,before)
            calls.append(key)
            cosine=.1 if key=='lvface' else .9
            v=cosine*reference+np.sqrt(1-cosine**2)*np.eye(512,dtype=np.float32)[1]
            return v,dict(version=versions[key],iqa_threshold=.5)
        def gpu(self):return dict(device='unit')
    engine=MeasuredEngine();engine.versions=versions
    pipe=Pipeline(engine,store)
    for i in range(5):
        result=pipe.run(image,i,i*.2)
        face=result['overlays'][0]
        assert face['quality']['face_pixels']==24
        assert face['quality']['quality_band']=='tiny'
        assert set(face['model_scores'])==set(versions)
        assert not result['events'] and face['name'] is None and face['status']!='confirmed'
        assert all(not s['calibrated'] for s in face['model_scores'].values())
    assert len(calls)==15 and np.array_equal(image,before)
    with pytest.raises(ValueError,match='Mặt quá nhỏ'):
        Engine().quality(image,np.array([20,20,44,44,.99]),store.settings())  # Enrollment remains strict.


def test_high_residual_alignment_cannot_use_clean_alignment_calibration(tmp_path):
    store=Store(tmp_path)
    store.calibrate('unit-model','normal',dict(threshold=.8,margin=.1))
    assert store.calibration('unit-model',100)['threshold']==.8
    assert store.calibration('unit-model',100,alignment_band='high_residual') is None
    assert store.calibration('unit-model',24,alignment_band='high_residual') is None


def test_crowd_single_model_false_match_does_not_block_real_targets(tmp_path):
    """Decision-logic regression with isolated vectors, not a face-accuracy benchmark."""
    store = Store(tmp_path)
    first, second, noise = np.eye(512,dtype=np.float32)[:3]
    versions = {'lvface':'crowd-lv','antelopev2':'crowd-ant'}
    target_a = store.add('Unit A',[(b'unit fixture',{v:first for v in versions.values()},{})])
    target_b = store.add('Unit B',[(b'unit fixture',{v:second for v in versions.values()},{})])
    store.update(target_b,selected=False)
    for version in versions.values():
        store.calibrate(version,'normal',dict(threshold=.8,margin=.08))

    class CrowdEngine:
        lock = threading.RLock()
        status = {key:dict(state='ready') for key in versions}
        def load(self): pass
        def detect(self,image):
            return [(np.array([10+column*180,10+row*150,110+column*180,130+row*150,.99]),
                     row*10+column,'unit detector') for row in range(5) for column in range(10)]
        def quality(self,*args,**kwargs): return dict(face_pixels=100,sharpness=100)
        def embed(self,image,index,key,**kwargs):
            if index < 2:
                return (first if index == 0 else second),{}
            if key == 'lvface':
                reference = first if index % 2 else second
                return .95*reference+np.sqrt(1-.95**2)*noise,{}
            return .1*first+.1*second+np.sqrt(.98)*noise,{}
        def gpu(self): return dict(device='unit')

    engine = CrowdEngine()
    engine.versions = versions
    pipeline = Pipeline(engine,store)
    image = np.zeros((800,1800,3),np.uint8)
    first_phase = [pipeline.run(image,index,index*.2) for index in range(4)]
    assert all(len(result['overlays']) == 50 for result in first_phase)
    assert {event['person_id'] for result in first_phase for event in result['events']} == {target_a}
    assert len({result['overlays'][0]['track_id'] for result in first_phase}) == 1
    for result in first_phase:
        assert all(face['status'] != 'confirmed' for face in result['overlays'][1:])
        assert all(face['name'] is None for face in result['overlays'][2:])
    assert first_phase[-1]['overlays'][0]['status'] == 'confirmed'
    assert first_phase[-1]['overlays'][0]['fusion']['conflicting_face_claims'] == 1

    store.update(target_b,selected=True)
    second_phase = [pipeline.run(image,index,index*.2) for index in range(4,7)]
    assert {event['person_id'] for result in second_phase for event in result['events']} == {target_b}
    final = second_phase[-1]['overlays']
    assert {face['person_id'] for face in final if face['status'] == 'confirmed'} == {target_a,target_b}
    assert all(face['name'] is None for face in final[2:])
    assert len({result['overlays'][1]['track_id'] for result in first_phase+second_phase}) == 1
    assert final[0]['bbox'] == [10,10,110,130]
    assert final[1]['bbox'] == [190,10,290,130]


def test_pipeline_bridges_only_fresh_calibrated_same_identity_on_stable_track(tmp_path):
    store=Store(tmp_path); first,noise=np.eye(512,dtype=np.float32)[:2]
    versions=dict(lvface='lvface:bridge-unit',antelopev2='antelopev2:bridge-unit')
    pid=store.add('Isolated fixture',[(b'unit image',{v:first for v in versions.values()},{})])
    for version in versions.values():store.calibrate(version,'normal',dict(threshold=.8,margin=.08))
    class LocalEngine:
        lock=threading.RLock()
        status={key:dict(state='ready') for key in versions}
        weak=False
        def load(self):pass
        def detect(self,image):return [(np.array([20,20,132,132,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):
            value=.78 if self.weak and key=='lvface' else .9
            return value*first+np.sqrt(1-value**2)*noise,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    engine=LocalEngine();engine.versions=versions;pipe=Pipeline(engine,store)
    image=np.zeros((160,160,3),np.uint8)
    for index in range(3): result=pipe.run(image,index,index*.1)
    assert result['overlays'][0]['status']=='confirmed' and len(result['events'])==1
    engine.weak=True
    result=pipe.run(image,3,.3)
    assert result['overlays'][0]['status']=='confirmed'
    assert result['overlays'][0]['person_id']==pid and result['overlays'][0]['fusion']['temporal_bridge']
    assert len(result['events'])==0
    result=pipe.run(image,4,.8)
    assert result['overlays'][0]['status']!='confirmed' and result['overlays'][0]['person_id'] is None
    assert not result['events']
    # One qualified model cannot start a second track from scratch.
    other=Pipeline(engine,store)
    for index in range(6): result=other.run(image,index,index*.1)
    assert result['overlays'][0]['status']!='confirmed' and not result['events']


def test_new_measurement_discards_preparation_identity_and_event_dedup(tmp_path):
    store=Store(tmp_path);reference=np.eye(512,dtype=np.float32)[0]
    versions=dict(lvface='lvface:prepare-unit',antelopev2='antelopev2:prepare-unit')
    store.add('Fixture',[(b'unit image',{v:reference for v in versions.values()},{})])
    for version in versions.values():store.calibrate(version,'normal',dict(threshold=.8,margin=.08))
    class LocalEngine:
        lock=threading.RLock()
        status={key:dict(state='ready') for key in versions}
        def load(self):pass
        def detect(self,image):return [(np.array([10,10,122,122,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):return reference,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    engine=LocalEngine();engine.versions=versions;pipeline=Pipeline(engine,store);image=np.zeros((140,140,3),np.uint8)
    warm=[pipeline.run(image,index,index*.1) for index in range(3)]
    assert warm[-1]['overlays'][0]['status']=='confirmed'
    track=warm[-1]['overlays'][0]['track_id']
    first=pipeline.run(image,3,.3,reset_temporal=True)
    assert first['measurement_evidence_reset'] and first['overlays'][0]['status']=='candidate' and not first['events']
    assert first['overlays'][0]['track_id']==track
    assert pipeline.run(image,4,.4)['overlays'][0]['status']=='candidate'
    confirmed=pipeline.run(image,5,.5)
    assert confirmed['overlays'][0]['status']=='confirmed' and len(confirmed['events'])==1
    assert confirmed['events'][0]['visit']==1


def test_profile_resolution_uses_native_height_only_for_high_residual_alignment(tmp_path):
    from backend.fusion import quality_band,quality_basis,model_gates
    store=Store(tmp_path)
    config=dict(threshold=.4,margin=.3,verification_enabled=True,quality_basis=quality_basis('high_residual'))
    store.calibrate('unit-pose','normal_high_residual',config)
    assert quality_band(85,'high_residual',150)=='normal_high_residual'
    assert quality_band(85,'regular',150)=='small'
    assert store.calibration('unit-pose',85,'high_residual',150)['threshold']==.4
    assert store.calibration('unit-pose',24,'high_residual',30) is None
    assert model_gates(.55,.2,config)==(False,True)
    assert model_gates(.55,.05,config)==(False,False)
    assert model_gates(.39,.3,config)==(False,False)
    assert model_gates(.9,.5,None)==(True,False)
    assert model_gates(.55,.2,dict(threshold=.4,margin=.3))==(False,False)


def test_reverification_requires_two_current_models_and_cannot_initialize_or_change_identity(tmp_path):
    store=Store(tmp_path);a,b,noise=np.eye(512,dtype=np.float32)[:3]
    versions=dict(lvface='lvface:verify-unit',antelopev2='antelopev2:verify-unit')
    pid=store.add('Fixture A',[(b'unit',{v:a for v in versions.values()},{})])
    store.add('Fixture B',[(b'unit',{v:b for v in versions.values()},{})])
    for v in versions.values():store.calibrate(v,'normal',dict(threshold=.7,margin=.3,verification_enabled=True))
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions}
        weak=False;conflict=False;one_model=False
        def load(self):pass
        def detect(self,image):return [(np.array([20,20,132,132,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):
            x,y=(.75,.55) if self.weak else (.9,.1)
            if self.one_model and key=='lvface':x,y=.6,.55
            if self.conflict and key=='lvface':x,y=.1,.9
            return x*a+y*b+np.sqrt(1-x*x-y*y)*noise,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    engine=LocalEngine();engine.versions=versions;pipe=Pipeline(engine,store);image=np.zeros((160,160,3),np.uint8)
    for i in range(3):result=pipe.run(image,i,i*.1)
    assert result['overlays'][0]['status']=='confirmed'
    engine.weak=True
    result=pipe.run(image,3,1.)
    assert result['overlays'][0]['person_id']==pid and result['overlays'][0]['fusion']['track_reverification']
    assert not result['events']
    fresh=Pipeline(engine,store)
    for i in range(5):result=fresh.run(image,i,i*.1)
    assert result['overlays'][0]['status']!='confirmed' and not result['events']
    engine.one_model=True
    result=pipe.run(image,4,1.2)
    assert result['overlays'][0]['status']!='confirmed' and not result['events']
    engine.one_model=False;engine.conflict=True
    result=pipe.run(image,5,1.3)
    assert result['overlays'][0]['person_id'] is None and not result['events']


def test_detector_rescues_invalid_landmark_order_without_identity_scores():
    from backend.ai import landmarks_order_valid
    engine=Engine();image=np.zeros((200,200,3),np.uint8)
    points=np.array([[65,70],[90,70],[80,95],[65,120],[90,120]],np.float32)
    bad=points.copy();bad[[0,1]]=bad[[1,0]]
    class Detector:
        def __init__(self,values):self.values=values;self.calls=0
        def detect(self,crop):self.calls+=1;assert crop.shape==image.shape;return self.values
    engine.detector=Detector([(np.array([35,40,140,170,.99]),bad,'primary')])
    engine.fallback=Detector([(np.array([36,41,141,171,.92]),points,'fallback')])
    found=engine.detect(image)
    assert engine.fallback.calls==1 and len(found)==1
    assert found[0][2]=='fallback' and landmarks_order_valid(found[0][1])
    assert np.array_equal(found[0][0],np.array([36,41,141,171,.92]))
    engine.detector=Detector([(np.array([35,40,140,170,.99]),points,'primary')])
    engine.fallback.calls=0
    assert engine.detect(image)[0][2]=='primary' and engine.fallback.calls==0



def test_weak_pose_does_not_create_a_second_visit_on_continuously_visible_track():
    from backend.temporal import Temporal
    state=Temporal();settings=dict(confirm_frames=3,absence_seconds=3.)
    for index in range(3):status,emit,visit=state.update(1,'a',index*.1,settings,frame_id=index)
    assert (status,emit,visit)==('confirmed',True,1)
    for index in range(3,50):
        status,emit,_=state.update(1,None,index*.1,settings,frame_id=index)
        assert status=='unknown' and not emit
        state.finish_frame(index*.1)
    for index in range(50,53):status,emit,visit=state.update(1,'a',index*.1,settings,frame_id=index)
    assert (status,emit,visit)==('confirmed',False,1)
    # Loss of the actual face still permits a genuine re-entry notification.
    state.finish_frame(9.)
    for index in range(3):status,emit,visit=state.update(2,'a',9.+index*.1,settings,frame_id=60+index)
    assert (status,emit,visit)==('confirmed',True,2)


def test_adapter_override_is_shadow_by_default_and_can_be_disabled_or_replaced(tmp_path):
    from backend.fusion import identity_evidence,verification_evidence
    store=Store(tmp_path);a,b=np.eye(512,dtype=np.float32)[:2]
    versions=dict(lvface='lvface:adapter-unit',antelopev2='antelopev2:adapter-unit')
    extra='specialist:adapter-unit'
    pid=store.add('A',[(b'unit',{**{v:a for v in versions.values()},extra:a},{})])
    store.add('B',[(b'unit',{**{v:b for v in versions.values()},extra:b},{})])
    for version in [*versions.values(),extra]:store.calibrate(version,'normal',dict(threshold=.8,margin=.08,verification_enabled=True))
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions}
        def load(self):pass
        def detect(self,image):return [(np.array([10,10,122,122,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):return a,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    class Specialist:
        version=extra
        def embed_many(self,image,landmarks,**kwargs):return [(b,dict(version=extra,alignment_band='regular')) for _ in landmarks]
    engine=LocalEngine();engine.versions=versions;image=np.zeros((140,140,3),np.uint8)
    base=Pipeline(engine,store);shadow=Pipeline(engine,store,model_overrides={'specialist':Specialist()})
    for index in range(4):
        original=base.run(image,index,index*.1);probe=shadow.run(image,index,index*.1)
        assert [o['person_id'] for o in original['overlays']]==[o['person_id'] for o in probe['overlays']]
        assert [o['track_id'] for o in original['overlays']]==[o['track_id'] for o in probe['overlays']]
        assert [e['person_id'] for e in original['events']]==[e['person_id'] for e in probe['events']]
    face=probe['overlays'][0]
    assert face['person_id']==pid and face['model_scores']['specialist']['shadow'] is True
    assert face['fusion']['model_roles']['specialist']=='shadow_only'
    assert probe['shadow_models']==['specialist']
    disabled=Pipeline(engine,store,disabled_models=('antelopev2',))
    for index in range(4):result=disabled.run(image,index,index*.1)
    assert result['overlays'][0]['person_id'] is None and not result['events']
    assert 'antelopev2' not in result['overlays'][0]['model_scores']
    # Explicit replacement still uses the same calibrated fusion and conflict rejection.
    replacement=Pipeline(engine,store,model_overrides={'antelopev2':Specialist()},shadow_models=())
    for index in range(4):result=replacement.run(image,index,index*.1)
    assert result['overlays'][0]['person_id'] is None and not result['events']
    only_shadow={'specialist':dict(shadow=True,person_id=pid,score=.999,margin=.9,passed=True,verification_passed=True,calibrated=True)}
    assert identity_evidence(only_shadow)[1] is False
    assert verification_evidence(only_shadow)[1] is False


def test_verification_only_other_identity_cannot_veto_two_identification_models(tmp_path):
    from backend.fusion import verification_evidence
    store=Store(tmp_path);a,b,noise=np.eye(512,dtype=np.float32)[:3]
    versions=dict(lvface='lvface:context-unit',antelopev2='antelopev2:context-unit');extra='specialist:context-unit'
    pid=store.add('A',[(b'unit',{**{v:a for v in versions.values()},extra:a},{})])
    store.add('B',[(b'unit',{**{v:b for v in versions.values()},extra:b},{})])
    for version in versions.values():store.calibrate(version,'normal',dict(threshold=.8,margin=.2,verification_enabled=True))
    store.calibrate(extra,'normal',dict(threshold=.8,margin=.5,verification_enabled=True))
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions}
        def load(self):pass
        def detect(self,image):return [(np.array([10,10,122,122,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):return .95*a+.05*b+np.sqrt(.095)*noise,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    class Specialist:
        version=extra
        strong_conflict=False
        def embed_many(self,image,landmarks,**kwargs):
            vector=.85*b+.45*a+np.sqrt(.075)*noise if not self.strong_conflict else b
            return [(vector,dict(version=extra,alignment_band='regular')) for _ in landmarks]
    engine=LocalEngine();engine.versions=versions;expert=Specialist()
    pipe=Pipeline(engine,store,model_overrides={'specialist':expert},shadow_models=())
    image=np.zeros((140,140,3),np.uint8)
    for index in range(8):
        result=pipe.run(image,index,index*.1)
        if index>=2:assert result['overlays'][0]['person_id']==pid
    wrong=result['overlays'][0]['model_scores']['specialist']
    assert wrong['verification_passed'] and not wrong['passed']
    assert result['overlays'][0]['fusion']['verification_roles']['specialist']=='abstain'
    expert.strong_conflict=True
    result=pipe.run(image,8,.8)
    assert result['overlays'][0]['person_id'] is None and not result['events']
    only_verification={key:dict(person_id=pid,score=.85,calibrated=True,passed=False,verification_passed=True) for key in versions}
    assert verification_evidence(only_verification)[1] is False
    assert verification_evidence(only_verification,pid)[1] is True


def test_continuous_track_reacquires_with_fresh_pair_after_vote_expiry_and_resets_on_conflict(tmp_path):
    store=Store(tmp_path);a,b,noise=np.eye(512,dtype=np.float32)[:3]
    versions=dict(lvface='lvface:anchor-unit',antelopev2='antelopev2:anchor-unit')
    pid=store.add('A',[(b'unit',{v:a for v in versions.values()},{})]);store.add('B',[(b'unit',{v:b for v in versions.values()},{})])
    for v in versions.values():store.calibrate(v,'normal',dict(threshold=.7,margin=.3,verification_enabled=True))
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions};phase='initial';one=False;conflict=False
        def load(self):pass
        def detect(self,image):return [(np.array([10,10,122,122,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):
            x,y={'initial':(.9,.1),'weak':(.6,.4),'verify':(.75,.55)}[self.phase]
            if self.one and key=='antelopev2':x,y=.6,.4
            if self.conflict and key=='antelopev2':x,y=.1,.9
            return x*a+y*b+np.sqrt(1-x*x-y*y)*noise,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    engine=LocalEngine();engine.versions=versions;pipe=Pipeline(engine,store);image=np.zeros((140,140,3),np.uint8)
    for i in range(3):result=pipe.run(image,i,i*.1)
    track=result['overlays'][0]['track_id'];assert result['overlays'][0]['person_id']==pid
    engine.phase='weak'
    for i in range(3,60):
        result=pipe.run(image,i,i*.1)
        assert result['overlays'][0]['person_id'] is None and not result['events']
    assert track not in pipe.temporal.tracks and pipe.temporal.can_reverify(track,pid,6.)
    engine.phase='verify'
    result=pipe.run(image,60,6.)
    assert result['overlays'][0]['person_id']==pid and pipe.temporal.tracks[track]['hits']==1
    assert result['overlays'][0]['fusion']['track_resumed'] and not result['events']
    for i in range(61,63):result=pipe.run(image,i,i*.1)
    assert result['overlays'][0]['person_id']==pid and result['overlays'][0]['fusion']['track_reverification']
    assert not result['events']
    engine.one=True
    for i in range(63,84):
        result=pipe.run(image,i,i*.1)
        if i>=69:assert result['overlays'][0]['person_id'] is None and not result['events']
    engine.one=False;engine.conflict=True
    result=pipe.run(image,84,8.4)
    assert result['overlays'][0]['person_id'] is None and not pipe.temporal.can_reverify(track,pid,8.4)
    engine.conflict=False
    for i in range(85,90):result=pipe.run(image,i,i*.1)
    assert result['overlays'][0]['person_id'] is None
    pipe.temporal.reset_identity_evidence();assert not pipe.temporal.identity_anchors


def test_expert_quality_routing_abstains_without_changing_detection_or_source_resolution(tmp_path):
    store=Store(tmp_path);a=np.eye(512,dtype=np.float32)[0];versions=dict(lvface='lvface:route-unit',antelopev2='antelopev2:route-unit')
    pid=store.add('A',[(b'unit',{**{v:a for v in versions.values()},'specialist:route-unit':a},{})])
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions};alignment='high_residual';height=145
        def load(self):pass
        def detect(self,image):return [(np.array([10,10,95,10+self.height,.99]),None,'unit')]
        def quality(self,*args,**kwargs):return dict(face_pixels=85,original_face_size=[85,self.height],sharpness=50)
        def embed(self,image,points,key,**kwargs):return a,dict(version=versions[key],alignment_band=self.alignment)
        def gpu(self):return dict(device='unit')
    class Expert:
        version='specialist:route-unit';allowed_bands={'normal_high_residual'};calls=0
        def embed_many(self,image,points,**kwargs):
            self.calls+=1;assert image.shape==(180,240,3)
            return [(a,dict(version=self.version,alignment_band='high_residual')) for _ in points]
    engine=LocalEngine();engine.versions=versions;expert=Expert();image=np.zeros((180,240,3),np.uint8);before=image.copy()
    pipe=Pipeline(engine,store,model_overrides={'specialist':expert})
    first=pipe.run(image,1,0.);assert expert.calls==1 and first['width']==240 and first['height']==180
    engine.alignment='regular';second=pipe.run(image,2,.1,diagnostic_details=True)
    engine.alignment='high_residual';engine.height=40;third=pipe.run(image,3,.2)
    assert expert.calls==1 and len(second['overlays'])==len(third['overlays'])==1
    assert 'specialist' not in second['overlays'][0]['model_scores'] and 'lvface' in second['overlays'][0]['model_scores']
    assert second['overlays'][0]['model_routing']['specialist']['reason']=='outside_validated_quality_band'
    assert not second['overlays'][0]['model_routing']['specialist']['inference_performed']
    assert np.array_equal(image,before)


def test_resume_needs_existing_anchor_fresh_pair_and_uninterrupted_observation():
    from backend.temporal import Temporal,stable_association
    settings=dict(confirm_frames=3,absence_seconds=3.)
    def confirmed():
        state=Temporal()
        for i in range(3):state.update(1,'a',i*.1,settings,frame_id=i)
        for i in range(3,24):state.update(1,None,i*.1,settings,frame_id=i)
        return state
    state=confirmed()
    assert state.update(1,'a',2.4,settings,reverified=True,fresh_model_count=2)==('confirmed',False,1)
    assert state.tracks[1]['hits']==1  # No invented repeated observations.
    for kwargs in (dict(fresh_model_count=1),dict(multimodel=False,fresh_model_count=2),dict(reset=True,fresh_model_count=2)):
        state=confirmed()
        assert state.update(1,'a',2.4,settings,reverified=True,**kwargs)[0]=='candidate'
    state=confirmed()
    assert state.update(1,'b',2.4,settings,reverified=True,fresh_model_count=2)[0]=='candidate'
    state=confirmed()
    assert state.update(1,'a',4.,settings,reverified=True,fresh_model_count=2)[0]=='candidate'
    fresh=Temporal()
    assert fresh.update(1,'a',0.,settings,reverified=True,fresh_model_count=4)[0]=='candidate'
    link=dict(method='overlap',candidate_count=1,previous_iou=.9,predicted_iou=.8,appearance_similarity=.9,appearance_age_seconds=.2)
    assert stable_association(link)
    for key,value in [('candidate_count',2),('method','geometry'),('identity_evidence_reset',True),('appearance_similarity',None),('appearance_age_seconds',2.)]:
        assert not stable_association(dict(link,**{key:value}))


def test_tracking_preserves_only_an_existing_unbroken_label_without_votes_or_alerts():
    from backend.temporal import Temporal
    from backend.identity_tracking import trusted_identity_link
    settings=dict(confirm_frames=3,absence_seconds=3.)
    state=Temporal()
    for i in range(3):state.update(1,'a',i*.1,settings,frame_id=i)
    for i in range(3,50):
        assert state.update(1,None,i*.1,settings,continuity=True)==('tracked',False,1)
    assert not state.tracks and state.identity_anchors[1]=='a'  # No biometric refresh.
    state.update(1,None,5.,settings,continuity=False)
    assert state.update(1,None,5.1,settings,continuity=True)[0]=='unknown'
    for i in range(52,55):state.update(1,'a',i*.1,settings)
    assert state.update(1,None,5.5,settings,reset=True,continuity=True)[0]=='unknown'
    assert Temporal().update(1,None,0.,settings,continuity=True)[0]=='unknown'
    face=dict(track_id=1,bbox=[0,0,100,100],model_scores=dict(lvface=dict(version='v',score=.2)),
        tracking=dict(method='overlap',candidate_count=1,previous_iou=.9,predicted_iou=.9,appearance_similarity=.9,appearance_age_seconds=.2))
    profile=dict(enabled=True,model='lvface',version='v',threshold=.8)
    assert trusted_identity_link(face,[face],[face],profile)
    assert not trusted_identity_link(face,[face],[face],dict(profile,version='other'))
    assert not trusted_identity_link(face,[],[face],profile)
    assert not trusted_identity_link(face,[face],[face,dict(face,track_id=2)],profile)
    assert not trusted_identity_link(dict(face,tracking=dict(face['tracking'],appearance_similarity=.7)),[face],[face],profile)
    assert not trusted_identity_link(dict(face,model_scores={}),[face],[face],profile)
    assert not trusted_identity_link(face,[face],[face],None)


def test_pipeline_tracking_names_only_a_confirmed_identity_and_conflict_clears_it(tmp_path):
    store=Store(tmp_path);a,b,noise=np.eye(512,dtype=np.float32)[:3]
    versions=dict(lvface='lvface:continuity-unit',antelopev2='antelopev2:continuity-unit')
    pid=store.add('A',[(b'unit',{v:a for v in versions.values()},{})]);other=store.add('B',[(b'unit',{v:b for v in versions.values()},{})])
    for version in versions.values():store.calibrate(version,'normal',dict(threshold=.7,margin=.3,verification_enabled=True))
    store.save_settings(dict(identity_tracking_profile=dict(enabled=True,model='lvface',version=versions['lvface'],threshold=.8)))
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions};weak=False;conflict=False;visible=True
        def load(self):pass
        def detect(self,image):return [(np.array([10,10,122,122,.99]),None,'unit')] if self.visible else []
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):
            x,y=(.69,.1) if self.weak else (.9,.1)
            if self.conflict and key=='antelopev2':x,y=.1,.9
            return x*a+y*b+np.sqrt(1-x*x-y*y)*noise,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    engine=LocalEngine();engine.versions=versions;pipe=Pipeline(engine,store);image=np.zeros((140,140,3),np.uint8)
    for i in range(3):result=pipe.run(image,i,i*.1)
    assert result['overlays'][0]['status']=='confirmed'
    engine.weak=True
    for i in range(3,30):
        result=pipe.run(image,i,i*.1);face=result['overlays'][0]
        assert face['status']=='tracked' and face['person_id']==pid and face['name']=='A'
        assert face['identity_source']=='track_continuity' and not result['events']
    engine.conflict=True
    result=pipe.run(image,30,3.);assert result['overlays'][0]['person_id'] is None and not result['events']
    engine.conflict=False
    result=pipe.run(image,31,3.1);assert result['overlays'][0]['person_id'] is None
    engine.weak=False
    for i in range(32,35):result=pipe.run(image,i,i*.1)
    engine.visible=False;assert not pipe.run(image,35,3.5)['overlays']
    engine.visible=True;engine.weak=True
    assert pipe.run(image,36,3.6)['overlays'][0]['person_id'] is None


def test_identity_continuity_uses_calibrated_appearance_not_the_biometric_resume_overlap_gate():
    from backend.identity_tracking import trusted_identity_link
    face=dict(track_id=1,bbox=[10,10,45,65],model_scores=dict(lvface=dict(version='v',score=.2)),
        tracking=dict(method='overlap',candidate_count=1,previous_iou=.46,predicted_iou=.33,
                      appearance_similarity=.65,appearance_age_seconds=.25))
    profile=dict(enabled=True,model='lvface',version='v',threshold=.544)
    assert trusted_identity_link(face,[face],[face],profile)
    def changed(**values):return dict(face,tracking=dict(face['tracking'],**values))
    assert not trusted_identity_link(changed(appearance_similarity=.54),[face],[face],profile)
    assert not trusted_identity_link(changed(candidate_count=2),[face],[face],profile)
    assert not trusted_identity_link(changed(identity_evidence_reset=True),[face],[face],profile)
    assert trusted_identity_link(changed(method='motion',previous_iou=.2,predicted_iou=.6),[face],[face],profile)
    assert trusted_identity_link(changed(method='motion',previous_iou=.2,predicted_iou=.4),[face],[face],profile)
    assert not trusted_identity_link(changed(method='motion',previous_iou=.2,predicted_iou=.25),[face],[face],profile)
    assert not trusted_identity_link(changed(method='appearance'),[face],[face],profile)
    assert not trusted_identity_link(face,[face],[face,dict(face,track_id=2)],profile)


def test_a_fresh_identity_claim_on_another_face_blocks_a_weak_tracked_name(tmp_path):
    store=Store(tmp_path);a,b,noise=np.eye(512,dtype=np.float32)[:3]
    versions=dict(lvface='lvface:duplicate-continuity',antelopev2='antelopev2:duplicate-continuity')
    person=store.add('A',[(b'unit',{v:a for v in versions.values()},{})])
    store.add('B',[(b'unit',{v:b for v in versions.values()},{})])
    for version in versions.values():store.calibrate(version,'normal',dict(threshold=.7,margin=.3,verification_enabled=True))
    store.save_settings(dict(identity_tracking_profile=dict(enabled=True,model='lvface',version=versions['lvface'],threshold=.8)))
    class LocalEngine:
        lock=threading.RLock();status={key:dict(state='ready') for key in versions};two=False
        def load(self):pass
        def detect(self,image):
            faces=[(np.array([10,10,122,122,.99]),np.zeros((5,2),np.float32),'unit')]
            if self.two:faces.append((np.array([260,10,372,122,.99]),np.ones((5,2),np.float32),'unit'))
            return faces
        def quality(self,*args,**kwargs):return dict(face_pixels=112,sharpness=50)
        def embed(self,image,points,key,**kwargs):
            x=.69 if self.two and points[0,0]==0 else .9;y=.1
            return x*a+y*b+np.sqrt(1-x*x-y*y)*noise,dict(version=versions[key],alignment_band='regular')
        def gpu(self):return dict(device='unit')
    engine=LocalEngine();engine.versions=versions;pipe=Pipeline(engine,store);image=np.zeros((150,400,3),np.uint8)
    for i in range(3):result=pipe.run(image,i,i*.1)
    assert result['overlays'][0]['person_id']==person
    engine.two=True;result=pipe.run(image,3,.3)
    assert len(result['overlays'])==2 and not result['events']
    assert result['overlays'][0]['person_id'] is None
    assert result['overlays'][1]['status']=='candidate'


def test_motion_continuity_preserves_an_anchor_without_fresh_votes_or_alerts():
    from backend.identity_tracking import trusted_identity_link
    from backend.temporal import Temporal
    settings=dict(confirm_frames=3,absence_seconds=3.)
    profile=dict(enabled=True,model='lvface',version='unit',threshold=.544)
    face=dict(track_id=1,bbox=[10,10,54,70],model_scores={'lvface':dict(version='unit',score=.31)},
              tracking=dict(method='motion',previous_iou=.075,predicted_iou=.437,
                            appearance_similarity=.787,appearance_age_seconds=.267,candidate_count=1))
    continuity=trusted_identity_link(face,[face],[face],profile)
    assert continuity
    state=Temporal()
    for i in range(3):state.update(1,'a',i*.2,settings,frame_id=i)
    assert state.update(1,None,.667,settings,continuity=continuity)==('tracked',False,1)
    assert Temporal().update(1,None,.667,settings,continuity=continuity)[0]=='unknown'
    assert not trusted_identity_link(dict(face,tracking=dict(face['tracking'],appearance_similarity=.54)),[face],[face],profile)
    assert not trusted_identity_link(face,[face],[face,dict(face,track_id=2)],profile)
    assert not trusted_identity_link(dict(face,tracking=dict(face['tracking'],identity_evidence_reset=True)),[face],[face],profile)



def test_same_recognizer_variants_cannot_supply_independent_identity_evidence():
    from backend.fusion import verification_evidence
    scores = dict(lvface=vote(), lvface_eye=vote(recognizer_family='lvface'))
    for evidence in (identity_evidence, verification_evidence):
        pid, eligible, conflict, info = evidence(scores)
        assert (pid, eligible, conflict) == ('a', False, False)
        assert info['calibrated_support'] == 1 and info['qualified_branches'] == 2
        scores['antelopev2'] = vote(passed=False)
        assert not evidence(scores)[1]
        scores['antelopev2'] = vote()
        assert evidence(scores)[1:3] == (True, False)
        scores['lvface_eye'] = vote('b', recognizer_family='lvface')
        assert evidence(scores)[1:3] == (False, True)
        scores['lvface_eye'] = vote('b', passed=False, recognizer_family='lvface')
        assert evidence(scores)[1:3] == (True, False)
        scores['lvface_eye'] = vote(recognizer_family='lvface')
        del scores['antelopev2']


def test_alignment_fallback_only_measures_faces_without_native_calibrated_evidence(tmp_path):
    store = Store(tmp_path)
    reference, other = np.eye(512, dtype=np.float32)[:2]
    versions = dict(lvface='native-lv', antelopev2='native-ant')
    store.add('Unit fixture', [(b'unit fixture', {v: reference for v in (*versions.values(), 'eye-unit')}, {})])
    for version in (*versions.values(), 'eye-unit'):
        store.calibrate(version, 'normal', dict(threshold=.8, margin=.1))
    class LocalEngine:
        lock = threading.RLock()
        status = {key: dict(state='ready') for key in versions}
        def load(self): pass
        def detect(self, image):
            return [(np.array([10+index*140,10,110+index*140,120,.99]),index,'unit detector') for index in range(2)]
        def quality(self, *args, **kwargs): return dict(face_pixels=100, sharpness=100)
        def embed(self, image, index, key, **kwargs):
            return (reference if index==0 else other), dict(version=versions[key],alignment_band='regular' if index==0 else 'high_residual')
        def gpu(self): return dict(device='unit')
    class EyeAdapter:
        version = 'eye-unit'
        allowed_bands = {'normal'}
        routing_alignment = 'regular'
        fallback_for = 'lvface'
        shadow_only = True
        def embed_many(self, image, landmarks, **kwargs):
            self.measured = landmarks
            return [(reference,dict(version=self.version,recognizer_family='lvface')) for _ in landmarks]
    engine = LocalEngine()
    engine.versions = versions
    adapter = EyeAdapter()
    pipeline = Pipeline(engine, store, model_overrides={'lvface_eye': adapter}, shadow_models=[])
    result = pipeline.run(np.zeros((180,320,3),np.uint8),1,0.,diagnostic_details=True)
    assert adapter.measured == [1]
    clear, weak = result['overlays']
    assert 'lvface_eye' not in clear['model_scores']
    assert clear['model_routing']['lvface_eye']['reason'] == 'native_model_has_calibrated_evidence'
    assert weak['model_scores']['lvface_eye']['shadow'] is True
    assert weak['fusion']['calibrated_support'] == 0
    assert not result['events']
