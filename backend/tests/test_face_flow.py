"""Optical-flow checks on real library image pixels in an isolated test."""
import cv2
import numpy as np
import pytest
from skimage import data
from backend.face_flow import FaceFlow, overlap


def sequence():
    original=cv2.cvtColor(data.astronaut(),cv2.COLOR_RGB2BGR)
    return [cv2.warpAffine(original,np.float32([[1,0,index*2],[0,1,index]]),(512,512)) for index in range(5)]


def result(index,status='confirmed'):
    box=[170+index*2,60+index,282+index*2,172+index]
    return dict(frame_id=index+1,width=512,height=512,media_timestamp=index*.04,captured_at=100+index*40,
        overlays=[dict(track_id=4,face_bbox=box,bbox=box,status=status,
                       person_id='unit' if status=='confirmed' else None,
                       name='Unit' if status=='confirmed' else None)])


@pytest.mark.parametrize('confirmed',[True,False])
def test_intermediate_boxes_follow_current_pixels_and_identity_requires_both_anchors(confirmed):
    images=sequence()
    flow=FaceFlow()
    assert flow.between(images[0],result(0),[]) == []
    frames=[(dict(frame_id=index+1,timestamp=index*.04,captured_at=100+index*40),images[index]) for index in range(1,4)]
    tracked=flow.between(images[4],result(4,'confirmed' if confirmed else 'candidate'),frames)
    assert len(tracked)==3
    for index,frame in enumerate(tracked,start=1):
        assert frame['frame_id']==index+1 and frame['media_timestamp']==index*.04
        assert frame['captured_at']==100+index*40 and frame['overlay_origin']=='optical_flow'
        assert len(frame['overlays'])==1
        face=frame['overlays'][0]
        assert overlap(face['face_bbox'],result(index)['overlays'][0]['face_bbox']) > .95
        assert face['detection_frame_id']==5 and face['flow_confidence'] >= .6
        assert (face['status']=='tracked')==confirmed
        assert face['status']!='confirmed'
        assert (face['identity_source']=='track_continuity')==confirmed
        assert (face['person_id']=='unit')==confirmed
        assert frame['events']==[] and 'model_scores' not in face


def test_missing_texture_never_creates_a_face_box_or_identity():
    blank=np.full((512,512,3),120,np.uint8)
    flow=FaceFlow();flow.between(blank,result(0),[])
    frames=[(dict(frame_id=2,timestamp=.04,captured_at=140),blank)]
    tracked=flow.between(blank,result(2),frames)
    assert len(tracked)==1 and tracked[0]['overlays']==[]


def test_occlusion_and_orientation_change_discard_tracking():
    images=sequence()
    flow=FaceFlow();flow.between(images[0],result(0),[])
    hidden=np.zeros_like(images[1])
    tracked=flow.between(images[4],result(4),[(dict(frame_id=2,timestamp=.04,captured_at=140),hidden)])
    assert tracked[0]['overlays']==[]
    rotated=np.zeros((256,512,3),np.uint8)
    changed=dict(result(5),height=256)
    assert flow.between(rotated,changed,[])==[]


def test_tracked_identity_is_preserved_as_tracked_and_never_emits_or_becomes_new_confirmation():
    images=sequence();flow=FaceFlow();first=result(0);flow.between(images[0],first,[])
    last=result(4);last['overlays'][0].update(status='tracked',identity_source='track_continuity')
    frames=[(dict(frame_id=i+1,timestamp=i*.04,captured_at=100+i*40),images[i]) for i in range(1,4)]
    output=flow.between(images[4],last,frames)
    assert output and all(not f['events'] for f in output)
    assert all(face['status']=='tracked' and face['person_id']=='unit' and face['identity_source']=='track_continuity' for f in output for face in f['overlays'])


def test_forward_path_recovers_real_pixels_when_current_corner_reseed_fails(monkeypatch):
    import backend.face_flow as module
    images=sequence();flow=FaceFlow();flow.between(images[0],result(0),[])
    original=module.feature_states;last_gray=cv2.cvtColor(images[4],cv2.COLOR_BGR2GRAY)
    def no_last_reseed(gray,faces):
        return [] if np.array_equal(gray,last_gray) else original(gray,faces)
    monkeypatch.setattr(module,'feature_states',no_last_reseed)
    frames=[(dict(frame_id=i+1,timestamp=i*.04,captured_at=100+i*40),images[i]) for i in range(1,4)]
    last=result(4);output=flow.between(images[4],last,frames)
    assert last['visual_tracking_summary']['forward_rescued_boxes']==3
    for i,frame in enumerate(output,start=1):
        assert frame['frame_id']==i+1 and frame['captured_at']==100+i*40
        assert len(frame['overlays'])==1
        face=frame['overlays'][0]
        assert overlap(face['face_bbox'],result(i)['overlays'][0]['face_bbox'])>.95
        assert face['flow_direction']=='forward' and face['status']=='tracked'
        assert face['identity_source']=='track_continuity' and not frame['events']


def test_forward_path_cannot_recover_a_box_without_a_matching_current_detection(monkeypatch):
    import backend.face_flow as module
    images=sequence();flow=FaceFlow();flow.between(images[0],result(0),[])
    original=module.feature_states;last_gray=cv2.cvtColor(images[4],cv2.COLOR_BGR2GRAY)
    monkeypatch.setattr(module,'feature_states',lambda gray,faces: [] if np.array_equal(gray,last_gray) else original(gray,faces))
    last=result(4);box=[350,300,462,412];last['overlays'][0].update(face_bbox=box,bbox=box)
    frames=[(dict(frame_id=i+1,timestamp=i*.04,captured_at=100+i*40),images[i]) for i in range(1,4)]
    output=flow.between(images[4],last,frames)
    assert all(not frame['overlays'] and not frame['events'] for frame in output)


def test_visual_tracking_abstains_on_crossing_regions_and_identity_conflicts():
    images=sequence();flow=FaceFlow()
    first=result(0);first['overlays'].append(dict(first['overlays'][0],track_id=5,person_id='other',name='Other'))
    flow.between(images[0],first,[])
    last=result(4);last['overlays'].append(dict(last['overlays'][0],track_id=5,person_id='other',name='Other'))
    frames=[(dict(frame_id=i+1,timestamp=i*.04,captured_at=100+i*40),images[i]) for i in range(1,4)]
    output=flow.between(images[4],last,frames)
    assert all(o['person_id'] is None and o['status']=='unknown' for f in output for o in f['overlays'])
    assert all(not f['events'] for f in output)
    flow=FaceFlow();flow.between(images[0],result(0),[])
    last=result(4);last['overlays'][0]['person_id']='other'
    output=flow.between(images[4],last,frames)
    assert all(o['person_id'] is None for f in output for o in f['overlays'])


def test_forward_rescue_rejects_low_original_feature_survival(monkeypatch):
    import backend.face_flow as module
    images=sequence();flow=FaceFlow();flow.between(images[0],result(0),[])
    original_seed=module.feature_states;original_move=module.move_states
    last_gray=cv2.cvtColor(images[4],cv2.COLOR_BGR2GRAY)
    monkeypatch.setattr(module,'feature_states',lambda gray,faces: [] if np.array_equal(gray,last_gray) else original_seed(gray,faces))
    def weak_survival(before,after,states):
        output=original_move(before,after,states)
        for state in output:state['cumulative_confidence']=.25
        return output
    monkeypatch.setattr(module,'move_states',weak_survival)
    frames=[(dict(frame_id=i+1,timestamp=i*.04,captured_at=100+i*40),images[i]) for i in range(1,4)]
    output=flow.between(images[4],result(4),frames)
    assert all(not frame['overlays'] and not frame['events'] for frame in output)
