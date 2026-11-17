"""Visual face tracking on actual intermediate frames; identity comes from AI anchors."""
import time
import io
from PIL import Image
import cv2
import numpy as np


def decode_tracking_frame(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.width*image.height > 40_000_000:
            raise ValueError('Frame theo dõi quá lớn.')
    gray = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('Không đọc được frame theo dõi.')
    return gray


def overlap(first, second):
    left, top = max(first[0],second[0]), max(first[1],second[1])
    right, bottom = min(first[2],second[2]), min(first[3],second[3])
    area = max(0,right-left)*max(0,bottom-top)
    return area/max(1,(first[2]-first[0])*(first[3]-first[1])+
                    (second[2]-second[0])*(second[3]-second[1])-area)


def feature_states(gray, faces):
    states = []
    height,width = gray.shape
    for face in faces:
        box = face.get('face_bbox')
        if not box or len(box) != 4 or not np.isfinite(box).all():
            continue
        x1,y1,x2,y2 = np.asarray(box,dtype=float)
        inset = max(1,int(min(x2-x1,y2-y1)*.12))
        left,top = max(0,int(x1)+inset),max(0,int(y1)+inset)
        right,bottom = min(width,int(x2)-inset),min(height,int(y2)-inset)
        if right-left < 4 or bottom-top < 4:
            continue
        points = cv2.goodFeaturesToTrack(gray[top:bottom,left:right],maxCorners=24,
            qualityLevel=.01,minDistance=1.5,blockSize=3)
        if points is None or len(points) < 4:
            continue
        points += np.array([left,top],np.float32)
        states.append(dict(face=face,box=np.asarray(box,np.float32),points=points,seed_count=len(points)))
    return states


def move_states(previous, current, states):
    if not states or previous.shape != current.shape:
        return []
    points = np.concatenate([state['points'] for state in states])
    moved,status,error = cv2.calcOpticalFlowPyrLK(previous,current,points,None,
        winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
    if moved is None:
        return []
    returned,back_status,_ = cv2.calcOpticalFlowPyrLK(current,previous,moved,None,
        winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
    if returned is None:
        return []
    good = (status.reshape(-1)>0)&(back_status.reshape(-1)>0)&(error.reshape(-1)<25)
    good &= np.linalg.norm(returned.reshape(-1,2)-points.reshape(-1,2),axis=1)<1.2
    updated,offset = [],0
    height,width = current.shape
    for state in states:
        count = len(state['points'])
        mask = good[offset:offset+count]
        before = points[offset:offset+count][mask]
        after = moved[offset:offset+count][mask]
        offset += count
        if len(after) < 4 or len(after) < count*.6:
            continue
        transform,inliers = cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,
            ransacReprojThreshold=1.2,maxIters=50,confidence=.95)
        if transform is None or inliers.sum() < 4:
            continue
        scale = float(np.linalg.norm(transform[:,0]))
        if not .75 <= scale <= 1.35:
            continue
        x1,y1,x2,y2 = state['box']
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
        corners = cv2.transform(corners[None],transform)[0]
        box = np.r_[corners.min(axis=0),corners.max(axis=0)]
        if box[0]<0 or box[1]<0 or box[2]>width or box[3]>height:
            continue
        selected = after[inliers.reshape(-1)>0]
        updated.append(dict(face=state['face'],box=box,points=selected,
                            confidence=float(len(selected)/count),seed_count=state.get('seed_count',count),
                            cumulative_confidence=float(len(selected)/state.get('seed_count',count))))
    return updated


class FaceFlow:
    def __init__(self):
        self.previous_gray = None
        self.previous_result = None

    def between(self, image, result, frames):
        started = time.perf_counter()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        previous = self.previous_result
        previous_gray = self.previous_gray
        self.previous_gray,self.previous_result = gray,result
        if not frames:
            return []
        states = feature_states(gray,result.get('overlays',[]))
        samples = [(metadata,frame if frame.ndim == 2 else cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)) for metadata,frame in frames]
        samples.reverse()
        same_source = previous is not None and previous_gray.shape == gray.shape
        if same_source:
            samples.append((dict(frame_id=previous['frame_id'],timestamp=previous['media_timestamp'],
                                 captured_at=previous['captured_at']),previous_gray))
        tracked = {}
        last_gray,last_timestamp = gray,result['media_timestamp']
        for metadata,current in samples:
            elapsed = last_timestamp-metadata['timestamp']
            if not 0 < elapsed <= .15 or current.shape != last_gray.shape:
                states = []
            else:
                states = move_states(last_gray,current,states)
            tracked[metadata['frame_id']] = {state['face']['track_id']:state for state in states}
            last_gray,last_timestamp = current,metadata['timestamp']
        previous_faces = {face['track_id']:face for face in previous.get('overlays',[])} if same_source else {}
        origin = tracked.get(previous['frame_id'],{}) if same_source else {}
        safe_identity,invalid = set(),set()
        for face in result.get('overlays',[]):
            key = face['track_id']
            old = previous_faces.get(key)
            matched = origin.get(key)
            if old and matched and overlap(old['face_bbox'],matched['box']) < .3:
                invalid.add(key)
            visible_identity=lambda value: value.get('status')=='confirmed' or (value.get('status')=='tracked' and value.get('identity_source')=='track_continuity')
            if old and matched and visible_identity(face) and visible_identity(old) and (
                    face.get('person_id') == old.get('person_id') and face.get('person_id') and
                    overlap(old['face_bbox'],matched['box']) >= .5 and
                    result['media_timestamp']-previous['media_timestamp'] <= .5):
                safe_identity.add(key)
        # Reseeding only from the newest image loses all intermediate boxes when
        # that image has too few corners. Recover from the older real detector
        # anchor, but accept a path only if it reaches the newer detection.
        current_faces = {face['track_id']:face for face in result.get('overlays',[])}
        missing = {key for key in previous_faces if key in current_faces and
                   (key in invalid or any(key not in tracked.get(metadata['frame_id'],{}) for metadata,_ in frames))}
        forward, forward_valid, safe_forward = {}, set(), set()
        if missing and 0 < result['media_timestamp']-previous['media_timestamp'] <= .5:
            forward_states = feature_states(previous_gray,[previous_faces[key] for key in missing])
            forward_gray, forward_timestamp = previous_gray, previous['media_timestamp']
            forward_samples = [(metadata,frame if frame.ndim==2 else cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)) for metadata,frame in frames]
            forward_samples.append((dict(frame_id=result['frame_id'],timestamp=result['media_timestamp']),gray))
            for metadata,current in forward_samples:
                elapsed = metadata['timestamp']-forward_timestamp
                if not 0 < elapsed <= .15 or current.shape != forward_gray.shape:
                    forward_states = []
                else:
                    forward_states = move_states(forward_gray,current,forward_states)
                forward[metadata['frame_id']] = {state['face']['track_id']:state for state in forward_states}
                forward_gray, forward_timestamp = current,metadata['timestamp']
            for key,state in forward.get(result['frame_id'],{}).items():
                face,old = current_faces[key],previous_faces[key]
                if overlap(state['box'],face['face_bbox']) < .5 or state.get('cumulative_confidence',0) < .6:
                    continue
                forward_valid.add(key)
                if key not in invalid and visible_identity(face) and visible_identity(old) and (
                        face.get('person_id') and face.get('person_id')==old.get('person_id')):
                    safe_forward.add(key)
        output = []
        rescued = 0
        for metadata,_ in frames:
            overlays = []
            backward_states = tracked.get(metadata['frame_id'],{})
            forward_states = forward.get(metadata['frame_id'],{})
            for key in sorted(set(backward_states) | (set(forward_states)&forward_valid)):
                backward = backward_states.get(key) if key not in invalid else None
                ahead = forward_states.get(key) if key in forward_valid else None
                if backward is not None and ahead is not None and overlap(backward['box'],ahead['box']) < .5:
                    continue  # Contradictory image paths cannot select a person or a box.
                state = backward if backward is not None else ahead
                if state is None:
                    continue
                direction = 'backward' if backward is not None else 'forward'
                rescued += direction=='forward'
                face = current_faces[key]
                confirmed = key in (safe_identity if backward is not None else safe_forward)
                box = [float(value) for value in state['box']]
                overlays.append(dict(track_id=key,bbox=box,face_bbox=box,
                    status='tracked' if confirmed else 'unknown',
                    identity_source='track_continuity' if confirmed else None,
                    person_id=face.get('person_id') if confirmed else None,
                    name=face.get('name') if confirmed else None,
                    tracking_origin='optical_flow',detection_frame_id=result['frame_id'],
                    flow_direction=direction,flow_confidence=state.get('confidence',0),cumulative_flow_confidence=state.get('cumulative_confidence'),reason='Theo dõi đặc trưng ảnh cùng frame'))
            output.append(dict(frame_id=metadata['frame_id'],media_timestamp=metadata['timestamp'],
                captured_at=metadata['captured_at'],width=image.shape[1],height=image.shape[0],
                overlays=overlays,events=[],overlay_origin='optical_flow'))
        # Crossing regions must abstain even if both endpoint labels agree.
        from .identity_tracking import overlaps
        for frame in output:
            for face in frame['overlays']:
                if any(other is not face and overlaps(face['face_bbox'],other['face_bbox']) for other in frame['overlays']):
                    face.update(status='unknown',identity_source=None,person_id=None,name=None)
        result['visual_tracking_summary'] = dict(anchor_frame_id=result['frame_id'],
            input_frames=len(frames),frames_with_box=sum(bool(frame['overlays']) for frame in output),
            forward_rescued_boxes=rescued,frames=[dict(frame_id=frame['frame_id'],timestamp=frame['media_timestamp'],
                faces=len(frame['overlays']),forward_boxes=sum(face.get('flow_direction')=='forward' for face in frame['overlays']),
                tracks=[dict(track_id=face['track_id'],status=face['status'],person_id=face.get('person_id')) for face in frame['overlays']]) for frame in output])
        result['visual_tracking_ms'] = (time.perf_counter()-started)*1000
        return output
