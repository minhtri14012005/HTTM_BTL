import numpy as np


def stable_association(tracking):
    """Only an existing reciprocal, unique link can resume a biometric anchor."""
    return bool(tracking.get("method") in ("overlap", "motion", "appearance") and
                tracking.get("candidate_count") == 1 and not tracking.get("identity_evidence_reset") and
                max(tracking.get("previous_iou") or 0, tracking.get("predicted_iou") or 0) >= .5 and
                tracking.get("appearance_similarity") is not None and tracking["appearance_similarity"] > .4 and
                0 <= tracking.get("appearance_age_seconds", float("inf")) <= 1.5)


def match(embedding, gallery, threshold, margin):
    scores = {}
    for person_id, name, reference in gallery:
        score = float(np.clip(np.dot(embedding, reference), -1, 1))
        if person_id not in scores or score > scores[person_id][0]:
            scores[person_id] = (score, name)
    ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    if not ranked:
        return None, None, 0.0
    pid, (best, name) = ranked[0]
    runner = ranked[1][1][0] if len(ranked) > 1 else -1
    return (pid, name, best) if best >= threshold and best-runner >= margin else (None, None, best)


class Temporal:
    """Never carry a confirmed label through a missing face or a conflicting match."""
    def __init__(self):
        self.tracks = {}
        self.present = {}
        self.last = {}
        self.visits = {}
        self.observed_tracks = {}
        self.alerted_tracks = {}
        self.identity_anchors = {}
        self.label_chains = {}

    def reset_identity_evidence(self):
        self.tracks.clear()
        self.identity_anchors.clear()
        self.label_chains.clear()

    def reset_track(self,track_id):
        self.tracks.pop(track_id,None)
        self.identity_anchors.pop(track_id,None)
        self.label_chains.pop(track_id,None)

    def anchor_identity(self,track_id,now):
        observed=self.observed_tracks.get(track_id)
        return self.identity_anchors.get(track_id) if observed is not None and 0<=now-observed<=1.5 else None

    def can_reverify(self, track_id, person_id, now):
        # This anchor cannot render a name or confirm a face. Two fresh calibrated
        # branches are required again after weak evidence expires; the anchor alone never suffices.
        return person_id is not None and self.anchor_identity(track_id,now)==person_id

    def can_bridge(self, track_id, person_id, now, settings):
        previous = self.tracks.get(track_id)
        return bool(previous and previous['pid']==person_id and previous.get('confirmed_once',False) and
                    0 <= now-previous['time'] <= .5 and
                    0 <= now-previous.get('multimodel_at',float('-inf')) <= .5)

    def update(self, track_id, person_id, now, settings, *, reset=False, frame_id=None, multimodel=True, reverified=False, fresh_model_count=0, continuity=False):
        previous = self.tracks.get(track_id)
        last_observed = self.observed_tracks.get(track_id)
        if reset or last_observed is None or not 0 <= now-last_observed <= 1.5:
            self.alerted_tracks.pop(track_id,None)
            self.identity_anchors.pop(track_id,None)
            self.label_chains.pop(track_id,None)
        resume = bool(not reset and reverified and multimodel and fresh_model_count >= 2 and
                      self.can_reverify(track_id,person_id,now))
        self.observed_tracks[track_id] = now
        if reset:
            self.tracks.pop(track_id, None)
            previous = None
        if not person_id:
            # Tracking may preserve an existing label, never add biometric votes or alerts.
            if previous is not None and not 0 <= now-previous['time'] <= 1.5:
                self.tracks.pop(track_id, None)
            if continuity and track_id in self.label_chains and self.label_chains[track_id]==self.anchor_identity(track_id,now):
                return 'tracked', False, self.visits.get(self.label_chains[track_id])
            self.label_chains.pop(track_id,None)
            return 'unknown', False, None
        if previous is None or previous['pid'] != person_id or not 0 <= now-previous['time'] <= 1.5:
            previous = dict(pid=person_id, hits=0, time=now, first_qualified=now, evidence=[])
        if resume:
            previous['confirmed_once'] = True
        if previous['hits'] and (now <= previous['time'] or (frame_id is not None and frame_id == previous.get('frame_id'))):
            return ('confirmed' if previous.get('confirmed_once') or previous['hits'] >= settings['confirm_frames'] else 'candidate'), False, self.visits.get(person_id)
        previous['evidence'] = [t for t in previous['evidence'] if 0 <= now-t <= 3.0]
        previous['evidence'].append(now)
        previous['hits'] = len(previous['evidence'])
        previous['time'],previous['frame_id'] = now,frame_id
        if multimodel:
            previous['multimodel_at'] = now
        previous['first_qualified'] = previous['evidence'][0]
        self.tracks[track_id] = previous
        last_present = self.present.get(person_id)
        self.present[person_id] = now
        if last_present is not None and now-last_present > settings['absence_seconds']:
            previous['reentry'] = True
        if previous['hits'] < settings['confirm_frames'] and not previous.get('confirmed_once'):
            self.label_chains.pop(track_id,None)
            return 'candidate', False, None
        previous['confirmed_once'] = True
        self.identity_anchors[track_id] = person_id
        self.label_chains[track_id] = person_id
        last_seen = self.last.get(person_id)
        is_new = last_seen is None or (last_present is not None and now-last_present > settings['absence_seconds'])
        is_new = is_new or previous.pop('reentry', False)
        self.last[person_id] = now
        # Notification memory is separate from biometric evidence: a continuously
        # observed face must requalify, but does not enter the room again on a weak frame.
        if self.alerted_tracks.get(track_id) == person_id:
            is_new = False
        self.alerted_tracks[track_id] = person_id
        if is_new:
            self.visits[person_id] = self.visits.get(person_id, 0)+1
        return 'confirmed', is_new, self.visits.get(person_id, 1)

    def finish_frame(self, now):
        self.tracks = {k: v for k, v in self.tracks.items() if now-v['time'] <= 1.5}
        self.observed_tracks = {k:t for k,t in self.observed_tracks.items() if 0<=now-t<=1.5}
        self.alerted_tracks = {k:pid for k,pid in self.alerted_tracks.items() if k in self.observed_tracks}
        self.identity_anchors = {k:pid for k,pid in self.identity_anchors.items() if k in self.observed_tracks}
        self.label_chains = {k:pid for k,pid in self.label_chains.items() if k in self.observed_tracks}
