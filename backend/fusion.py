"""Quality-specific, calibrated votes; an abstaining branch cannot veto evidence."""
import math


def identity_evidence(scores):
    support, roles = [], {}
    families = set()
    reference = []
    for key, score in scores.items():
        value = score.get('score')
        if score.get('shadow'):
            roles[key] = 'shadow_only'
            continue
        if score.get('error') or value is None or not math.isfinite(value) or not score.get('person_id'):
            roles[key] = 'unavailable'
            continue
        if score.get('passed'):
            reference.append(score)
        if not score.get('calibrated'):
            roles[key] = 'uncalibrated'
        elif key == 'petalface' and not score.get('iqa_threshold_verified'):
            roles[key] = 'iqa_unverified'
        elif not score.get('passed'):
            roles[key] = 'below_calibrated_gate'
        else:
            roles[key] = 'support'
            support.append(score)
            families.add(score.get('recognizer_family', key))
    identities = {s['person_id'] for s in support}
    conflict = len(identities) > 1
    eligible = len(families) >= 2 and len(identities) == 1
    reference_ids = {s['person_id'] for s in reference}
    proposed = next(iter(identities)) if len(identities) == 1 else (
        next(iter(reference_ids)) if not identities and len(reference_ids) == 1 else None)
    return proposed, eligible, conflict, dict(model_roles=roles, calibrated_support=len(families),
                                            qualified_branches=len(support),
                                            calibrated_conflict=conflict)


def quality_basis(alignment='regular'):
    return 'native_height_for_high_residual_v1' if alignment!='regular' else 'native_min_dimension_v1'


def quality_band(pixels, alignment='regular', face_height=None):
    # A near profile narrows horizontally without losing its native vertical detail.
    # Calibration uses the same metric; this never changes model input pixels.
    detail = face_height if alignment!='regular' and face_height is not None else pixels
    band = 'tiny' if detail < 28 else 'small' if detail < 90 else 'normal'
    return band if alignment=='regular' else band+'_high_residual'


def model_gates(value, margin, profile, person_id=None, runner_up_person_id=None):
    if value is None or not math.isfinite(value) or (margin is not None and (not math.isfinite(margin) or margin<0)):
        return False,False
    base = profile or dict(threshold=.5,margin=.08)
    identity_gates = base.get('identity_gates', {})
    gate = dict(base, **identity_gates.get(person_id, {}))
    competitor_threshold = gate['threshold']
    if identity_gates:
        if runner_up_person_id:
            competitor_threshold = identity_gates.get(runner_up_person_id, base)['threshold']
        else:
            # Older measurements may omit the runner-up id. Use the strictest
            # possible rival threshold instead of guessing the missing identity.
            competitor_threshold = min([base['threshold']] + [item['threshold']
                for identity, item in identity_gates.items() if identity != person_id])
    identify = value>=gate['threshold'] and (margin is None or margin>=gate['margin'])
    # Re-verification is for an already confirmed track only. A competing identity
    # above the same calibrated absolute threshold makes the evidence ambiguous.
    verify = bool(profile and profile.get('verification_enabled') is True and value>=gate['threshold'] and (margin is None or
                  (margin>gate.get('numeric_guard',.001) and value-margin<competitor_threshold)))
    return identify,verify


def verification_evidence(scores, expected_person_id=None):
    votes,roles=[],{}
    families=set()
    for key,score in scores.items():
        value=score.get('score')
        valid=(not score.get('shadow') and value is not None and math.isfinite(value) and score.get('person_id') and
               not score.get('error') and score.get('calibrated') and
               (key!='petalface' or score.get('iqa_threshold_verified')))
        if valid and (score.get('passed') or
                      (score.get('verification_passed') and expected_person_id is not None and
                       score.get('person_id')==expected_person_id)):
            votes.append(score);roles[key]='support'
            families.add(score.get('recognizer_family',key))
        else:roles[key]='abstain'
    identities={score['person_id'] for score in votes}
    return (next(iter(identities)) if len(identities)==1 else None,
            len(families)>=2 and len(identities)==1,len(identities)>1,
            dict(model_roles=roles,calibrated_support=len(families),qualified_branches=len(votes)))
