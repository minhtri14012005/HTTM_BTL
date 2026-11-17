"""Preserve an established identity only through measured, unambiguous image links."""
import math


def overlaps(first, second):
    horizontal = min(first[2], second[2]) > max(first[0], second[0])
    vertical = min(first[3], second[3]) > max(first[1], second[1])
    return horizontal and vertical


def trusted_identity_link(face, previous_faces, current_faces, profile):
    if not profile or profile.get("enabled") is not True:
        return False
    track_id = face["track_id"]
    previous_face = next((item for item in previous_faces if item["track_id"] == track_id), None)
    link = face.get("tracking", {})
    if previous_face is None or link.get("candidate_count") != 1 or link.get("identity_evidence_reset"):
        return False
    if not 0 <= link.get("appearance_age_seconds", float("inf")) <= 1.5:
        return False
    # The tracker has already required reciprocal unique association. Continuity
    # uses its measured appearance gate, rather than the biometric resume IoU gate.
    # Observed and motion-predicted links use the tracker's existing >.25 spatial
    # admission gate. Both still require the unchanged calibrated face-appearance
    # gate below; geometry alone cannot preserve identity or create an alert.
    overlap_link = link.get("method") == "overlap" and (link.get("previous_iou") or 0) > .25
    predicted_link = link.get("method") == "motion" and (link.get("predicted_iou") or 0) > .25
    if not (overlap_link or predicted_link):
        return False
    threshold = profile.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        return False
    similarity = link.get("appearance_similarity")
    if similarity is None or not math.isfinite(similarity) or similarity < threshold:
        return False
    model, version = profile.get("model"), profile.get("version")
    # Both vectors must come from fresh inference, never a cached vector through an error.
    for sample in (previous_face, face):
        score = sample.get("model_scores", {}).get(model, {})
        if score.get("version") != version or score.get("error") or score.get("score") is None:
            return False
        if sample.get("tracking", {}).get("method") == "geometry":
            return False
    # A unique assignment alone is insufficient when another face occupies either region.
    for sample, group in ((previous_face, previous_faces), (face, current_faces)):
        box = sample.get("face_bbox", sample.get("bbox"))
        if not box:
            return False
        for other in group:
            other_box = other.get("face_bbox", other.get("bbox"))
            if other["track_id"] != track_id and overlaps(box, other_box):
                return False
    return True

