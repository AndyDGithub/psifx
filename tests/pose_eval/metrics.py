"""Precision (OKS/PCK) and temporal-stability (normalized jerk) metrics.

Kept dependency-free (numpy only) and backend-agnostic: every function
operates on the COCO_KEYPOINT_NAMES-ordered arrays defined in coco.py, so
it works identically for MediaPipe, RTMPose or ViTPose predictions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from tests.pose_eval.coco import COCO_OKS_SIGMAS, NUM_COCO_KEYPOINTS, PersonPose


def object_keypoint_similarity(
    pred: PersonPose,
    gt_keypoints: np.ndarray,
    gt_visible: np.ndarray,
    area: float,
) -> Optional[float]:
    """Standard COCO OKS between one prediction and one ground-truth person.

    gt_keypoints: (17, 2) ground-truth (x, y). gt_visible: (17,) bool, True
    where the ground-truth keypoint is labeled/visible. area: ground-truth
    person bbox area in pixels^2 (COCO convention). Returns None if the
    ground truth has no visible keypoints at all (undefined OKS).
    """
    if not np.any(gt_visible):
        return None
    d2 = np.sum((pred.keypoints - gt_keypoints) ** 2, axis=1)
    k2 = (2 * COCO_OKS_SIGMAS) ** 2
    # COCO normalizes by (area + eps) so a person with near-zero bbox area
    # (annotation artifact) doesn't produce an exploding OKS denominator.
    e = d2 / (k2 * (area + np.spacing(1)) * 2)
    e = e[gt_visible & ~np.isnan(pred.keypoints[:, 0])]
    if e.size == 0:
        return 0.0
    return float(np.mean(np.exp(-e)))


def pck(
    pred: PersonPose,
    gt_keypoints: np.ndarray,
    gt_visible: np.ndarray,
    ref_length: float,
    threshold: float = 0.2,
) -> Optional[float]:
    """Percentage of Correct Keypoints: fraction of visible GT keypoints
    within `threshold * ref_length` pixels of the prediction.

    ref_length is a person-scale reference (here: bbox diagonal), matching
    the "PCK relative to a scale reference" definition used throughout
    02_methodologie_metriques.md.
    """
    if not np.any(gt_visible) or ref_length <= 0:
        return None
    dist = np.linalg.norm(pred.keypoints - gt_keypoints, axis=1)
    valid = gt_visible & ~np.isnan(pred.keypoints[:, 0])
    if not np.any(valid):
        return 0.0
    correct = dist[valid] <= (threshold * ref_length)
    return float(np.mean(correct))


def match_prediction_to_gt(
    predictions: Sequence[PersonPose], gt_keypoints: np.ndarray
) -> Optional[PersonPose]:
    """Pick the predicted person closest to one ground-truth person.

    COCO images can contain several people; for a single-GT-person image we
    just take the prediction whose mean keypoint position is nearest the
    ground truth's, which is sufficient for this benchmark's purpose
    (comparing backends against each other, not full COCO-style multi-person
    assignment).
    """
    if not predictions:
        return None
    gt_center = np.nanmean(gt_keypoints, axis=0)
    best, best_dist = None, np.inf
    for p in predictions:
        c = np.nanmean(p.keypoints, axis=0)
        if np.any(np.isnan(c)):
            continue
        d = np.linalg.norm(c - gt_center)
        if d < best_dist:
            best, best_dist = p, d
    return best


def normalized_jerk_series(
    keypoint_track: np.ndarray, fps: float, scale: float
) -> float:
    """Temporal-stability proxy (no ground truth needed): mean magnitude of
    the second derivative (acceleration change / jerk) of one keypoint's
    trajectory across frames, normalized by fps and person scale so it is
    comparable across backends and videos.

    keypoint_track: (T, 2) pixel positions of one keypoint over T frames
    (NaN where undetected -- those frames are skipped via interpolation-free
    finite differences on the valid runs only).
    scale: a person-scale reference (e.g. shoulder-to-hip pixel distance)
    to normalize out "bigger person in frame = bigger raw pixel jerk".

    This is a proxy for the "temporal stability" axis of the MBRCS
    composite score discussed in 02_methodologie_metriques.md (Sec. 9.2) --
    not a re-implementation of the full published metric, which needs
    kinematic ground truth we don't have here.
    """
    valid = ~np.any(np.isnan(keypoint_track), axis=1)
    if valid.sum() < 3 or scale <= 0:
        return float("nan")
    pos = keypoint_track[valid]
    dt = 1.0 / fps
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    magnitude = np.linalg.norm(jerk, axis=1)
    return float(np.mean(magnitude) / scale)


def person_scale(pose: PersonPose) -> float:
    """Shoulder-to-hip pixel distance as a rough person-scale reference,
    used to normalize the jerk metric. Falls back to NaN if either side is
    missing.
    """
    from tests.pose_eval.coco import COCO_KEYPOINT_NAMES

    ls = COCO_KEYPOINT_NAMES.index("left_shoulder")
    rh = COCO_KEYPOINT_NAMES.index("right_hip")
    a, b = pose.keypoints[ls], pose.keypoints[rh]
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))
