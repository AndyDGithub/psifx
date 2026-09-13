"""Shared types and COCO-17 keypoint mapping used by every backend adapter.

Every backend adapter exposes a `PoseBackend` with a single `predict(image)`
method returning a list of `PersonPose` in COCO-17 order, so the metrics code
(`metrics.py`) never needs to know which backend produced the keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Canonical COCO keypoint order (17 points) -- every backend adapter must
# emit keypoints in this exact order, filling missing points with NaN.
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
NUM_COCO_KEYPOINTS = len(COCO_KEYPOINT_NAMES)

# Per-keypoint OKS falloff constants (COCO evaluation standard), same order
# as COCO_KEYPOINT_NAMES.
COCO_OKS_SIGMAS = np.array(
    [
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
        0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    ]
)

# MediaPipe Pose (BlazePose, 33 landmarks) -> COCO-17 index mapping.
MEDIAPIPE_TO_COCO17 = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class PersonPose:
    """One detected person's COCO-17 keypoints for a single frame/image.

    keypoints: (17, 2) array of (x, y) in pixel coordinates, NaN where a
    keypoint is missing/unavailable.
    scores: (17,) array of per-keypoint confidence in [0, 1], NaN where
    missing. Backends without per-keypoint confidence report 1.0 for
    detected points.
    bbox: optional (x1, y1, x2, y2) person bounding box, used to normalize
    OKS/PCK when available (falls back to a keypoint-extent estimate).
    """

    keypoints: np.ndarray
    scores: np.ndarray
    bbox: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.keypoints.shape == (NUM_COCO_KEYPOINTS, 2), self.keypoints.shape
        assert self.scores.shape == (NUM_COCO_KEYPOINTS,), self.scores.shape


@dataclass
class FramePrediction:
    """All persons detected in one frame/image, plus backend timing."""

    people: List[PersonPose] = field(default_factory=list)
    inference_seconds: float = 0.0


def empty_person_pose() -> PersonPose:
    kp = np.full((NUM_COCO_KEYPOINTS, 2), np.nan, dtype=np.float64)
    sc = np.full((NUM_COCO_KEYPOINTS,), np.nan, dtype=np.float64)
    return PersonPose(keypoints=kp, scores=sc)


class PoseBackend:
    """Interface every backend adapter implements."""

    name: str

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        raise NotImplementedError

    def close(self) -> None:
        pass
