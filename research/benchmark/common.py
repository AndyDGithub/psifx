"""Benchmark-specific types; the COCO-17 definitions live in tests/pose_eval/coco.py.

Every backend adapter exposes a `PoseBackend` with a single `predict(image)`
method returning a list of `PersonPose` in COCO-17 order, so the metrics code
never needs to know which backend produced the keypoints.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# The evaluation code is shared with the psifx tests: import it from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.pose_eval.coco import (  # noqa: E402,F401
    COCO_KEYPOINT_NAMES,
    COCO_OKS_SIGMAS,
    MEDIAPIPE_TO_COCO17,
    NUM_COCO_KEYPOINTS,
    PersonPose,
    empty_person_pose,
)


@dataclass
class FramePrediction:
    """All persons detected in one frame/image, plus backend timing."""

    people: List[PersonPose] = field(default_factory=list)
    inference_seconds: float = 0.0


class PoseBackend:
    """Interface every backend adapter implements."""

    name: str

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        raise NotImplementedError

    def close(self) -> None:
        pass
