"""
Accuracy of the body pose tools on the whole OCHuman test split (occluded people).

Complements test_pose_coco_accuracy.py (8 well-known COCO images, every run) with a full,
less-known dataset: needs ``OCHUMAN_ROOT`` (see tests/pose_eval/ochuman.py) and is marked
``dataset``, so it only runs on demand:

    OCHUMAN_ROOT=/path/to/ochuman pytest -m dataset -s

Set ``OCHUMAN_LIMIT`` to evaluate only the first N images during development (scores are printed,
the thresholds are not checked).
"""

import os

import pytest
import torch

from tests.pose_eval.ochuman import ENV_VAR, evaluate_tool, find_split

# model -> (required module, minimum COCO keypoint AP on the OCHuman test split), about 3 points
# under the AP measured on 2026-10-20 (2231 images, RTX 4060): RTMO-m 0.289, ViTPose-B 0.343.
# Low AP is expected: OCHuman does not annotate every person, so correct detections of unlabeled
# people count as false positives, and occluded people are often missed by the detectors.
MIN_AP = {
    "rtmo": ("rtmlib", 0.26),
    "vitpose": ("transformers", 0.31),
}


def make_tool(model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "rtmo":
        from psifx.video.pose.rtmo.tool import RTMOPoseEstimationTool

        return RTMOPoseEstimationTool(device=device, verbose=False)
    from psifx.video.pose.vitpose.tool import ViTPosePoseEstimationTool

    return ViTPosePoseEstimationTool(device=device, verbose=False)


@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.parametrize("model", sorted(MIN_AP))
def test_pose_tool_ochuman_ap(model):
    if not os.environ.get(ENV_VAR):
        pytest.skip(f"{ENV_VAR} not set")
    module, min_ap = MIN_AP[model]
    pytest.importorskip(module, reason=f"{module} not installed")
    images, annotations = find_split("test")
    limit = int(os.environ["OCHUMAN_LIMIT"]) if os.environ.get("OCHUMAN_LIMIT") else None

    scores = evaluate_tool(make_tool(model).predict, images, annotations, limit=limit)
    print(f"\n[{model}] OCHuman test {scores}")
    if limit is not None:
        # The thresholds hold for the whole split; a few images can score far lower or higher.
        pytest.skip(f"OCHUMAN_LIMIT={limit}: AP printed, thresholds only checked on the whole split")
    assert scores["AP"] >= min_ap, f"{model} AP {scores['AP']:.3f} < {min_ap}"
