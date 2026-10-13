"""
Accuracy of the body pose tools on the whole OCHuman test split (occluded people).

Complements test_pose_coco_accuracy.py (8 well-known COCO images, every run) with a full,
less-known dataset: needs ``OCHUMAN_ROOT`` (see tests/pose_eval/ochuman.py) and is marked
``dataset``, so it only runs on demand:

    OCHUMAN_ROOT=/path/to/ochuman pytest -m dataset -s

Set ``OCHUMAN_LIMIT`` to evaluate only the first N images during development.
"""

import os

import pytest
import torch

from tests.pose_eval.ochuman import ENV_VAR, evaluate_tool, find_split

# model -> (required module, minimum COCO keypoint AP on the OCHuman test split).
MIN_AP = {
    "rtmo": ("rtmlib", 0.0),  # TODO: set a few points under the measured AP once OCHuman is available
    "vitpose": ("transformers", 0.0),
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
    assert scores["AP"] >= min_ap, f"{model} AP {scores['AP']:.3f} < {min_ap}"
