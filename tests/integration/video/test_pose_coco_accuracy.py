"""
Accuracy regression test for psifx pose tools on a small COCO subset.

The 8 COCO val2017 images in ``data/CocoPose`` (one large, single annotated
person each) are letterboxed into a video and run through the real CLI
(``psifx video pose <tool> single-inference``). The resulting archive is
scored with COCO OKS / PCK@0.2 (see research/benchmark/psifx_eval.py).

Every pose tool is evaluated the same way, so a new model can be compared
directly to the one it replaces. To add a tool: register its COCO-17 adapter in
``psifx_eval.ADAPTERS`` and its minimum scores in ``MIN_SCORES`` below.
"""

from pathlib import Path

import pytest

from psifx_eval import build_coco_video, evaluate_archive, load_coco_samples
from tests.integration.conftest import run_command

COCO_DIR = Path(__file__).resolve().parents[1] / "data" / "CocoPose"

# tool -> (required module, minimum OKS, minimum PCK@0.2, extra CLI args).
# Minimums sit a few points under the measured score: they catch regressions,
# not small run-to-run noise.
MIN_SCORES = {
    "mediapipe": ("mediapipe", 0.60, 0.90, []),  # measured: OKS 0.647, PCK 0.968
}


@pytest.fixture(scope="module")
def coco_video(tmp_path_factory):
    """Letterboxed COCO video + its ground truth, built once per module."""
    samples = load_coco_samples(COCO_DIR / "annotations.json", COCO_DIR / "images")
    video_path = tmp_path_factory.mktemp("coco") / "coco.mp4"
    ground_truth = build_coco_video(samples, video_path)
    return video_path, ground_truth


@pytest.mark.integration
@pytest.mark.parametrize("tool", sorted(MIN_SCORES))
def test_pose_tool_coco_accuracy(tool, coco_video, output_dir):
    module, min_oks, min_pck, extra = MIN_SCORES[tool]
    pytest.importorskip(module, reason=f"{module} not installed")
    video_path, ground_truth = coco_video
    poses_path = output_dir / f"{tool}.tar.gz"

    run_command(
        "psifx", "video", "pose", tool, "single-inference",
        "--video", video_path,
        "--poses", poses_path,
        "--overwrite", "--no-verbose", *extra,
    )

    scores = evaluate_archive(poses_path, ground_truth, tool)
    print(f"\n[{tool}] {scores}")
    assert scores["oks"] >= min_oks, f"{tool} OKS {scores['oks']:.3f} < {min_oks}"
    assert scores["pck"] >= min_pck, f"{tool} PCK {scores['pck']:.3f} < {min_pck}"
