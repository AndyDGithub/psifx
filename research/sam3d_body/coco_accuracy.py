"""SAM 3D Body on the COCO val2017 subset of research/mediapipe_tasks/coco_accuracy.py, with the ground-truth box.

SAM 3D Body is top-down: like the paper (Table 2, COCO PCK@0.05), it gets the annotated person's box,
so the scores are not directly comparable with MediaPipe, which finds the person itself. Reports:
- ``oks`` and ``pck_0.2_diag``: the psifx metrics (tests/pose_eval/metrics.py), PCK relative to the
  box diagonal;
- ``pck_0.05_box``: PCK@0.05 relative to the longest box side, the convention of the HMR literature.

    python research/sam3d_body/coco_accuracy.py .e2e_outputs/scratch/coco500 \\
        --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tracks import MHR70_TO_COCO17, load_estimator  # noqa: E402
from tests.pose_eval.coco import PersonPose  # noqa: E402
from tests.pose_eval.metrics import object_keypoint_similarity, pck  # noqa: E402
from tests.pose_eval.psifx_eval import load_coco_samples  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="folder with annotations.json and images/")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = load_coco_samples(args.data / "annotations.json", args.data / "images")
    estimator = load_estimator(args.repo, args.checkpoint, args.device)
    oks, pck_diag, pck_box, seconds = [], [], [], []
    for sample in samples:
        ann = sample["annotation"]
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        x, y, w, h = ann["bbox"]
        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        gt, visible = kp[:, :2], kp[:, 2] > 0
        start = time.perf_counter()
        with open(os.devnull, "w") as null:
            stdout, sys.stdout = sys.stdout, null
            try:
                out = estimator.process_one_image(image, bboxes=np.array([[x, y, x + w, y + h]], dtype=np.float32))
            finally:
                sys.stdout = stdout
        seconds.append(time.perf_counter() - start)
        if not out:
            oks.append(0.0), pck_diag.append(0.0), pck_box.append(0.0)
            continue
        pred = PersonPose(keypoints=out[0]["pred_keypoints_2d"][MHR70_TO_COCO17, :2], scores=np.ones(17))
        oks.append(object_keypoint_similarity(pred, gt, visible, ann["area"]) or 0.0)
        pck_diag.append(pck(pred, gt, visible, float(np.hypot(w, h))) or 0.0)
        pck_box.append(pck(pred, gt, visible, float(max(w, h)), threshold=0.05) or 0.0)

    scores = {
        "oks": round(float(np.mean(oks)), 3),
        "pck_0.2_diag": round(float(np.mean(pck_diag)), 3),
        "pck_0.05_box": round(float(np.mean(pck_box)), 3),
        "images": len(samples),
        "seconds_per_image": round(float(np.median(seconds)), 3),
    }
    (args.data / "scores_sam3d_body.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
