"""Precision pass: run every backend on the COCO subset, compute OKS/PCK.

ViTPose is fed the ground-truth bounding box directly (oracle-box top-down
evaluation, see backends.py docstring) -- MediaPipe and RTMPose detect the
person themselves.

Usage: python run_accuracy.py [--backends mediapipe,rtmpose,vitpose] [--device cuda]
Writes: results/accuracy.csv
"""

from __future__ import annotations

# Import order fix -- see the matching comment in run_video_benchmark.py:
# torch must be imported before onnxruntime (via rtmlib) claims the
# process's CUDA runtime soname slot, or `import torch` for ViTPose later
# crashes with an undefined-symbol ImportError.
import torch  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backends import ViTPoseBackend, build_backend
from common import NUM_COCO_KEYPOINTS
from metrics import match_prediction_to_gt, object_keypoint_similarity, pck

HERE = Path(__file__).resolve().parent
COCO_DIR = HERE / "data" / "coco_subset"
RESULTS_DIR = HERE / "results"


def parse_gt(ann: dict) -> tuple[np.ndarray, np.ndarray]:
    kp = np.array(ann["keypoints"], dtype=np.float64).reshape(NUM_COCO_KEYPOINTS, 3)
    xy = kp[:, :2]
    visible = kp[:, 2] > 0  # COCO: 0=not labeled, 1=labeled not visible, 2=labeled visible
    xy[~visible] = np.nan
    return xy, visible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="mediapipe,rtmpose,vitpose")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pck-threshold", type=float, default=0.2)
    args = parser.parse_args()

    ann_path = COCO_DIR / "annotations.json"
    if not ann_path.exists():
        raise SystemExit(f"{ann_path} not found -- run prepare_coco_subset.py first")
    coco = json.loads(ann_path.read_text())
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, dict] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], ann)  # one per image by construction

    # Load every image once up front -- also lets Sapiens2 (subprocess per
    # backend, not per image) run in a single batched call instead of
    # paying interpreter/model-load startup cost per image.
    items = []  # (image_id, ann, image_bgr)
    for image_id, ann in anns_by_image.items():
        img_meta = images_by_id[image_id]
        img_path = COCO_DIR / "images" / img_meta["file_name"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  skip unreadable {img_path}")
            continue
        items.append((image_id, ann, image_bgr))

    backend_names = args.backends.split(",")
    rows = []
    for backend_name in backend_names:
        print(f"=== {backend_name} ===")
        backend = build_backend(backend_name, device=args.device)

        if hasattr(backend, "predict_all"):
            # Subprocess-based backends (Sapiens2, DETRPose) batch every
            # image into one call -- interpreter/model-load startup cost
            # would otherwise dominate a per-image subprocess timing.
            batch_preds = backend.predict_all([img for _, _, img in items])
        else:
            batch_preds = None

        for idx, (image_id, ann, image_bgr) in enumerate(items):
            gt_xy, gt_visible = parse_gt(ann)
            x, y, w, h = ann["bbox"]

            if batch_preds is not None:
                pred_frame = batch_preds[idx]
            elif isinstance(backend, ViTPoseBackend):
                boxes = np.array([[x, y, x + w, y + h]], dtype=np.float64)
                pred_frame = backend.predict_with_boxes(image_bgr, boxes)
            else:
                pred_frame = backend.predict(image_bgr)

            best = match_prediction_to_gt(pred_frame.people, gt_xy)
            if best is None:
                rows.append(dict(backend=backend_name, image_id=image_id, oks=0.0, pck=0.0, detected=False))
                continue

            oks = object_keypoint_similarity(best, gt_xy, gt_visible, area=float(ann["area"]))
            diag = float(np.hypot(w, h))
            pck_score = pck(best, gt_xy, gt_visible, ref_length=diag, threshold=args.pck_threshold)
            rows.append(
                dict(
                    backend=backend_name,
                    image_id=image_id,
                    oks=oks if oks is not None else float("nan"),
                    pck=pck_score if pck_score is not None else float("nan"),
                    detected=True,
                )
            )
        backend.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "accuracy_raw.csv", index=False)

    summary = df.groupby("backend").agg(
        mean_oks=("oks", "mean"),
        mean_pck=("pck", "mean"),
        detection_rate=("detected", "mean"),
        n_images=("image_id", "count"),
    )
    summary.to_csv(RESULTS_DIR / "accuracy_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
