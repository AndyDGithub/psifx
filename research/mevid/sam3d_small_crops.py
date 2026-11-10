"""Does SAM 3D Body hold on the small MEVID crops? Measured by crop height, without pose annotations.

Each chip of ``sample.json`` is given whole to SAM 3D Body (box = the chip, no mask), in ``body`` and ``full``
mode. ViTPose (the psifx default, OKS 0.893 on the talk-show annotations) on the same box is the 2D reference; it
also gets worse on small crops, so its mean score is reported beside.

Per height bin and mode:
- ``oks_vs_vitpose``: OKS against ViTPose keypoints scored above 0.3, the chip area as the object area (larger than
  a segment area, so the OKS is lenient), and the share of chips under 0.5 (``failed``);
- ``shoulder_cv``: shoulder width variation inside a tracklet (std / mean), a size that should not change;
- ``shape_within_between``: MHR identity parameters, variance inside a tracklet over variance between tracklets;
  a stable identity gives a small value;
- ``accel_mm``: 3D acceleration around the pelvis between consecutive frames, the jitter;
- ``seconds_per_chip``.
Also writes ``sheet.jpg``: chips per bin, SAM 3D Body points in green, ViTPose in red.

    python research/mevid/sam3d_small_crops.py .e2e_outputs/ext/mevid/small_crops \\
        --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sam3d_body"))
from run_tracks import MHR70_TO_COCO17, load_estimator  # noqa: E402

SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 9, 10  # MHR-70 indices.
MODES = ("body", "full")


def oks(points: np.ndarray, reference: np.ndarray, scores: np.ndarray, area: float, min_score: float = 0.3) -> float:
    visible = scores > min_score
    if not visible.any():
        return float("nan")
    distance = ((points - reference) ** 2).sum(1)
    return float(np.exp(-distance / (2 * area * (2 * SIGMAS) ** 2))[visible].mean())


def sam3d(estimator, image: np.ndarray, mode: str) -> dict:
    h, w = image.shape[:2]
    with open(os.devnull, "w") as null:  # The estimator prints a line per call.
        stdout, sys.stdout = sys.stdout, null
        try:
            out = estimator.process_one_image(image, bboxes=np.array([[0, 0, w, h]], np.float32), inference_type=mode)
        finally:
            sys.stdout = stdout
    if len(out) != 1:
        sys.exit(f"error: {len(out)} results for one box")
    o = out[0]
    return {"k2": o["pred_keypoints_2d"][:, :2].astype(np.float32), "k3": np.asarray(o["pred_keypoints_3d"], np.float32),
            "cam_t": np.asarray(o["pred_cam_t"], np.float32), "shape": np.asarray(o["shape_params"], np.float32)}


@torch.inference_mode()
def vitpose(processor, model, image: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    boxes = [np.array([[0, 0, w, h]], np.float32)]  # COCO [x, y, width, height].
    inputs = processor(image, boxes=boxes, return_tensors="pt").to(device)
    result = processor.post_process_pose_estimation(model(**inputs), boxes=boxes)[0][0]
    return result["keypoints"].cpu().numpy()[:17], result["scores"].cpu().numpy()[:17]


def summarize(tracks: list[dict], mode: str) -> dict:
    o, shoulders_cv, accelerations, shapes, means = [], [], [], [], []
    for t in tracks:
        chips = t["chips"]
        o += [c[f"oks_{mode}"] for c in chips]
        k3 = np.stack([c[mode]["k3"] for c in chips])
        shoulders = np.linalg.norm(k3[:, LEFT_SHOULDER] - k3[:, RIGHT_SHOULDER], axis=1)
        shoulders_cv.append(float(shoulders.std() / max(shoulders.mean(), 1e-6)))
        relative = k3 - k3[:, [LEFT_HIP, RIGHT_HIP]].mean(1, keepdims=True)
        accelerations.append(float(np.median(np.linalg.norm(relative[2:] - 2 * relative[1:-1] + relative[:-2], axis=2)) * 1000))
        shape = np.stack([c[mode]["shape"] for c in chips])
        shapes.append(shape.var(0))
        means.append(shape.mean(0))
    o = np.array(o)
    within, between = np.mean(shapes, axis=0), np.var(np.stack(means), axis=0)
    return {"oks_vs_vitpose_p50": round(float(np.nanmedian(o)), 3), "failed": round(float(np.nanmean(o < 0.5)), 3),
            "shoulder_cv_p50": round(float(np.median(shoulders_cv)), 3), "accel_mm_p50": round(float(np.median(accelerations)), 1),
            "shape_within_between": round(float(within.sum() / max(between.sum(), 1e-9)), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sample.json and chips/")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vitpose", default="usyd-community/vitpose-base-simple")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sample_path = args.folder / "sample.json"
    if not sample_path.is_file():
        sys.exit(f"error: missing {sample_path}")
    sample = json.loads(sample_path.read_text())
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    processor = AutoProcessor.from_pretrained(args.vitpose)
    pose_model = VitPoseForPoseEstimation.from_pretrained(args.vitpose).to(args.device).eval()
    estimator = load_estimator(args.repo, args.checkpoint, args.device)

    seconds = {m: 0.0 for m in MODES}
    for n, track in enumerate(sample):
        track["chips"] = []
        for name in track["frames"]:
            path = args.folder / "chips" / "bbox_test" / name[:4] / name
            if not path.is_file():
                sys.exit(f"error: missing chip {path}")
            image = np.asarray(Image.open(path).convert("RGB"))
            h, w = image.shape[:2]
            chip = {"name": name, "height": h, "width": w}
            chip["vit_k"], chip["vit_s"] = vitpose(processor, pose_model, image, args.device)
            for mode in MODES:
                start = time.time()
                chip[mode] = sam3d(estimator, image, mode)
                seconds[mode] += time.time() - start
                chip[f"oks_{mode}"] = oks(chip[mode]["k2"][MHR70_TO_COCO17], chip["vit_k"], chip["vit_s"], h * w)
            track["chips"].append(chip)
        if n % 10 == 0:
            print(f"{n + 1}/{len(sample)} tracklets", flush=True)

    chips = sum(len(t["chips"]) for t in sample)
    report = {"tracklets": len(sample), "chips": chips,
              "seconds_per_chip": {m: round(seconds[m] / chips, 3) for m in MODES}, "bins": {}}
    for low, high in sorted({tuple(t["bin"]) for t in sample}):
        tracks = [t for t in sample if tuple(t["bin"]) == (low, high)]
        all_chips = [c for t in tracks for c in t["chips"]]
        entry = {"tracklets": len(tracks), "height_p50": int(np.median([c["height"] for c in all_chips])),
                 "vitpose_score_p50": round(float(np.median([c["vit_s"].mean() for c in all_chips])), 3),
                 # Same body, two decoders: how far apart are their 2D points, in share of the crop height?
                 "body_vs_full_px_share_p50": round(float(np.median([np.linalg.norm(c["body"]["k2"] - c["full"]["k2"], axis=1).mean()
                                                                    / c["height"] for c in all_chips])), 4)}
        for mode in MODES:
            entry[mode] = summarize(tracks, mode)
        report["bins"][f"{low}-{high}"] = entry
        print(f"{low}-{high}", json.dumps(entry), flush=True)
    (args.folder / "results.json").write_text(json.dumps(report, indent=1))

    # Sheet: per bin, the first chip of 6 tracklets, scaled to 256 px high.
    rows = []
    for low, high in sorted({tuple(t["bin"]) for t in sample}):
        cells = []
        for t in [t for t in sample if tuple(t["bin"]) == (low, high)][:6]:
            c = t["chips"][0]
            path = args.folder / "chips" / "bbox_test" / c["name"][:4] / c["name"]
            image = cv2.imread(str(path))
            scale = 256 / c["height"]
            image = cv2.resize(image, (max(int(c["width"] * scale), 1), 256))
            for points, color in ((c["body"]["k2"][MHR70_TO_COCO17], (0, 220, 0)), (c["vit_k"], (0, 0, 255))):
                for x, y in points * scale:
                    cv2.circle(image, (int(x), int(y)), 3, color, -1)
            cv2.putText(image, f"{c['height']}px {c['oks_body']:.2f}", (2, 250), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cells.append(cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT))
        row = np.hstack(cells)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
    cv2.imwrite(str(args.folder / "sheet.jpg"), sheet)
    print("SMALL_CROPS_OK")


if __name__ == "__main__":
    main()
