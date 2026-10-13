"""Why is ViTPose less accurate in psifx than the literature suggests? Ablation, one factor at a time.

In psifx (CLI, 50 COCO images), ViTPose-B gets OKS 0.850, below RTMPose-m (0.873), while
the research benchmark gave ViTPose 0.890 -- but there ViTPose used ground-truth boxes and
RTMPose its own detector. This script isolates each factor on the same 50 images:

- frames: decoded from the H.264 video psifx reads, or the raw letterboxed images;
- person boxes: RT-DETR (psifx ViTPose), YOLOX (rtmlib RTMPose), or ground truth (oracle);
- person choice: largest box (what psifx does) or the box that best overlaps the annotation;
- flip test (average with the horizontally flipped image), used by published ViTPose numbers;
- model size: ViTPose-B simple (psifx default), ViTPose-B classic, ViTPose++ S/B/L/H.

RTMPose and RTMO are run the same way for reference. Per-image OKS/PCK go to
results/vitpose_gap_per_image.csv, and paired bootstrap confidence intervals of the key
differences are printed (50 images is small: differences under ~0.02 may be noise).

Usage: python vitpose_gap.py [--device cuda] [--bootstrap 10000]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import common  # noqa: F401  (puts the repository root on sys.path)
from tests.pose_eval.coco import NUM_COCO_KEYPOINTS, PersonPose
from tests.pose_eval.metrics import object_keypoint_similarity, pck
from tests.pose_eval.psifx_eval import GroundTruth, build_coco_video, letterbox, load_coco_samples

HERE = Path(__file__).resolve().parent
COCO_DIR = HERE / "data" / "coco_subset"
OUT_CSV = HERE / "results" / "vitpose_gap_per_image.csv"
SIZE = 640
# COCO-17 left/right keypoint pairs, for the flip test.
FLIP_PAIRS = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])


@dataclass
class Sample:
    raw: np.ndarray  # letterboxed RGB frame, never encoded
    decoded: np.ndarray  # same frame after the H.264 round trip psifx goes through
    gt: GroundTruth
    gt_box: np.ndarray  # [x1, y1, x2, y2] in letterboxed coordinates


def load_samples() -> List[Sample]:
    """Letterboxed frames, their H.264-decoded version and the ground truth, in the same order."""
    import tempfile
    from psifx.io.video import VideoReader

    coco = load_coco_samples(COCO_DIR / "annotations.json", COCO_DIR / "images")
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "coco.mp4"
        gts = build_coco_video(coco, video, size=SIZE)
        with VideoReader(path=video) as reader:
            decoded = [frame.copy() for frame in reader]
    samples = []
    for s, gt, dec in zip(coco, gts, decoded):
        frame, scale, ox, oy = letterbox(np.asarray(Image.open(s["image_path"]).convert("RGB")), SIZE)
        x, y, w, h = s["annotation"]["bbox"]
        box = np.array([x * scale + ox, y * scale + oy, (x + w) * scale + ox, (y + h) * scale + oy])
        samples.append(Sample(raw=frame, decoded=dec, gt=gt, gt_box=box))
    return samples


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select(boxes: np.ndarray, sample: Sample, how: str) -> Optional[np.ndarray]:
    """'largest' (psifx behaviour) or 'best' (box overlapping the annotated person the most)."""
    if len(boxes) == 0:
        return None
    if how == "largest":
        return boxes[int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))]
    return boxes[int(np.argmax([iou(b, sample.gt_box) for b in boxes]))]


def score(keypoints: Optional[np.ndarray], gt: GroundTruth) -> tuple[float, float]:
    """OKS and PCK of one prediction; a missed person scores 0, as in psifx_eval."""
    if keypoints is None:
        return 0.0, 0.0
    pred = PersonPose(keypoints=np.asarray(keypoints, dtype=np.float64)[:NUM_COCO_KEYPOINTS],
                      scores=np.ones(NUM_COCO_KEYPOINTS))
    oks = object_keypoint_similarity(pred, gt.keypoints, gt.visible, gt.area) or 0.0
    return oks, pck(pred, gt.keypoints, gt.visible, gt.bbox_diagonal) or 0.0


class ViTPose:
    """ViTPose on given boxes, with an optional flip test (transformers only flips the heatmaps back)."""

    def __init__(self, name: str, device: str):
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.name, self.device = name, device
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = VitPoseForPoseEstimation.from_pretrained(name).to(device).eval()
        # ViTPose++ has one expert head per training dataset: 0 is COCO.
        self.moe = "plus" in name

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, box_xyxy: np.ndarray, flip: bool = False) -> np.ndarray:
        box = np.array([[box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]]])
        inputs = self.processor(image, boxes=[box], return_tensors="pt").to(self.device)
        extra = {"dataset_index": torch.tensor([0], device=self.device)} if self.moe else {}
        heatmaps = self.model(**inputs, **extra).heatmaps
        if flip:
            flipped = self.model(pixel_values=inputs["pixel_values"].flip(-1), flip_pairs=FLIP_PAIRS, **extra).heatmaps
            heatmaps = (heatmaps + flipped) / 2
        outputs = type("Out", (), {"heatmaps": heatmaps.float()})()
        result = self.processor.post_process_pose_estimation(outputs, boxes=[box])[0][0]
        return result["keypoints"].cpu().numpy()[:NUM_COCO_KEYPOINTS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    from psifx.video.pose.onnx_optimization import preload_cuda_libraries
    from psifx.video.pose.vitpose.tool import ViTPosePoseEstimationTool
    from rtmlib import Body

    if args.device.startswith("cuda"):
        preload_cuda_libraries()
    samples = load_samples()
    print(f"{len(samples)} images")

    # Detectors, computed once per frame source.
    rtdetr = ViTPosePoseEstimationTool(device=args.device, verbose=False)
    rtm = Body(mode="balanced", backend="onnxruntime", device=args.device)
    rtmo = Body(pose="rtmo", mode="balanced", backend="onnxruntime", device=args.device)
    boxes: Dict[str, Dict[str, list]] = {}
    for source in ("raw", "decoded"):
        frames = [getattr(s, source) for s in samples]
        boxes[source] = {
            "rtdetr": [rtdetr.detect(f) for f in frames],
            "yolox": [np.asarray(rtm.det_model(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)))[:, :4].reshape(-1, 4)
                      for f in frames],
        }

    def box_for(i: int, source: str, detector: str, how: str) -> Optional[np.ndarray]:
        if detector == "gt":
            return samples[i].gt_box
        return select(np.asarray(boxes[source][detector][i]).reshape(-1, 4), samples[i], how)

    def rtmpose(image, box):
        kp, _ = rtm.pose_model(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), bboxes=[box.tolist()])
        return np.asarray(kp)[0][:NUM_COCO_KEYPOINTS]

    def rtmo_run(i, source, how):
        kp, _ = rtmo(cv2.cvtColor(getattr(samples[i], source), cv2.COLOR_RGB2BGR))
        kp = np.asarray(kp).reshape(-1, NUM_COCO_KEYPOINTS, 2)
        if len(kp) == 0:
            return None
        person_boxes = np.concatenate([kp.min(axis=1), kp.max(axis=1)], axis=1)
        return kp[int(np.argmax([iou(b, samples[i].gt_box) for b in person_boxes]))] if how == "best" else \
            kp[int(np.argmax((person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])))]

    vit_b = ViTPose("usyd-community/vitpose-base-simple", args.device)
    # (config name, function(i) -> keypoints or None)
    configs: List[tuple[str, Callable[[int], Optional[np.ndarray]]]] = []

    def pose_cfg(name, model, source, detector, how="largest", flip=False):
        def run(i):
            box = box_for(i, source, detector, how)
            return None if box is None else model(getattr(samples[i], source), box, flip) if model is not rtmpose \
                else rtmpose(getattr(samples[i], source), box)
        configs.append((name, run))

    # ViTPose-B simple: from the psifx setting to the published setting, one factor at a time.
    pose_cfg("vitB|video|rtdetr|largest", vit_b, "decoded", "rtdetr")
    pose_cfg("vitB|raw|rtdetr|largest", vit_b, "raw", "rtdetr")
    pose_cfg("vitB|raw|rtdetr|best", vit_b, "raw", "rtdetr", "best")
    pose_cfg("vitB|raw|yolox|best", vit_b, "raw", "yolox", "best")
    pose_cfg("vitB|raw|gt", vit_b, "raw", "gt")
    pose_cfg("vitB|raw|gt|flip", vit_b, "raw", "gt", flip=True)
    pose_cfg("vitB|raw|yolox|best|flip", vit_b, "raw", "yolox", "best", flip=True)
    # RTMPose-m (body7) under the same conditions.
    pose_cfg("rtmpose|video|yolox|largest", rtmpose, "decoded", "yolox")
    pose_cfg("rtmpose|raw|yolox|largest", rtmpose, "raw", "yolox")
    pose_cfg("rtmpose|raw|yolox|best", rtmpose, "raw", "yolox", "best")
    pose_cfg("rtmpose|raw|rtdetr|best", rtmpose, "raw", "rtdetr", "best")
    pose_cfg("rtmpose|raw|gt", rtmpose, "raw", "gt")
    configs.append(("rtmo|video|largest", lambda i: rtmo_run(i, "decoded", "largest")))
    configs.append(("rtmo|raw|best", lambda i: rtmo_run(i, "raw", "best")))

    rows = {}
    for name, run in configs:
        rows[name] = [score(run(i), samples[i].gt) for i in range(len(samples))]
        print(f"{name:32s} OKS {np.mean([r[0] for r in rows[name]]):.3f}  PCK {np.mean([r[1] for r in rows[name]]):.3f}")
    del vit_b
    torch.cuda.empty_cache()

    # Model size, all with ground-truth boxes on raw frames (pose model quality only).
    for checkpoint in ["usyd-community/vitpose-base", "usyd-community/vitpose-plus-small",
                       "usyd-community/vitpose-plus-base", "usyd-community/vitpose-plus-large",
                       "usyd-community/vitpose-plus-huge"]:
        try:
            model = ViTPose(checkpoint, args.device)
        except OSError as error:  # download failure: report it and keep the other results
            print(f"skipped {checkpoint}: {error}")
            continue
        for flip in (False, True):
            name = f"{checkpoint.split('/')[-1]}|raw|gt" + ("|flip" if flip else "")
            rows[name] = [score(model(s.raw, s.gt_box, flip), s.gt) for s in samples]
            print(f"{name:32s} OKS {np.mean([r[0] for r in rows[name]]):.3f}  PCK {np.mean([r[1] for r in rows[name]]):.3f}")
        del model
        torch.cuda.empty_cache()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "image", "oks", "pck"])
        for name, values in rows.items():
            for i, (o, p) in enumerate(values):
                writer.writerow([name, i, round(o, 5), round(p, 5)])

    # Paired bootstrap on the same images: is a difference larger than the sampling noise?
    rng = np.random.default_rng(0)
    pairs = [
        ("vitB|video|rtdetr|largest", "rtmpose|video|yolox|largest", "psifx setting: ViTPose vs RTMPose"),
        ("vitB|raw|gt", "rtmpose|raw|gt", "same GT boxes: pose model only"),
        ("vitB|raw|yolox|best", "rtmpose|raw|yolox|best", "same YOLOX boxes"),
        ("vitB|raw|rtdetr|best", "vitB|raw|yolox|best", "ViTPose: RT-DETR vs YOLOX boxes"),
        ("vitB|raw|rtdetr|largest", "vitB|raw|rtdetr|best", "ViTPose: largest vs best box"),
        ("vitB|video|rtdetr|largest", "vitB|raw|rtdetr|largest", "ViTPose: video vs raw frames"),
        ("vitB|raw|gt|flip", "vitB|raw|gt", "ViTPose: flip test"),
    ]
    print("\nPaired differences in mean OKS (A - B), 95% bootstrap CI:")
    for a, b, label in pairs:
        diff = np.array([x[0] for x in rows[a]]) - np.array([x[0] for x in rows[b]])
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(args.bootstrap)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:38s} {diff.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
