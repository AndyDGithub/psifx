"""Picks the frames of a video to annotate by hand and pre-annotates them, to correct in CVAT.

- Frames: ``--count`` frames evenly spaced over the whole video. Even spacing keeps the share of easy
  and hard frames (profile, occlusion by the table, cuts) of the video, so the measured accuracy is
  the one a user gets on it.
- Pre-annotation: every person found by the RT-DETR detector, with the 17 COCO keypoints of
  ViTPose++ (``--model``, by default the huge variant, which psifx does not run). A different model
  than the ones measured later limits the bias towards them: a corrector tends to keep a point that is
  almost right. A keypoint under ``--min_score`` is left out (COCO visibility 0), to add by hand if
  it can be placed; the others are marked visible (2), to switch to occluded (1) when hidden.

Writes into ``out_dir``: ``images/<video>_<frame>.jpg``, ``frames.csv`` (file, frame, time) and
``annotations/person_keypoints_default.json`` (COCO Keypoints, the CVAT import format), zipped as
``preannotations_coco.zip``.

    python research/pose_annotation/prepare.py example/data/ExtraitTalkShow.mp4 \\
        .e2e_outputs/extrait_talkshow/pose_annotation --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from psifx.video.pose.vitpose.tool import ViTPosePoseEstimationTool

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]
# COCO skeleton, 1-based keypoint indices.
COCO_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                 [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


class ViTPosePlusTool(ViTPosePoseEstimationTool):
    """ViTPose++ is a mixture of experts, one head per training dataset: index 0 is COCO."""

    @torch.inference_mode()
    def predict(self, image: np.ndarray):
        boxes_xyxy = self.detect(image)
        if len(boxes_xyxy) == 0:
            return []
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        inputs = self.pose_processor(image, boxes=[boxes_xywh], return_tensors="pt").to(self.device)
        dataset_index = torch.zeros(len(boxes_xywh), dtype=torch.long, device=self.device)
        outputs = self.pose_model(**inputs, dataset_index=dataset_index)
        results = self.pose_processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
        return [(r["keypoints"].cpu().numpy()[:17], r["scores"].cpu().numpy()[:17], box)
                for r, box in zip(results, boxes_xyxy)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--count", type=int, default=200, help="frames to annotate")
    parser.add_argument("--model", default="usyd-community/vitpose-plus-huge")
    parser.add_argument("--min_score", type=float, default=0.3, help="keypoints below are left out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        sys.exit(f"error: missing {args.video}")
    archive = args.out_dir / "preannotations_coco.zip"
    if archive.exists() and not args.overwrite:
        sys.exit(f"error: {archive} exists, pass --overwrite to redo it (it would replace hand corrections "
                 f"made from it only if they were saved there)")
    capture = cv2.VideoCapture(str(args.video))
    n_frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    if n_frames <= 0 or fps <= 0:
        sys.exit(f"error: cannot read the frame count of {args.video}")
    # The middle of each of ``count`` equal intervals: the container's frame count can be one too
    # high, so the very last frame is never asked for.
    frames = np.unique(((np.arange(args.count) + 0.5) * n_frames / args.count).astype(int))

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    plus = "plus" in args.model
    tool = (ViTPosePlusTool if plus else ViTPosePoseEstimationTool)(model_name=args.model, device=args.device,
                                                                   verbose=False)
    images, annotations, rows = [], [], []
    for image_id, frame in enumerate(tqdm(frames, desc="Pre-annotating"), 1):
        # Seeking frame by frame is exact with OpenCV's FFmpeg backend for H.264 (it decodes from the
        # previous keyframe); the frame index is also written to frames.csv to check it.
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, bgr = capture.read()
        if not ok:
            sys.exit(f"error: cannot read frame {frame} of {args.video}")
        name = f"{args.video.stem}_{frame:06d}.jpg"
        cv2.imwrite(str(images_dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = bgr.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": w, "height": h})
        rows.append({"file": name, "frame": int(frame), "time_s": round(frame / fps, 3)})
        for keypoints, scores, box in tool.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
            flat = []
            for (x, y), s in zip(keypoints, scores):
                flat += [float(x), float(y), 2] if s >= args.min_score else [0.0, 0.0, 0]
            x1, y1, x2, y2 = (float(v) for v in box)
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "keypoints": flat, "num_keypoints": sum(v > 0 for v in flat[2::3]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "segmentation": [],
            })
    capture.release()

    coco = {
        "info": {"description": f"pre-annotation of {args.video.name} with {args.model}"},
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "", "keypoints": COCO_KEYPOINTS,
                        "skeleton": COCO_SKELETON}],
    }
    (args.out_dir / "annotations").mkdir(exist_ok=True)
    json_path = args.out_dir / "annotations" / "person_keypoints_default.json"
    json_path.write_text(json.dumps(coco))
    with zipfile.ZipFile(archive, "w") as z:
        z.write(json_path, "annotations/person_keypoints_default.json")
    with open(args.out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "frame", "time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(images)} frames, {len(annotations)} people -> {archive}")


if __name__ == "__main__":
    main()
