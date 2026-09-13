"""Standalone Sapiens2 pose inference script -- run via subprocess in the
`sapiens2-env` (Python 3.12) conda env, from Sapiens2Backend in backends.py.

Not imported by the Python 3.11 psifx-env benchmark process: Sapiens2
requires Python >=3.12 (see sapiens2/pyproject.toml), incompatible with the
rest of this benchmark and with psifx itself. Bridged via a JSON file rather
than an in-process import -- the same "separate runtime, invoked as an
external process" pattern psifx already uses for OpenFace.

Adapted from sapiens2/sapiens/pose/tools/vis/vis_pose.py's process_one_image
(detector + pose inference), stripped of visualization output and with
per-image timing added, since the benchmark needs speed numbers.

Usage:
  python sapiens2_infer.py \
    --sapiens-root /home/psifx/sapiens2 \
    --det-checkpoint $SAPIENS_CHECKPOINT_ROOT/detector/detr-resnet-101-dc5 \
    --config configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py \
    --checkpoint $SAPIENS_CHECKPOINT_ROOT/pose/sapiens2_0.4b_pose.safetensors \
    --images-dir path/to/images \
    --output results.json \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sapiens-root", required=True, help="Path to the cloned sapiens2 repo")
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Relative to --sapiens-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    args = parser.parse_args()

    sys.path.insert(0, args.sapiens_root)
    sys.path.insert(0, os.path.join(args.sapiens_root, "sapiens", "pose"))
    os.chdir(os.path.join(args.sapiens_root, "sapiens", "pose"))  # configs use relative imports

    from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
    from sapiens.pose.evaluators import nms
    from sapiens.pose.models import init_model
    from transformers import DetrForObjectDetection, DetrImageProcessor

    device = args.device
    det_processor = DetrImageProcessor.from_pretrained(args.det_checkpoint)
    det_model = DetrForObjectDetection.from_pretrained(args.det_checkpoint).eval().to(device)

    model = init_model(args.config, args.checkpoint, device=device)
    # init_model() loads in fp32 -- even the 0.4B model OOMs on an 8GB
    # consumer GPU at 1024x768 in fp32 (confirmed: "Tried to allocate 1.22
    # GiB" with only 214MB free after loading). bf16 is well-supported on
    # this hardware (Ada Lovelace) and roughly halves memory use; inputs
    # are cast to match right before each forward pass below.
    if device.startswith("cuda"):
        model = model.to(torch.bfloat16)
    if model.cfg.num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(dict(from_file="configs/_base_/keypoints308.py"))
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    def detect_persons(image_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = det_processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = det_model(**inputs)
        target_sizes = torch.tensor([image_rgb.shape[:2]], device=device)
        results = det_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.bbox_thr
        )[0]
        person_mask = results["labels"] == 1  # COCO person
        boxes = results["boxes"][person_mask].cpu().numpy()
        scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)
        bboxes = np.concatenate([boxes, scores], axis=1)
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
        return bboxes

    def run_forward_batch(inputs_list, data_samples_list):
        """One forward pass over a list of per-person crops. All person
        boxes in an image are batched together for throughput, but larger
        checkpoints (0.8B/1B) can OOM on images with many people (confirmed:
        26-person COCO image OOMs 0.8B/1B in bf16 even with
        expandable_segments, while 1-2 person images are fine on the same
        GPU) -- caller retries with smaller sub-batches on OOM, mirroring
        psifx's own SAM3 tool chunk-size retry pattern
        (psifx/video/tracking/sam3/tool.py) rather than crashing the whole
        run over one crowded frame.
        """
        inputs = torch.cat(inputs_list, dim=0)
        if device.startswith("cuda"):
            inputs = inputs.to(torch.bfloat16)
        with torch.no_grad():
            pred = model(inputs)
        pred = pred.float().cpu().numpy()

        keypoints, keypoint_scores = [], []
        for i, data_samples in enumerate(data_samples_list):
            keypoints_i, keypoint_scores_i = model.codec.decode(pred[i])
            input_size = data_samples["meta"]["input_size"]
            bbox_center = data_samples["meta"]["bbox_center"]
            bbox_scale = data_samples["meta"]["bbox_scale"]
            keypoints_i = keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
            keypoints.append(keypoints_i[0])
            keypoint_scores.append(keypoint_scores_i[0])
        return keypoints, keypoint_scores

    def run_forward_with_oom_retry(inputs_list, data_samples_list):
        try:
            result = run_forward_batch(inputs_list, data_samples_list)
        except torch.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            n = len(inputs_list)
            if n <= 1:
                raise
            mid = n // 2
            print(f"CUDA OOM on a {n}-person batch; retrying as {mid} + {n - mid}", file=sys.stderr)
            kp_a, sc_a = run_forward_with_oom_retry(inputs_list[:mid], data_samples_list[:mid])
            # Larger checkpoints (1B) OOM again on the second half otherwise:
            # torch's caching allocator doesn't release the first half's
            # activations just because they're out of scope -- explicit
            # empty_cache() between sub-batches is needed, not just on
            # failure (confirmed: single-person batches still OOM'd without
            # this, deep in the recursion, on an already near-full GPU).
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            kp_b, sc_b = run_forward_with_oom_retry(inputs_list[mid:], data_samples_list[mid:])
            return kp_a + kp_b, sc_a + sc_b
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    def process_one_image(image_bgr: np.ndarray):
        bboxes = detect_persons(image_bgr)
        if len(bboxes) == 0:
            return [], [], bboxes

        inputs_list, data_samples_list = [], []
        for bbox in bboxes:
            data_info = dict(img=image_bgr)
            data_info["bbox"] = bbox[None]
            data_info["bbox_score"] = np.ones(1, dtype=np.float32)
            data = model.pipeline(data_info)
            data = model.data_preprocessor(data)
            inputs_list.append(data["inputs"])
            data_samples_list.append(data["data_samples"])

        keypoints, keypoint_scores = run_forward_with_oom_retry(inputs_list, data_samples_list)
        return keypoints, keypoint_scores, bboxes

    image_names = sorted(
        n for n in os.listdir(args.images_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    records = []
    for name in image_names:
        image_bgr = cv2.imread(os.path.join(args.images_dir, name))
        if image_bgr is None:
            continue
        t0 = time.perf_counter()
        try:
            keypoints, keypoint_scores, bboxes = process_one_image(image_bgr)
        except torch.OutOfMemoryError:
            # Splitting down to single-person batches (run_forward_with_oom_retry)
            # still wasn't enough on a very crowded image (confirmed: a
            # 26-person frame OOMs 0.8B/1B in bf16 even one person at a
            # time, once the model + detector already occupy most of an
            # 8GB GPU) -- a genuine capacity wall, not a batching bug. Skip
            # this one image (recorded with zero instances) rather than
            # losing every other image's results in the same run.
            print(f"{name}: CUDA OOM even at batch size 1, skipping this image", file=sys.stderr)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            keypoints, keypoint_scores, bboxes = [], [], []
        dt = time.perf_counter() - t0

        instances = []
        for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes):
            instances.append(
                {
                    "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                    "keypoints": np.asarray(kpts, dtype=float).tolist(),
                    "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
                }
            )
        records.append({"image_name": name, "inference_seconds": dt, "instances": instances})
        print(f"{name}: {len(instances)} person(s), {dt * 1000:.1f} ms", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump({"frames": records}, f)


if __name__ == "__main__":
    main()
