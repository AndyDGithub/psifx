"""Standalone DETRPose inference script -- run via subprocess in the
`detrpose-env` conda env, from DETRPoseBackend in backends.py.

Not imported by the benchmark process directly: DETRPose's requirements.txt
pins numpy==1.23.5 with unpinned torch/torchvision, a real risk of breaking
psifx-env's own numpy-dependent packages (mediapipe/rtmlib/opencv) if
installed into the same env. Bridged via a JSON file, same "separate
runtime, invoked as an external process" pattern already used for Sapiens2
and (ultimately) OpenFace.

Adapted from DETRPose's tools/inference/torch_inf.py (process_image), with
visualization dropped and per-image timing added.

Usage:
  python detrpose_infer.py \
    --detrpose-root /home/psifx/detrpose \
    --config configs/detrpose/detrpose_hgnetv2_m.py \
    --checkpoint /home/psifx/detrpose/detrpose_hgnetv2_m.pth \
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

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detrpose-root", required=True, help="Path to the cloned DETRPose repo")
    parser.add_argument("--config", required=True, help="Relative to --detrpose-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.5)
    args = parser.parse_args()

    sys.path.insert(0, args.detrpose_root)
    os.chdir(args.detrpose_root)  # configs use relative imports, like torch_inf.py itself

    from src.core import LazyConfig, instantiate

    device = args.device
    cfg = LazyConfig.load(args.config)
    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    infer_model = Model().to(device).eval()
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    image_names = sorted(
        n for n in os.listdir(args.images_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    records = []
    for name in image_names:
        path = os.path.join(args.images_dir, name)
        im_pil = Image.open(path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            scores, labels, keypoints = infer_model(im_data, orig_size)
        dt = time.perf_counter() - t0

        # One score per PERSON INSTANCE, not per keypoint -- postprocessor
        # doesn't expose a per-joint confidence (verified in torch_inf.py:
        # only scores/labels/keypoints are returned, keypoints is (N, K, 2)
        # with no separate visibility channel).
        scores = scores[0].detach().cpu().numpy()
        keypoints = keypoints[0].detach().cpu().numpy()
        idx = scores > args.score_thr
        valid_keypoints = keypoints[idx]
        valid_scores = scores[idx]

        instances = []
        for kpts, inst_score in zip(valid_keypoints, valid_scores):
            instances.append(
                {
                    "bbox": None,
                    "keypoints": np.asarray(kpts, dtype=float).tolist(),
                    "keypoint_scores": [1.0] * len(kpts),
                    "instance_score": float(inst_score),
                }
            )
        records.append({"image_name": name, "inference_seconds": dt, "instances": instances})
        print(f"{name}: {len(instances)} person(s), {dt * 1000:.1f} ms", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump({"frames": records}, f)


if __name__ == "__main__":
    main()
