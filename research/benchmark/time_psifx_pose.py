"""Steady-state speed of a psifx COCO-17 pose tool, to measure inference optimizations.

``run_psifx_cli_benchmark.py`` reports wall time including model loading (and,
with ``--compile``, compilation), which hides per-frame gains. This script
loads the tool once, runs warm-up passes over the COCO frames, then times
several more passes, synchronizing CUDA so GPU work is fully counted.

Frames go through ``predict_video`` (one call per pass), so tools running
out of process (DETRPose) are timed as the CLI runs them; such tools may also
expose ``last_inference_seconds`` (model time per frame, reported separately).

Usage:
    python time_psifx_pose.py --tool vitpose --device cuda
    python time_psifx_pose.py --tool vitpose --device cuda --kwargs '{"fp16": true, "compile": true}'
    python time_psifx_pose.py --tool rtmo --device cuda --kwargs '{"precision": "fp16"}'
"""

from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np
import torch
from PIL import Image

from psifx_eval import letterbox, load_coco_samples
from run_psifx_cli_benchmark import COCO_SOURCES

# tool name -> (module, class) of its psifx implementation. Imported lazily:
# each model only exists on its own branch.
TOOLS = {
    "rtmpose": ("psifx.video.pose.rtmpose.tool", "RTMPosePoseEstimationTool"),
    "rtmo": ("psifx.video.pose.rtmo.tool", "RTMOPoseEstimationTool"),
    "vitpose": ("psifx.video.pose.vitpose.tool", "ViTPosePoseEstimationTool"),
    "detrpose": ("psifx.video.pose.detrpose.tool", "DETRPosePoseEstimationTool"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, choices=sorted(TOOLS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kwargs", default="{}", help="JSON keyword arguments for the tool")
    parser.add_argument("--coco", default="benchmark", choices=sorted(COCO_SOURCES))
    parser.add_argument("--warmup", type=int, default=2, help="untimed passes over the frames")
    parser.add_argument("--passes", type=int, default=3, help="timed passes over the frames")
    args = parser.parse_args()

    coco_dir = COCO_SOURCES[args.coco]
    samples = load_coco_samples(coco_dir / "annotations.json", coco_dir / "images")
    frames = [letterbox(np.asarray(Image.open(s["image_path"]).convert("RGB")), 640)[0] for s in samples]

    module, cls = TOOLS[args.tool]
    tool = getattr(importlib.import_module(module), cls)(device=args.device, verbose=False, **json.loads(args.kwargs))

    def run_pass():
        for _ in tool.predict_video(frames):
            pass
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        run_pass()
    warmup_seconds = time.perf_counter() - t0

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.passes):
        run_pass()
    seconds = time.perf_counter() - t0

    n = len(frames) * args.passes
    model_seconds = getattr(tool, "last_inference_seconds", None)
    print(json.dumps({
        "tool": args.tool,
        "kwargs": json.loads(args.kwargs),
        "ms_per_frame": round(1000 * seconds / n, 2),
        "fps": round(n / seconds, 2),
        "warmup_seconds": round(warmup_seconds, 1),
        "model_ms_per_frame": round(1000 * float(np.mean(model_seconds)), 2) if model_seconds else None,
        # torch allocator only: misses onnxruntime (RTMPose/RTMO) and subprocess (DETRPose) memory.
        "peak_torch_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else None,
    }))


if __name__ == "__main__":
    main()
