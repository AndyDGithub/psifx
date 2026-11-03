"""Accuracy and speed of pose models (``models.py`` specs) on the hand-corrected frames.

For each spec, in order: load, warm up on ``--warmup`` frames (a compiled model compiles there), then
time every annotated frame one by one, as a video would go, from the RGB frame to the people found
(detector included for the top-down models; CUDA synchronized). The predictions are scored with
``evaluate.py`` (OKS, found, PCK, phantom), and ``ms_per_frame``, ``load_s`` (load and warm-up) and
``peak_gpu_mb`` (PyTorch allocations only: ONNX Runtime memory is not counted) are added.

Each model is written to ``--output`` as soon as it is done, and a spec already there is skipped
(``--overwrite`` redoes it), so a crash or an out-of-memory error loses one model, not the run.

    python research/pose_annotation/compare.py .e2e_outputs/extrait_talkshow/pose_annotation \\
        vitpose:usyd-community/vitpose-plus-base vitpose:usyd-community/vitpose-plus-base@fp16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402  (imports torch first, see models.py)
import torch  # noqa: E402
from evaluate import evaluate, load_ground_truth  # noqa: E402


def run(spec: str, images: list[Path], device: str, warmup: int) -> tuple[dict, dict]:
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    frames = {p.name: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in images}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predict = models.load(spec, device)
    for image in list(frames.values())[:warmup]:
        predict(image)
    sync()
    load_s = time.perf_counter() - start
    predictions, times = {}, []
    for name, image in frames.items():
        start = time.perf_counter()
        predictions[name] = predict(image)
        sync()
        times.append(time.perf_counter() - start)
    for people in predictions.values():
        for p in people:
            if p.shape != (models.N, 3):
                raise ValueError(f"{spec}: a person has shape {p.shape}, expected ({models.N}, 3)")
    speed = {"ms_per_frame": round(1000 * float(np.mean(times)), 1),
             "ms_per_frame_median": round(1000 * float(np.median(times)), 1), "load_s": round(load_s, 1),
             "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else None}
    return predictions, speed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder of prepare.py and fetch_annotations.py")
    parser.add_argument("specs", nargs="+", help="models.py specs")
    parser.add_argument("--min_score", type=float, default=0.3, help="a point under it counts as not predicted")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None, help="default: out_dir/pose_models.json")
    parser.add_argument("--overwrite", action="store_true", help="redo the specs already in the output")
    args = parser.parse_args()

    gt_path, frames_path = args.out_dir / "ground_truth.json", args.out_dir / "frames.csv"
    for path in (gt_path, frames_path):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    for spec in args.specs:
        models.parse(spec)  # fails on a malformed spec before any model is loaded
    truth = load_ground_truth(gt_path)
    with open(frames_path, newline="") as f:
        images = [args.out_dir / "images" / row["file"] for row in csv.DictReader(f)]
    if missing := [p for p in images if not p.is_file()]:
        sys.exit(f"error: {len(missing)} images missing, first {missing[0]}")
    output = args.output or args.out_dir / "pose_models.json"
    results = json.loads(output.read_text()) if output.exists() else {}

    failed = []
    for spec in args.specs:
        if spec in results and not args.overwrite:
            print(f"{spec}: already measured, skipped")
            continue
        try:
            predictions, speed = run(spec, images, args.device, args.warmup)
        except Exception:
            # One model failing (out of memory, missing package) must not stop the others.
            traceback.print_exc()
            failed.append(spec)
            continue
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        results[spec] = {**evaluate(predictions, truth, args.min_score), **speed, "min_score": args.min_score,
                         "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"}
        output.write_text(json.dumps(results, indent=2))
        r = results[spec]
        print(f"{spec:55s} OKS {r['oks']:.3f}  found {r['found']:.3f}  PCK@0.1 {r['pck@0.1']:.3f}  "
              f"phantom {r['phantom']}  {r['ms_per_frame']} ms/frame  {r['peak_gpu_mb']} MB")
    if failed:
        sys.exit(f"error: failed {failed}")


if __name__ == "__main__":
    main()
