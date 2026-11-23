"""Why do two runs of SAM 3D Body give different hands on a few frames?

The same frames are computed several times in each setting:
- ``full`` (hand decoder), backbone in bfloat16 as in the checkpoint config;
- ``full`` with the backbone in float32;
- ``body`` (no hand decoder), once.
Frames: those where the hands of two earlier dense runs differ by more than ``--outlier_mm``, plus random ones.

Hypothesis: in ``full`` mode, the hand decoder result is kept or dropped per hand by hard thresholds (wrist angle,
box size, keypoints in the box, wrist distance). A tiny numeric change near a threshold flips the choice, and the
hand jumps. If so, runs disagree mostly in bfloat16, and a disagreeing run's hand is close to the ``body`` hand.

Writes ``<output>`` (JSON) with per setting: share of person-frames where runs differ by more than ``--outlier_mm``.

    python research/sam3d_body/repeat_study.py .e2e_outputs/extrait_talkshow/pipeline \\
        --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_keyframes import Model  # noqa: E402
from run_tracks import load_estimator  # noqa: E402

HANDS = np.arange(21, 63)
LEFT_HIP, RIGHT_HIP = 9, 10


def hands(k3: np.ndarray) -> np.ndarray:
    """Hand keypoints around the pelvis, so that a global shift does not count."""
    return k3[HANDS] - k3[[LEFT_HIP, RIGHT_HIP]].mean(0)


def max_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(hands(a) - hands(b), axis=1).max() * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4, masks/ and the two dense runs")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs", nargs=2, default=["sam3d_body", "sam3d_params"], help="two earlier dense runs")
    parser.add_argument("--outlier_mm", type=float, default=50)
    parser.add_argument("--random", type=int, default=150, help="random person-frames added")
    parser.add_argument("--repeats", type=int, default=3, help="runs per full setting")
    parser.add_argument("--output", type=Path, help="default: <pipeline>/repeat_study.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    masks = {p.stem: p for p in sorted((args.pipeline / "masks").glob("*.mp4"))}
    # Arrays read once: an npz decompresses the whole array at each access.
    earlier = [{p: {k: np.load(args.pipeline / run / f"{p}.npz")[k] for k in ("frames", "keypoints_3d")} for p in masks}
               for run in args.runs]
    rng = np.random.default_rng(0)
    chosen = []  # (person, frame, outlier)
    for person in masks:
        a, b = earlier[0][person], earlier[1][person]
        index = {int(f): i for i, f in enumerate(b["frames"])}
        common = [(int(f), i, index[int(f)]) for i, f in enumerate(a["frames"]) if int(f) in index]
        outliers = [f for f, i, j in common if max_mm(a["keypoints_3d"][i], b["keypoints_3d"][j]) > args.outlier_mm]
        others = [f for f, _, _ in common if f not in set(outliers)]
        picked = rng.choice(others, size=min(args.random // len(masks), len(others)), replace=False)
        chosen += [(person, f, True) for f in outliers] + [(person, int(f), False) for f in picked]
    requests: dict[int, list[str]] = {}
    for person, frame, _ in chosen:
        requests.setdefault(frame, []).append(person)
    print(f"{len(chosen)} person-frames ({sum(o for *_, o in chosen)} outliers) on {len(requests)} frames", flush=True)

    settings = [("full_bf16", "full", False, args.repeats), ("full_fp32", "full", True, args.repeats),
                ("body_bf16", "body", False, 1)]
    results: dict[str, list[dict]] = {}
    for name, inference, fp32, repeats in settings:
        estimator = load_estimator(args.repo, args.checkpoint, args.device, fp32)
        results[name] = []
        for r in range(repeats):
            # A run must cover every requested frame: no deadline.
            model = Model(estimator, args.pipeline / "input.mp4", masks, inference, float("inf"))
            out = {p: {} for p in masks}
            start = time.time()
            model.run(requests, out)
            results[name].append(out)
            print(f"{name} run {r}: {time.time() - start:.0f} s", flush=True)
        del estimator
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    report = {"person_frames": len(chosen), "outliers": int(sum(o for *_, o in chosen))}
    for name, runs in results.items():
        if len(runs) < 2:
            continue
        for group, wanted in (("outliers", True), ("random", False)):
            spread = [max(max_mm(x[p][f]["keypoints_3d"], y[p][f]["keypoints_3d"]) for x, y in combinations(runs, 2))
                      for p, f, o in chosen if o == wanted]
            report[f"{name}_{group}"] = {
                "runs_disagree_share": round(float(np.mean(np.array(spread) > args.outlier_mm)), 3),
                "spread_mm_p50_p95_max": [round(float(v), 2) for v in np.percentile(spread, [50, 95, 100])]}
    # Is a jumping hand the body decoder hand? Distance of each run's hands to the body-only hands.
    body = results["body_bf16"][0]
    for name in ("full_bf16", "full_fp32"):
        near_body, far_body = [], []
        for p, f, o in chosen:
            if not o:
                continue
            distances = [max_mm(run[p][f]["keypoints_3d"], body[p][f]["keypoints_3d"]) for run in results[name]]
            near_body.append(min(distances)), far_body.append(max(distances))
        report[f"{name}_outliers_to_body_mm_closest_run_p50"] = round(float(np.median(near_body)), 1)
        report[f"{name}_outliers_to_body_mm_farthest_run_p50"] = round(float(np.median(far_body)), 1)
    # The earlier runs too: on the outliers, which of the two is near the body-only hand?
    near = []
    for p, f, o in chosen:
        if o:
            rows = [run[p]["keypoints_3d"][list(run[p]["frames"]).index(f)] for run in earlier]
            near.append(sorted(max_mm(k, body[p][f]["keypoints_3d"]) for k in rows))
    report["earlier_outliers_to_body_mm_near_far_p50"] = np.median(np.array(near), axis=0).round(1).tolist()
    path = args.output or args.pipeline / "repeat_study.json"
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("REPEAT_OK")


if __name__ == "__main__":
    main()
