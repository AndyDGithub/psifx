"""Benchmark a psifx pose CLI tool on COCO and append the result to a CSV.

Runs ``psifx video pose <tool> single-inference`` exactly as a user would,
so each model/optimization branch can be compared on the same input and the
same output format. One row per run, tagged with the current git branch and
commit, so results from several branches accumulate in the same file.

Usage:
    python run_psifx_cli_benchmark.py --tool mediapipe
    python run_psifx_cli_benchmark.py --tool rtmo --extra "--device cuda"
    python run_psifx_cli_benchmark.py --tool mediapipe --coco tests   # 8-image test subset
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from psifx_eval import ADAPTERS, build_coco_video, evaluate_archive, load_coco_samples

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
COCO_SOURCES = {
    "benchmark": HERE / "data" / "coco_subset",  # 50 images, see prepare_coco_subset.py
    "tests": REPO / "tests" / "integration" / "data" / "CocoPose",  # 8 images, committed
}
RESULTS_CSV = HERE / "results" / "psifx_cli_results.csv"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--coco", default="benchmark", choices=sorted(COCO_SOURCES))
    parser.add_argument("--extra", default="", help="extra CLI arguments passed to single-inference")
    parser.add_argument("--size", type=int, default=640, help="letterboxed frame size")
    args = parser.parse_args()

    coco_dir = COCO_SOURCES[args.coco]
    if not (coco_dir / "annotations.json").is_file():
        raise SystemExit(f"COCO data missing in {coco_dir} (run prepare_coco_subset.py, or use --coco tests)")
    samples = load_coco_samples(coco_dir / "annotations.json", coco_dir / "images")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path, poses_path = tmp / "coco.mp4", tmp / "poses.tar.gz"
        ground_truth = build_coco_video(samples, video_path, size=args.size)

        cmd = [
            "psifx", "video", "pose", args.tool, "single-inference",
            "--video", str(video_path), "--poses", str(poses_path),
            "--overwrite", "--no-verbose", *shlex.split(args.extra),
        ]
        # Wall-clock includes model loading: that is what a psifx user pays per video.
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        wall_seconds = time.perf_counter() - t0

        scores = evaluate_archive(poses_path, ground_truth, args.tool)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "tool": args.tool,
        "extra": args.extra,
        "coco": args.coco,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()},
        "wall_seconds": round(wall_seconds, 2),
    }
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
