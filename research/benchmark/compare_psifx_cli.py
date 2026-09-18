"""Compare the latest psifx CLI benchmark result of every branch.

Reads results/psifx_cli/*.csv from all local branches (via ``git show``, so the
branches don't need to be checked out or merged) and prints one line per
(branch, tool, coco subset): the most recent run.

Usage: python compare_psifx_cli.py [--coco benchmark|tests]
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS_PREFIX = "research/benchmark/results/psifx_cli/"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", default="benchmark", choices=["benchmark", "tests"])
    args = parser.parse_args()

    latest = {}
    for branch in git("for-each-ref", "--format=%(refname:short)", "refs/heads").split():
        csv_path = f"{RESULTS_PREFIX}{branch.replace('/', '-')}.csv"
        try:
            content = git("show", f"{branch}:{csv_path}")
        except subprocess.CalledProcessError:
            continue  # branch has no results yet
        for row in csv.DictReader(io.StringIO(content)):
            if row["coco"] != args.coco:
                continue
            key = (row["branch"], row["tool"], row["extra"])
            # ISO timestamps sort chronologically as strings.
            if key not in latest or row["timestamp"] > latest[key]["timestamp"]:
                latest[key] = row

    if not latest:
        raise SystemExit(f"no committed results for coco={args.coco}")
    header = f"{'branch':32} {'tool':10} {'extra':18} {'OKS':>6} {'PCK':>6} {'det':>5} {'wall s':>7}"
    print(header)
    print("-" * len(header))
    for row in sorted(latest.values(), key=lambda r: -float(r["oks"])):
        print(
            f"{row['branch']:32} {row['tool']:10} {row['extra'][:18]:18} "
            f"{float(row['oks']):6.3f} {float(row['pck']):6.3f} {float(row['detection_rate']):5.2f} "
            f"{float(row['wall_seconds']):7.1f}"
        )


if __name__ == "__main__":
    main()
