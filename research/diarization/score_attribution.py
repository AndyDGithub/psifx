"""Scores who-speaks-when at the person level: diarization turns mapped to people by the
voice<->person link (``identification_lips.json``), against a reference RTTM.

A 10 ms frame of reference speech is correct when the person the reference names is among the
people the hypothesis gives at that time; silence in the hypothesis counts as wrong. This is the
error a viewer sees on the colored subtitles, unlike the DER that maps speakers optimally.

    python research/diarization/score_attribution.py .e2e_outputs/diar_e2e/v1_* \\
        --reference .e2e_outputs/diar/{case}/reference.rttm --names mic_left=offscreen mic_right=person_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STEP = 0.01


def turns(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            fields = line.split()
            yield float(fields[3]), float(fields[3]) + float(fields[4]), fields[7]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="folders named <case>__<tag>")
    parser.add_argument("--reference", required=True, help="reference RTTM, {case} is replaced")
    parser.add_argument("--names", nargs="*", default=[], help="reference label=person renames")
    parser.add_argument("--out", type=Path, default=None, help="JSON file to write")
    args = parser.parse_args()
    names = dict(item.split("=", 1) for item in args.names)
    results = {}
    for run in args.runs:
        case, tag = run.name.split("__", 1)
        reference = Path(args.reference.format(case=case))
        mapping_path = run / "identification_lips.json"
        if not reference.exists() or not mapping_path.exists():
            sys.exit(f"error: missing {reference} or {mapping_path}")
        mapping = json.loads(mapping_path.read_text())["mapping"]
        ref = [(s, e, names.get(p, p)) for s, e, p in turns(reference)]
        hyp = [(s, e, mapping.get(p, "offscreen")) for s, e, p in turns(run / "diarization.rttm")]
        end = max(e for _, e, _ in ref + hyp)
        people = sorted({p for *_, p in ref + hyp})
        grid = {p: np.zeros(int(end / STEP) + 1, dtype=bool) for p in people}
        truth = np.full(len(grid[people[0]]), "", dtype=object)
        for s, e, p in hyp:
            grid[p][int(s / STEP): int(e / STEP)] = True
        for s, e, p in ref:
            truth[int(s / STEP): int(e / STEP)] = p
        scored = truth != ""
        correct = np.array([grid[p][i] for i, p in enumerate(truth) if p])
        results.setdefault(case, {})[tag] = round(float(correct.mean()), 3)
        print(f"{case:18s} {tag:26s} attribution={results[case][tag]:.3f}  ({scored.sum() * STEP:.1f} s scored)")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
