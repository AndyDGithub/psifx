"""Collects the ``metrics.json`` of several end-to-end runs into Markdown tables.

    python research/e2e/summarize.py .e2e_outputs/v1_example .e2e_outputs/v2_ava ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STAGES = ["tracking_sam3", "pose_body_rtmo", "pose_body_vitpose", "pose_mediapipe", "vocals_separation",
          "diarization", "transcription", "link_speakers", "render"]


def fmt(value, digits=2):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    runs = {}
    for run in args.runs:
        path = run / "metrics.json"
        if not path.exists():
            sys.exit(f"error: missing {path}")
        runs[run.name] = json.loads(path.read_text())

    print("| Vidéo | Durée (s) | Pistes SAM3 | Poses RTMO (% images) | Poses MediaPipe (% images) | Locuteurs diarisés "
          "| Texte attribué à une personne visible | Facteur temps réel |")
    print("|---|---|---|---|---|---|---|---|")
    for name, m in runs.items():
        body = m["poses"].get("poses_body", {})
        mp = m["poses"].get("poses_mp", {})
        print(f"| {name} | {m['video']['seconds']} | {m['tracking']['tracks']} "
              f"| {', '.join(fmt(s['detected_frames']) for s in body.values())} "
              f"| {', '.join(fmt(s['detected_frames']) for s in mp.values())} "
              f"| {m['audio'].get('diarized_speakers', '–')} "
              f"| {fmt(m['audio'].get('subtitle_time_on_visible_person'))} | {fmt(m.get('real_time_factor'), 1)} |")

    print("\n| Vidéo | " + " | ".join(STAGES) + " |")
    print("|---|" + "---|" * len(STAGES))
    for name, m in runs.items():
        seconds = m["video"]["seconds"]
        cells = []
        for stage in STAGES:
            t = m.get("timings_s", {}).get(stage)
            cells.append("–" if t is None else f"{t:.0f} s ({t / seconds:.1f}×)")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
