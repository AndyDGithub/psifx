"""Per-person archives from the per-track SAM 3D Body archives and an existing ``people.json``.

``merge_people.py`` decides which pieces of which SAM3 track are which person and writes the person archives once.
When more frames are computed later (``run_tracks.py --stride 2 --offset 1`` fills the odd frames of a stride 2
run), this script gives them to the same people without deciding again: the archives of several runs are joined
per track, then each piece of ``people.json`` takes the frames of its track from its first frame to the frame
before the next computed one of the same shot (``frames`` in ``people.json`` are the first and last computed frames
of the piece at the old stride 2).

    python research/sam3d_body/apply_people.py .e2e_outputs/extrait_talkshow/pipeline \\
        --params sam3d_params sam3d_params_odd --output sam3d_people60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_keyframes import cut_labels  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--params", nargs="+", required=True, help="per-track archive folders to join")
    parser.add_argument("--people", type=Path, help="default: <pipeline>/people.json")
    parser.add_argument("--output", required=True, help="output folder, inside the pipeline")
    parser.add_argument("--cut", type=float, default=18)
    args = parser.parse_args()

    d = args.pipeline
    people_path = args.people or d / "people.json"
    for path in [people_path, d / "input.mp4", *(d / p for p in args.params)]:
        if not path.exists():
            sys.exit(f"error: missing {path}")
    out = d / args.output
    if out.exists() and any(out.iterdir()):
        sys.exit(f"error: {out} is not empty")
    people = json.loads(people_path.read_text())
    labels = cut_labels(d / "input.mp4", args.cut)

    tracks: dict[str, dict] = {}
    for folder in args.params:
        for path in sorted((d / folder).glob("*.npz")):
            with np.load(path) as npz:
                part = {k: npz[k] for k in npz.files}
            if path.stem in tracks:
                faces = tracks[path.stem].pop("faces")
                tracks[path.stem] = {k: np.concatenate([tracks[path.stem][k], part[k]]) for k in tracks[path.stem]}
                tracks[path.stem]["faces"] = faces
            else:
                tracks[path.stem] = part
    for t, a in tracks.items():
        order = np.argsort(a["frames"], kind="stable")
        tracks[t] = {k: (v if k == "faces" else v[order]) for k, v in a.items()}
        if len(np.unique(a["frames"])) != len(a["frames"]):
            sys.exit(f"error: track {t} has a frame twice: the folders overlap")

    chosen: dict[int, list] = {p: [] for p in range(people["people"])}
    for piece in people["assignment"]:
        if not isinstance(piece["person"], int):
            continue
        a = tracks.get(str(piece["track"]))
        if a is None:
            sys.exit(f"error: no archive for track {piece['track']}")
        first, last = piece["frames"]
        frames = a["frames"].astype(int)
        # Up to the frame before the next one of the old stride 2 (odd frame after the last even one).
        rows = np.where((frames >= first) & (frames <= last + 1) & (labels[frames] == labels[first]))[0]
        chosen[piece["person"]].append((str(piece["track"]), rows))

    out.mkdir(exist_ok=True)
    for p, pieces in chosen.items():
        keys = [k for k in next(iter(tracks.values())) if k != "faces"]
        merged = {k: np.concatenate([tracks[t][k][r] for t, r in pieces]) for k in keys}
        order = np.argsort(merged["frames"], kind="stable")
        merged = {k: v[order] for k, v in merged.items()}
        if len(np.unique(merged["frames"])) != len(merged["frames"]):
            sys.exit(f"error: person {p} gets a frame from two pieces")
        merged["faces"] = next(iter(tracks.values()))["faces"]
        np.savez_compressed(out / f"{p}.npz", **merged)
        print(f"person {p}: {len(merged['frames'])} frames")
    print("APPLY_OK")


if __name__ == "__main__":
    main()
