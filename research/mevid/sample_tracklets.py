"""Picks MEVID test tracklets by crop height, and the frames to extract from the chip archive.

The height of a tracklet is the height of its first chip (``first_frames/``, already extracted). For each height
bin, ``--per_bin`` tracklets are drawn (fixed seed), and ``--frames`` consecutive frames around their middle are kept:
consecutive, so that jitter can be measured.

With ``--all``, every tracklet is kept (no drawing), and a tracklet shorter than ``--frames`` keeps all its frames.

Writes ``<output>/sample.json`` and ``<output>/members.txt``, the archive paths for ``tar -T``.

    python research/mevid/sample_tracklets.py .e2e_outputs/ext/mevid --output .e2e_outputs/ext/mevid/small_crops
    python research/mevid/sample_tracklets.py .e2e_outputs/ext/mevid --output .e2e_outputs/ext/mevid/test_split \\
        --all --frames 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BINS = (0, 64, 96, 128, 192, 288, 10 ** 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mevid", type=Path, help="folder with annotations/ and chips/first_frames/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per_bin", type=int, default=20)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="every tracklet, short ones included")
    args = parser.parse_args()

    notes = args.mevid / "annotations" / "x" / "mevid-v1-annotation-data"
    first = args.mevid / "chips" / "first_frames"
    for path in (notes / "track_test_info.txt", notes / "test_name.txt", first):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    names = (notes / "test_name.txt").read_text().split()
    # Columns: first and last index in test_name.txt, person id, outfit, camera.
    tracks = np.loadtxt(notes / "track_test_info.txt").astype(int)

    rows = []
    for t, (start, end, person, outfit, camera) in enumerate(tracks):
        path = first / "bbox_test" / names[start][:4] / names[start]
        if not path.is_file():
            sys.exit(f"error: missing first chip {path}")
        with Image.open(path) as image:
            width, height = image.size
        rows.append((t, start, end, person, outfit, camera, width, height))

    rng = np.random.default_rng(args.seed)
    sample, members = [], []
    for low, high in zip(BINS, BINS[1:]):
        inside = [r for r in rows if low <= r[7] < high and (args.all or r[2] - r[1] + 1 >= args.frames)]
        drawn = inside if args.all else [inside[i] for i in rng.choice(len(inside), min(args.per_bin, len(inside)),
                                                                       replace=False)]
        for t, start, end, person, outfit, camera, width, height in drawn:
            # Window around the middle, kept inside the tracklet (a short one gives all its frames).
            middle = max(start, (start + end) // 2 - args.frames // 2)
            frames = names[middle:min(middle + args.frames, end + 1)]
            sample.append({"track": int(t), "person": int(person), "outfit": int(outfit), "camera": int(camera),
                           "bin": [low, high], "first_height": int(height), "first_width": int(width),
                           "frames": frames})
            members += [f"bbox_test/{name[:4]}/{name}" for name in frames]
        print(f"bin {low}-{high}: {len(inside)} tracklets, {len(drawn)} kept", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sample.json").write_text(json.dumps(sample, indent=1))
    (args.output / "members.txt").write_text("\n".join(members) + "\n")
    print(f"{len(sample)} tracklets, {len(members)} chips")


if __name__ == "__main__":
    main()
