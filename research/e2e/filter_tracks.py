"""Sets aside the SAM3 tracks present in too few frames, before the (costly) pose estimation.

Edited videos give short tracks that are not the people of interest: on-screen graphics (logo,
photo insets), a hand in extreme close-up, a shoulder at the frame edge. Pose estimation runs once
per track over the whole video, so each of them costs as much as a real person.

Tracks present in less than ``--min_presence`` of the frames are moved to ``masks_dropped/``
(nothing is deleted), and the decision is written to ``tracks.json``.

    python research/e2e/filter_tracks.py .e2e_outputs/v3_talkshow --min_presence 0.05
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.e2e.evaluate import mask_presence  # noqa: E402
from psifx.io import video  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min_presence", type=float, default=0.05, help="minimum share of frames with the track")
    args = parser.parse_args()
    d = args.out_dir
    masks, dropped = d / "masks", d / "masks_dropped"
    if not masks.is_dir() or not any(masks.glob("*.mp4")):
        sys.exit(f"error: no mask videos in {masks}")

    with video.VideoReader(path=d / "input.mp4") as reader:
        n_frames = len(reader)
    presence = {p.stem: float(mask_presence(p, n_frames).mean()) for p in sorted(masks.glob("*.mp4"))}
    keep = [t for t, share in presence.items() if share >= args.min_presence]
    if not keep:
        # Stop before moving every mask away: the threshold is wrong for this video.
        sys.exit(f"error: no track reaches {args.min_presence:.0%} presence: {presence}")

    dropped.mkdir(exist_ok=True)
    for track, share in presence.items():
        if share < args.min_presence:
            shutil.move(str(masks / f"{track}.mp4"), str(dropped / f"{track}.mp4"))
    result = {"min_presence": args.min_presence, "presence": presence, "kept": keep,
              "dropped": sorted(set(presence) - set(keep), key=int)}
    (d / "tracks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
