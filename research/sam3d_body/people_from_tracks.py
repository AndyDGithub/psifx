"""people.json from the SAM3 tracks alone: one person per track, one piece per shot, without the pieces that are
not a person of the scene.

SAM3 finds "persons" that are not in the scene, and its re-identification can hand a dancer's track to them:

- **pictures**: a still image of a person (an inset frozen on a frame, a poster, a magazine cover). A real person,
  even standing still, breathes, blinks and carries sensor noise; a picture does not change at all. A piece whose
  pixels (inside its mask, at 1/4 size) change by less than ``--still`` gray levels per frame for its median frame,
  over at least ``--min_still_s`` seconds, is marked ``"picture"`` and gets no avatar (apply_people.py skips pieces
  whose person is not a number).
- **jumps**: inside a shot, when a track's mask moves or changes size abruptly (centroid moved by more than
  ``--jump`` of the image diagonal, or area x3), the track has switched to something else: the piece is cut there,
  so each part is judged on its own.

Only the largest connected part of each mask is used (SAM3 masks sometimes keep a strip on the image border).

    python research/sam3d_body/people_from_tracks.py .e2e_outputs/v5_danse_montage
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_keyframes import cut_labels  # noqa: E402
from run_tracks import largest_part  # noqa: E402


def gray_frames(path: Path, width: int, height: int):
    """Yields the frames of a video in gray at width x height."""
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={width}:{height}",
                                "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    try:
        while len(buffer := process.stdout.read(width * height)) == width * height:
            yield np.frombuffer(buffer, np.uint8).reshape(height, width)
    finally:
        process.stdout.close()
        process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4 and masks/")
    parser.add_argument("--min_area", type=int, default=400, help="smallest mask (pixels, full size) of a present track")
    parser.add_argument("--cut", type=float, default=18, help="cut threshold, as in run_keyframes.py")
    parser.add_argument("--still", type=float, default=1.0,
                        help="median change inside the mask (gray levels per frame, 1/4 size) under which it is a picture")
    parser.add_argument("--min_still_s", type=float, default=1.0, help="shortest piece judged a picture (seconds)")
    parser.add_argument("--jump", type=float, default=0.15, help="centroid jump (share of the diagonal) that cuts a piece")
    parser.add_argument("--output", type=Path, help="default: <pipeline>/people.json")
    args = parser.parse_args()

    d = args.pipeline
    video = d / "input.mp4"
    tracks = sorted((d / "masks").glob("*.mp4"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not video.is_file() or not tracks:
        sys.exit(f"error: {d} needs input.mp4 and masks/*.mp4")
    capture = cv2.VideoCapture(str(video))
    width, height, fps = int(capture.get(3)), int(capture.get(4)), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not fps:
        sys.exit(f"error: no frame rate in {video}")
    w4, h4 = max(width // 4, 8), max(height // 4, 8)
    scale = (width * height) / (w4 * h4)
    diagonal = np.hypot(w4, h4)
    labels = cut_labels(video, args.cut)
    frames = np.stack(list(gray_frames(video, w4, h4))).astype(np.int16)
    n = min(len(labels), len(frames))

    assignment, report = [], {}
    for person, path in enumerate(tracks):
        present = np.zeros(n, bool)
        area = np.zeros(n)
        centroid = np.zeros((n, 2))
        change = np.full(n, np.nan)  # mean gray change inside the mask from the previous frame
        for f, mask in enumerate(gray_frames(path, w4, h4)):
            if f >= n:
                break
            part = largest_part(mask > 127)
            area[f] = part.sum()
            if area[f] * scale < args.min_area:
                continue
            present[f] = True
            ys, xs = np.nonzero(part)
            centroid[f] = xs.mean(), ys.mean()
            if f > 0:
                change[f] = np.abs(frames[f][part] - frames[f - 1][part]).mean()
        pieces = []
        for shot in np.unique(labels[:n]):
            idx = np.where((labels[:n] == shot) & present)[0]
            if len(idx) == 0:
                continue
            # Cut the run of the track in this shot at gaps and at jumps of the mask.
            start = idx[0]
            for a, b in zip(idx[:-1], idx[1:]):
                jump = (np.linalg.norm(centroid[b] - centroid[a]) > args.jump * diagonal
                        or max(area[a], area[b]) > 3 * max(min(area[a], area[b]), 1))
                if b != a + 1 or jump:
                    pieces.append((int(shot), int(start), int(a)))
                    start = b
            pieces.append((int(shot), int(start), int(idx[-1])))
        kept, pictures = 0, 0
        for shot, first, last in pieces:
            c = change[first + 1:last + 1]
            c = c[np.isfinite(c)]
            still = (last - first + 1) >= args.min_still_s * fps and len(c) and float(np.median(c)) < args.still
            assignment.append({"track": path.stem, "shot": shot, "frames": [first, last],
                               "person": "picture" if still else person,
                               "change_median": round(float(np.median(c)), 2) if len(c) else None})
            pictures += bool(still)
            kept += not still
        report[path.stem] = {"present_share": round(float(present.mean()), 3), "pieces": kept, "pictures": pictures}
        print(f"track {path.stem}: {100 * present.mean():.1f}% of frames, {kept} pieces, {pictures} pictures set aside",
              flush=True)

    out = args.output or d / "people.json"
    out.write_text(json.dumps({"method": "one person per SAM3 track, pieces cut at shots and mask jumps, still "
                                         "pictures set aside", "people": len(tracks),
                               "track_to_person": {p.stem: i for i, p in enumerate(tracks)},
                               "tracks": report, "assignment": assignment}, indent=1))
    print(f"wrote {out}")
    print("PEOPLE_OK")


if __name__ == "__main__":
    main()
