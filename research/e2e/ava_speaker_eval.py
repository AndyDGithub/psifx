"""Speaker labels against AVA-ActiveSpeaker: at each labelled face that speaks, is the voice heard given to that face?

AVA gives, about 20 times per second, each face box and whether it is speaking (``SPEAKING_AUDIBLE``). Each
speaking face is matched to a SAM3 track by the overlap of its box with the track's MediaPipe face box (IoU >=
``--min_iou``); the hypothesis at that time is the person the diarization cluster heard then is given to, by an
identification json (``voice_ids.py`` or ``link_speakers.py``). Scores:

- ``right``: share of the matched speaking faces whose person is the one the voice is given to;
- ``silent_or_off``: share where no voice is heard or the voice is given to no one / off-screen.

    python research/e2e/ava_speaker_eval.py .e2e_outputs/v2_ava --identification voices.json \\
        --diarization diarization_over6.rttm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from psifx.io import rttm, video
from psifx.video.pose.tool import PoseEstimationTool


def face_boxes(poses: dict, n_frames: int) -> np.ndarray:
    boxes = np.full((n_frames, 4), np.nan)
    for i, pose in poses.items():
        if i < n_frames:
            face = np.asarray(pose.get("face_keypoints_2d", []), dtype=float).reshape(-1, 3)
            points = face[(face[:, :2] > 0).all(1), :2]
            if len(points) >= 100:
                boxes[i] = (*points.min(0), *points.max(0))
    return boxes


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4, poses_mp/ and ava_active_speaker.csv")
    parser.add_argument("--identification", nargs="+", required=True, help="identification jsons, in the folder")
    parser.add_argument("--diarization", nargs="+", required=True, help="RTTM of each identification, in the folder")
    parser.add_argument("--min_iou", type=float, default=0.3)
    args = parser.parse_args()

    d = args.pipeline
    if len(args.identification) != len(args.diarization):
        sys.exit("error: one --diarization per --identification")
    for path in [d / "ava_active_speaker.csv", d / "input.mp4", d / "poses_mp",
                 *(d / p for p in args.identification + args.diarization)]:
        if not path.exists():
            sys.exit(f"error: missing {path}")
    with video.VideoReader(path=d / "input.mp4") as reader:
        n_frames, fps = len(reader), float(Fraction(reader.frame_rate))
    import cv2
    capture = cv2.VideoCapture(str(d / "input.mp4"))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    decoder = PoseEstimationTool(device="cpu", verbose=False)
    boxes = {a.name.removesuffix(".tar.gz"): face_boxes(decoder.decode(a)[0], n_frames)
             for a in sorted((d / "poses_mp").glob("*.tar.gz"))}

    speaking = []  # (time, matched track)
    unmatched = 0
    with open(d / "ava_active_speaker.csv") as f:
        for row in csv.DictReader(f):
            if row["label"] != "SPEAKING_AUDIBLE":
                continue
            t = float(row["t"])
            frame = min(n_frames - 1, int(round(t * fps)))
            box = [float(row["x1"]) * width, float(row["y1"]) * height, float(row["x2"]) * width,
                   float(row["y2"]) * height]
            best = max(((iou(box, b[frame]), track) for track, b in boxes.items() if not np.isnan(b[frame]).any()),
                       default=(0.0, None))
            if best[0] >= args.min_iou:
                speaking.append((t, best[1]))
            else:
                unmatched += 1
    report = {"speaking_faces": len(speaking) + unmatched, "matched": len(speaking)}
    for ident, diar in zip(args.identification, args.diarization):
        mapping = json.loads((d / ident).read_text())["mapping"]
        turns = rttm.RTTMReader.read(d / diar, verbose=False)
        right = silent_or_off = 0
        for t, track in speaking:
            heard = [s["speaker_name"] for s in turns if s["start"] <= t < s["start"] + s["duration"]]
            people = {mapping.get(h) for h in heard}
            if f"person_{track}" in people:
                right += 1
            elif not any(p and p.startswith("person_") for p in people):
                silent_or_off += 1
        report[ident] = {"right": round(right / max(1, len(speaking)), 3),
                         "silent_or_off": round(silent_or_off / max(1, len(speaking)), 3)}
    (d / "ava_speaker_eval.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
