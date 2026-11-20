"""Which 2D keypoints can be trusted: inside the picture and on the person's SAM3 mask.

A 2D detector predicts every keypoint, even when the body part is out of the frame: RTMW then puts the wrist or the
hand on the border of the picture, often with a confidence above 0.5 (AVA, hands below the frame). Fitting or judging
an avatar on these points pulls it toward the border. A point is kept only if it is

- inside the picture, at least ``margin`` (share of the picture height) from its border. The picture is the area
  without the black bars of a letterboxed video, found once per video;
- on the SAM3 mask of the person it belongs to, within ``reach`` pixels (masks are a little tight on hands).

The track of each frame of a person comes from people.json, as in robust_smooth.visibility.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


def picture_area(video: Path, samples: int = 24, level: int = 24) -> tuple[int, int, int, int]:
    """(left, top, right, bottom), exclusive right/bottom, of the area without black bars, from ``samples`` frames."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"error: cannot open {video}")
    n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    brightest = np.zeros((height, width), np.uint8)
    for f in np.linspace(0, max(n - 1, 0), samples).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, image = capture.read()
        if ok:
            np.maximum(brightest, image.max(axis=2), out=brightest)
    capture.release()
    rows, cols = np.where(brightest.max(axis=1) > level)[0], np.where(brightest.max(axis=0) > level)[0]
    if len(rows) == 0 or len(cols) == 0:  # black video: keep the whole frame
        return 0, 0, width, height
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def track_of_frames(people: dict, person: str) -> dict[int, str]:
    """frame -> SAM3 track of this person, from the pieces of people.json."""
    out = {}
    for piece in people["assignment"]:
        if str(piece["person"]) == person:
            for f in range(piece["frames"][0], piece["frames"][1] + 1):
                out[f] = str(piece["track"])
    return out


def in_view(pipeline: Path, people: dict, person: str, frames: np.ndarray, points: np.ndarray,
            margin: float = 0.015, reach: int = 12, use_mask: bool = True) -> np.ndarray:
    """(n, k) bool: True where points[i, k] (pixels, frame frames[i]) is inside the picture and on the person's mask.

    Frames of the person without a track in people.json keep only the picture test.
    """
    left, top, right, bottom = picture_area(pipeline / "input.mp4")
    m = margin * (bottom - top)
    u, v = points[..., 0], points[..., 1]
    ok = np.isfinite(u) & np.isfinite(v) & (u >= left + m) & (u < right - m) & (v >= top + m) & (v < bottom - m)
    if not use_mask:
        return ok
    import cv2

    capture = cv2.VideoCapture(str(pipeline / "input.mp4"))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    tracks = track_of_frames(people, person)
    by_track: dict[str, dict[int, list[int]]] = {}  # track -> frame -> rows
    for i, f in enumerate(frames.astype(int)):
        t = tracks.get(int(f))
        if t is not None:
            by_track.setdefault(t, {}).setdefault(int(f), []).append(i)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    for track, rows_of in by_track.items():
        path = pipeline / "masks" / f"{track}.mp4"
        if not path.is_file():
            sys.exit(f"error: missing {path}")
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                                   stdout=subprocess.PIPE)
        try:
            last = max(rows_of)
            for f in range(last + 1):
                buffer = process.stdout.read(width * height)
                if len(buffer) < width * height:
                    sys.exit(f"error: {path} ends at frame {f}, frame {last} is needed")
                if f not in rows_of:
                    continue
                mask = cv2.dilate((np.frombuffer(buffer, np.uint8).reshape(height, width) > 127).astype(np.uint8), kernel)
                for i in rows_of[f]:
                    uu = np.clip(np.round(u[i]).astype(int), 0, width - 1)
                    vv = np.clip(np.round(v[i]).astype(int), 0, height - 1)
                    ok[i] &= mask[vv, uu] > 0
        finally:
            process.stdout.close()
            process.kill()
            process.wait()
    return ok
