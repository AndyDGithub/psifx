"""Which 2D keypoints can be trusted: inside the picture and on the person's SAM3 mask.

A 2D detector predicts every keypoint, even when the body part is out of the frame: RTMW then puts the wrist or the
hand on the border of the picture, often with a confidence above 0.5 (AVA, hands below the frame). Fitting or judging
an avatar on these points pulls it toward the border. A point is kept only if it is

- inside the picture, at least ``margin`` (share of the picture height) from its border. The picture is the area
  without the black bars of a letterboxed video, found once per video;
- on the SAM3 mask of the person it belongs to, within ``reach`` pixels (masks are a little tight on hands),
  **unless the mask is partial on that frame**. SAM3 sometimes keeps only a piece of the person (V5, frames
  310-370: the sweater, without head or legs, on 15 % of the dancer's frames); the mask test then hid legs seen
  plainly by RTMW, and the foot stamps were never fitted. With ``cover`` (RTMW body points and scores), a frame
  whose dilated mask holds less than ``cover_min`` of the confident body points inside the picture keeps only the
  picture test.

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


def mask_frames(pipeline: Path, last: int, width: int, height: int):
    """Yields (frame, {track: bool mask}) for frames 0..last, all SAM3 tracks read together (a track whose video has
    ended is absent)."""
    streams = {}
    try:
        for path in sorted((pipeline / "masks").glob("*.mp4")):
            streams[path.stem] = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                                                   "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
        for f in range(last + 1):
            masks = {}
            for track, process in list(streams.items()):
                buffer = process.stdout.read(width * height)
                if len(buffer) < width * height:
                    process.stdout.close()
                    process.kill()
                    process.wait()
                    del streams[track]
                    continue
                masks[track] = np.frombuffer(buffer, np.uint8).reshape(height, width) > 127
            yield f, masks
    finally:
        for process in streams.values():
            process.stdout.close()
            process.kill()
            process.wait()


def in_view(pipeline: Path, people: dict, person: str, frames: np.ndarray, points: np.ndarray,
            margin: float = 0.015, reach: int = 12, use_mask: bool = True,
            cover: tuple[np.ndarray, np.ndarray] | None = None, cover_min: float = 0.7, cover_score: float = 3.0,
            stats: dict | None = None) -> np.ndarray:
    """(n, k) bool: True where points[i, k] (pixels, frame frames[i]) is inside the picture and on the person's mask.

    Frames of the person without a track in people.json keep only the picture test. cover: (body points (n, m, 2),
    scores (n, m)) of RTMW; frames where the mask holds less than cover_min of the points scored above cover_score
    keep only the picture test too (partial mask). A point on another person's mask (and off the person's own,
    dilated) is never kept, partial mask or not: it is the detector reading the other person (V5's push: the man's
    legs put on the girl's), and such points are left out of the cover count, else they made the mask look partial
    and switched the mask test off. stats, if given, receives "partial_mask_frames" and "partial_rows" (their rows).
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
    rows_of: dict[int, list[int]] = {}
    for i, f in enumerate(frames.astype(int)):
        t = tracks.get(int(f))
        if t is not None:
            if not (pipeline / "masks" / f"{t}.mp4").is_file():
                sys.exit(f"error: missing {pipeline / 'masks' / f'{t}.mp4'}")
            rows_of.setdefault(int(f), []).append(i)
    partial, partial_rows = 0, []
    if not rows_of:
        if stats is not None:
            stats["partial_mask_frames"], stats["partial_rows"] = 0, np.zeros(0, int)
        return ok
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    if cover is not None:
        cu, cv_ = cover[0][..., 0], cover[0][..., 1]
        confident = (cover[1] > cover_score) & np.isfinite(cu) & np.isfinite(cv_) & (cu >= left) & (cu < right) & \
            (cv_ >= top) & (cv_ < bottom)

    def pixels(x, y):
        return (np.clip(np.round(np.nan_to_num(y)).astype(int), 0, height - 1),
                np.clip(np.round(np.nan_to_num(x)).astype(int), 0, width - 1))

    for f, masks in mask_frames(pipeline, max(rows_of), width, height):
        if f not in rows_of:
            continue
        track = tracks[f]
        if track not in masks:
            sys.exit(f"error: {pipeline / 'masks' / f'{track}.mp4'} ends before frame {f}")
        own = cv2.dilate(masks[track].astype(np.uint8), kernel) > 0
        others = np.zeros_like(own)
        for t, mk in masks.items():
            if t != track:
                others |= mk
        others &= ~own
        for i in rows_of[f]:
            off_others = ~others[pixels(u[i], v[i])]
            if cover is not None and confident[i].sum() >= 5:
                yy, xx = pixels(cu[i][confident[i]], cv_[i][confident[i]])
                mine = ~others[yy, xx]  # points read on another person do not judge this mask
                if mine.sum() >= 5 and own[yy, xx][mine].mean() < cover_min:
                    partial += 1  # partial mask: it would hide parts that are plainly seen
                    partial_rows.append(i)
                    ok[i] &= off_others
                    continue
            ok[i] &= own[pixels(u[i], v[i])]
    if stats is not None:
        stats["partial_mask_frames"] = partial
        stats["partial_rows"] = np.array(sorted(partial_rows), int)
    return ok


def allowed_distance(pipeline: Path, people: dict, person: str, frames: np.ndarray, skip_rows: np.ndarray,
                     reach: int = 12, scale: int = 4, border: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Where a body part of the person may be, and the frames where the question is asked.

    Returns (dist, edge): dist (n, H/scale, W/scale) uint8, the distance (in pixels at 1/scale, capped at 255) from
    each pixel to the nearest pixel that is on a SAM3 mask of anyone (dilated by ``reach``) or outside the picture
    (black bars included); edge (n,) bool, True on frames where the person's own mask comes within ``border``
    pixels of the picture's border: the person enters or leaves the frame, so a part seen on no mask is out of the
    frame. Elsewhere a part seen on no mask may be hidden by a thing (a table), and the frame is not used (edge
    False, as on rows in ``skip_rows``: partial masks)."""
    import cv2

    capture = cv2.VideoCapture(str(pipeline / "input.mp4"))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    w, h = width // scale, height // scale
    left, top, right, bottom = picture_area(pipeline / "input.mp4")
    picture = np.zeros((h, w), bool)
    picture[top // scale:bottom // scale, left // scale:right // scale] = True
    inner = np.zeros((h, w), bool)
    b = max(1, border // scale)
    inner[top // scale + b:bottom // scale - b, left // scale + b:right // scale - b] = True
    tracks = track_of_frames(people, person)
    frames = frames.astype(int)
    wanted = {int(f) for f in frames}
    union = {f: np.zeros((h, w), bool) for f in wanted}
    own = {}
    for path in sorted((pipeline / "masks").glob("*.mp4")):
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}:flags=area",
                                    "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
        try:
            for f in range(max(wanted) + 1):
                buffer = process.stdout.read(w * h)
                if len(buffer) < w * h:
                    break  # a mask video may end early: no mask after
                if f not in wanted:
                    continue
                m = np.frombuffer(buffer, np.uint8).reshape(h, w) > 63
                union[f] |= m
                if tracks.get(f) == path.stem:
                    own[f] = m
        finally:
            process.stdout.close()
            process.kill()
            process.wait()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (reach // scale) + 1,) * 2)
    dist = np.zeros((len(frames), h, w), np.uint8)
    edge = np.zeros(len(frames), bool)
    skip = set(int(r) for r in skip_rows)
    for i, f in enumerate(frames):
        if i in skip or f not in own or not own[f].any():
            continue
        edge[i] = bool((own[f] & ~inner).any())
        if not edge[i]:
            continue
        allowed = cv2.dilate(union[f].astype(np.uint8), kernel).astype(bool) | ~picture
        dist[i] = np.minimum(cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 3), 255).astype(np.uint8)
    return dist, edge
