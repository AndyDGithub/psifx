"""Robust smoothing of SAM 3D Body parameters: outliers removed, hidden parts trusted less, depth apart.

The Savitzky-Golay filter of ``smooth_params.py`` spreads a wrong frame over its neighbours instead of removing it
(``motion_diag.py``: a 1.7 m depth error on one frame becomes 0.3 m on the frames around it), smooths a visible hand
and a leg hidden under the table the same way, and moves the avatar on the image when the depth is smoothed.
Here, per run (no gap, no shot cut):

1. **Outliers** (Hampel filter): a value far from the median of its window (more than ``HAMPEL_K`` times the
   channel's usual distance to that median, and its noise) is not a measurement: it is dropped, and the smoother
   fills it from its neighbours.
2. **Visibility**: each body part is seen or not on each frame: its 2D keypoints inside the image and on the person's
   SAM3 mask (a leg under the table is inside the image but not on the mask). A hidden part keeps a weak pull
   (noise x ``HIDDEN_NOISE``), so it moves slowly and smoothly instead of following the model's guesses.
3. **Kalman smoother** (Rauch-Tung-Striebel, forward then backward, so no lag), constant velocity, per channel. Its
   strength is a cut-off frequency per part (``CUTOFF_HZ``): with measurement noise ``r`` (estimated per channel from
   the data) the process noise is ``q = r * (2 pi f / rate)^4``, the smoother then halves the power at ``f``.
4. **Depth apart**: the camera translation is smoothed as the image position (``x/z``, ``y/z``, accurate) and the
   log depth (inaccurate, slow for people sitting), then rebuilt: smoothing the depth no longer moves the avatar on
   the image, only its size.

Which part each of the 136 pose values moves is found once on MHR itself (the first joint whose rotation changes
when the value changes), so there is no hand-made table.

Used by ``smooth_params.py`` (variants ``robust`` and ``robust:<scale>``, the cut-offs multiplied by ``scale``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

POSE, FIRST_ANGLE = 136, 3
CUTOFF_HZ = {"root": 2.0, "torso_head": 3.0, "arms": 4.0, "wrists": 4.0, "fingers": 2.5, "legs": 6.0,
             "image_xy": 3.0, "depth": 2.0}
HAMPEL_S = {"depth": 0.6, "image_xy": 0.6, "root": 0.6}  # Window (s); the others use HAMPEL_DEFAULT_S.
HAMPEL_DEFAULT_S, HAMPEL_K, HIDDEN_NOISE = 0.2, 5.0, 100.0
# Smallest error called an outlier, whatever the channel's noise: radians for the angles, cm for the 3 root
# translations, log depth and image position (x/z) for the camera. A still person would otherwise have every
# small movement flagged.
ABS_FLOOR = {"angle": 0.12, "fingers": 0.2, "translation": 2.0, "depth": 0.08, "image_xy": 0.015}
# Legs at 6 Hz (2 Hz before): on a dance, 2 Hz kept 0.69 of the feet amplitude, 6 Hz 0.86 (raw 0.87). Hidden
# legs (under a table) are still held by their weak measure.
# MHR70 keypoints of each part, per side (left, right): hands 42-62 / 21-41 (wrists 62 / 41).
VISIBILITY_KEYPOINTS = {
    "legs": ([11, 13, 15, 16, 17], [12, 14, 18, 19, 20]),
    "arms": ([7], [8]),
    "wrists": (list(range(42, 63)), list(range(21, 42))),
    "fingers": (list(range(42, 63)), list(range(21, 42))),
}


def part_of_joint(name: str) -> str:
    if name in ("body_world", "root"):
        return "root"
    if name.startswith("c_") or name.endswith(("_eye", "_eye_null")):
        return "torso_head"
    if any(k in name for k in ("upleg", "lowleg", "foot", "talocrural", "subtalar", "transversetarsal", "ball")):
        return "legs"
    if any(k in name for k in ("clavicle", "uparm", "lowarm")):
        return "arms"
    if "wrist" in name:
        return "wrists"
    return "fingers"


@torch.inference_mode()
def parameter_parts(mhr, device: str) -> list[tuple[str, str | None]]:
    """(part, side) of each of the 136 pose values: the first joint whose global rotation or position changes."""
    names = list(mhr.character_torch.skeleton.joint_names)
    shape = torch.zeros(1, 45, device=device)
    expr = torch.zeros(1, 72, device=device)
    base = torch.zeros(1, 204, device=device)
    _, ref = mhr(shape, base, expr)
    params = base.repeat(POSE, 1)
    params[torch.arange(POSE), torch.arange(POSE)] += 0.3
    _, skel = mhr(shape.repeat(POSE, 1), params, expr.repeat(POSE, 1))
    changed = (skel[..., :7] - ref[..., :7]).abs().amax(-1) > 1e-4  # (136, joints)
    out = []
    for i in range(POSE):
        joints = torch.nonzero(changed[i]).flatten()
        if len(joints) == 0:
            out.append(("root", None))
            continue
        name = names[int(joints[0])]
        side = name[0] if name[:2] in ("l_", "r_") else None
        out.append((part_of_joint(name), side))
    return out


def hampel_mask(x: np.ndarray, window: int, noise: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """True where a value is an outlier (x: samples x channels): far from the median of its centred window, in
    units of the channel's typical distance to that median over the run (not of the window's own spread, which a
    fast but real movement inflates and which then hides a burst of 2 or 3 wrong frames)."""
    n = len(x)
    half = max(1, window // 2)
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1, axis=0)[:n]  # (n, c, w)
    residual = np.abs(x - np.median(windows, axis=-1))
    scale = 1.4826 * np.median(residual, axis=0)
    return residual > np.maximum(HAMPEL_K * np.maximum(scale, noise), floor)[None]


def channel_noise(x: np.ndarray) -> np.ndarray:
    """Robust measurement noise per channel, from the second difference (white noise: var(d2) = 6 sigma^2)."""
    if len(x) < 4:
        return np.full(x.shape[1], 1e-3)
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0) / np.sqrt(6)
    return np.maximum(sigma, 1e-6)


def rts_smooth(z: np.ndarray, r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Constant-velocity Kalman filter then Rauch-Tung-Striebel smoother, every channel at once.

    z: (n, c) measurements, r: (n, c) measurement variances (inf = no measurement), q: (c,) process noise.
    """
    n, c = z.shape
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q[:, None, None] * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])  # White acceleration, one sample step.
    first = np.argmax(np.isfinite(r), axis=0)  # First measured sample of each channel.
    x = np.zeros((c, 2))
    x[:, 0] = z[first, np.arange(c)]
    P = np.zeros((c, 2, 2))
    P[:, 0, 0] = np.where(np.isfinite(r[first, np.arange(c)]), r[first, np.arange(c)], 1.0) * 1e2
    P[:, 1, 1] = P[:, 0, 0]
    xs_pred, Ps_pred, xs, Ps = (np.empty((n, c, 2)), np.empty((n, c, 2, 2)), np.empty((n, c, 2)),
                                np.empty((n, c, 2, 2)))
    for t in range(n):
        if t:
            x = x @ F.T
            P = F @ P @ F.T + Q
        xs_pred[t], Ps_pred[t] = x, P
        seen = np.isfinite(r[t])
        if seen.any():
            s = P[seen, 0, 0] + r[t, seen]
            k = P[seen, :, 0] / s[:, None]  # Kalman gain (measurement = position only).
            x[seen] = x[seen] + k * (z[t, seen] - x[seen, 0])[:, None]
            P[seen] = P[seen] - k[:, :, None] * P[seen, 0, :][:, None, :]
        xs[t], Ps[t] = x, P
    out = np.empty((n, c))
    xb, Pb = xs[-1], Ps[-1]
    out[-1] = xb[:, 0]
    for t in range(n - 2, -1, -1):
        G = Ps[t] @ F.T @ np.linalg.inv(Ps_pred[t + 1])
        xb = xs[t] + np.einsum("cij,cj->ci", G, xb - xs_pred[t + 1])
        Pb = Ps[t] + G @ (Pb - Ps_pred[t + 1]) @ np.transpose(G, (0, 2, 1))
        out[t] = xb[:, 0]
    return out


def smooth_channels(x: np.ndarray, hidden: np.ndarray, cutoff_hz: np.ndarray, hampel_s: np.ndarray,
                    floor: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Hampel then RTS on (n, c) values; hidden (n, c) bool; returns the smoothed values and the outlier mask."""
    noise = channel_noise(x)
    outliers = np.zeros_like(x, dtype=bool)
    for seconds in np.unique(hampel_s):
        cols = hampel_s == seconds
        window = max(3, int(round(seconds * rate)) | 1)
        if len(x) >= window:
            outliers[:, cols] = hampel_mask(x[:, cols], window, noise[cols], floor[cols])
    r = np.broadcast_to(noise ** 2, x.shape).copy()
    r[hidden] *= HIDDEN_NOISE
    r[outliers] = np.inf
    lost = ~np.isfinite(r).any(0)  # A channel with no measurement left keeps its values.
    q = noise ** 2 * (2 * np.pi * cutoff_hz / rate) ** 4
    out = x.copy()
    if (~lost).any():
        out[:, ~lost] = rts_smooth(x[:, ~lost], r[:, ~lost], q[~lost])
    return out, outliers


def robust(a: dict, segments: list[np.ndarray], rate: float, parts: list, hidden: dict | None,
           scale: float = 1.0, depth_hz: float | None = None, legs_hz: float | None = None) -> tuple[dict, dict]:
    """Locked body, then the robust smoother per run. hidden: part -> (n_samples, 2 sides) bool, or None.
    legs_hz: cut-off of the legs instead of CUTOFF_HZ["legs"] (times scale)."""
    b = {k: v.copy() for k, v in a.items()}
    b["shape_params"][:] = np.median(a["shape_params"], axis=0)
    b["mhr_model_params"][:, POSE:] = np.median(a["mhr_model_params"][:, POSE:], axis=0)
    position = {int(f): i for i, f in enumerate(a["frames"])}
    cutoffs = dict(CUTOFF_HZ, **({"legs": legs_hz} if legs_hz else {}))
    pose_cutoff = np.array([cutoffs[p] for p, _ in parts]) * scale
    pose_hampel = np.array([HAMPEL_S.get(p, HAMPEL_DEFAULT_S) for p, _ in parts])
    pose_floor = np.array([ABS_FLOOR["translation"] if j < FIRST_ANGLE else
                           ABS_FLOOR["fingers" if p == "fingers" else "angle"] for j, (p, _) in enumerate(parts)])
    stats = {"outliers": 0, "values": 0, "hidden_share": {}}
    for segment in segments:
        rows = np.array([position[int(f)] for f in segment])
        if len(rows) < 4:
            continue
        # Pose: angles unwrapped inside the run, so that no 2 pi jump is smoothed.
        x = a["mhr_model_params"][rows, :POSE].astype(np.float64)
        x[:, FIRST_ANGLE:] = np.unwrap(x[:, FIRST_ANGLE:], axis=0)
        mask = np.zeros_like(x, dtype=bool)
        if hidden is not None:
            for j, (part, side) in enumerate(parts):
                if part in hidden and side is not None:
                    mask[:, j] = hidden[part][rows, 0 if side == "l" else 1]
        y, outliers = smooth_channels(x, mask, pose_cutoff, pose_hampel, pose_floor, rate)
        b["mhr_model_params"][rows, :POSE] = y
        stats["outliers"] += int(outliers.sum())
        stats["values"] += outliers.size
        # Camera: image position and log depth, rebuilt after.
        t = a["cam_t"][rows].astype(np.float64)
        z = np.clip(t[:, 2], 0.05, None)
        cam = np.stack([t[:, 0] / z, t[:, 1] / z, np.log(z)], axis=1)
        cutoff = np.array([CUTOFF_HZ["image_xy"] * scale] * 2 + [depth_hz or CUTOFF_HZ["depth"] * scale])
        hampel = np.array([HAMPEL_S["image_xy"]] * 2 + [HAMPEL_S["depth"]])
        floor = np.array([ABS_FLOOR["image_xy"]] * 2 + [ABS_FLOOR["depth"]])
        c, outliers = smooth_channels(cam, np.zeros_like(cam, dtype=bool), cutoff, hampel, floor, rate)
        depth = np.exp(c[:, 2])
        b["cam_t"][rows] = np.stack([c[:, 0] * depth, c[:, 1] * depth, depth], axis=1)
        stats["outliers"] += int(outliers.sum())
        stats["values"] += outliers.size
    if hidden is not None:
        stats["hidden_share"] = {p: round(float(h.mean()), 3) for p, h in hidden.items()}
    return b, stats


def visibility(pipeline: Path, people: dict, archives: dict, margin: int = 10, reach: int = 12,
               references: dict | None = None, cover_min: float = 0.7, cover_score: float = 3.0) -> dict:
    """Per person: part -> (n_samples, 2) bool, True where the part is hidden (keypoints off the image or off
    the person's SAM3 mask). The track of each sample comes from people.json (the pieces given to that person).

    references: person -> RTMW reference (frames, body, body_score) of hand_reference.py. A frame whose mask holds
    less than cover_min of RTMW's confident body points is a partial mask (SAM3 kept only the sweater): its parts
    are judged on the picture only, else legs seen plainly are called hidden and frozen by the smoother."""
    import cv2

    capture = cv2.VideoCapture(str(pipeline / "input.mp4"))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    # Picture without the black bars of a letterboxed video: a point on a bar is out of the picture.
    from keypoint_view import picture_area
    left, top, right, bottom = picture_area(pipeline / "input.mp4")
    track_of = {person: {} for person in archives}  # person -> frame -> track
    for piece in people["assignment"]:
        person = str(piece["person"]) if isinstance(piece["person"], int) else None
        if person in track_of:
            for f in range(piece["frames"][0], piece["frames"][1] + 1):
                track_of[person][f] = str(piece["track"])
    needed: dict[str, dict[int, list]] = {}  # track -> frame -> [(person, sample)]
    for person, a in archives.items():
        for i, f in enumerate(a["frames"].astype(int)):
            track = track_of[person].get(int(f))
            if track is not None:
                needed.setdefault(track, {}).setdefault(int(f), []).append((person, i))
    keypoints = sorted({k for sides in VISIBILITY_KEYPOINTS.values() for side in sides for k in side})
    seen = {person: np.zeros((len(a["frames"]), 70), bool) for person, a in archives.items()}
    references = references or {}
    ref_row = {person: {int(f): r for r, f in enumerate(ref["frames"])} for person, ref in references.items()}
    partial_frames = {person: 0 for person in archives}
    by_frame: dict[int, list] = {}  # frame -> [(track, person, sample)]
    for track, frames in needed.items():
        if not (pipeline / "masks" / f"{track}.mp4").is_file():
            sys.exit(f"error: missing {pipeline / 'masks' / f'{track}.mp4'}")
        for f, items in frames.items():
            by_frame.setdefault(f, []).extend((track, person, i) for person, i in items)
    from keypoint_view import mask_frames
    for f, masks in mask_frames(pipeline, max(by_frame) if by_frame else -1, width, height):
        if f not in by_frame:
            continue
        for track, person, i in by_frame[f]:
            if track not in masks:
                sys.exit(f"error: {pipeline / 'masks' / f'{track}.mp4'} ends before frame {f}")
            mask = masks[track]
            others = np.zeros_like(mask)
            for t, mk in masks.items():
                if t != track:
                    others |= mk
            near = lambda u, v: mask[max(0, v - reach):v + reach + 1, max(0, u - reach):u + reach + 1].any()  # noqa: E731
            # A point on another person and not near the own mask: read on the other person (or hidden behind).
            other = lambda u, v: bool(others[min(max(v, 0), height - 1), min(max(u, 0), width - 1)]) and not near(u, v)  # noqa: E731
            partial = False
            r = ref_row.get(person, {}).get(f)
            if r is not None:
                pts, sc = references[person]["body"][r, :23], references[person]["body_score"][r, :23]
                ok = (sc > cover_score) & (pts[:, 0] >= left) & (pts[:, 0] < right) & (pts[:, 1] >= top) & \
                    (pts[:, 1] < bottom)
                mine = [(int(round(x)), int(round(y))) for x, y in pts[ok]]
                mine = [(x, y) for x, y in mine if not other(x, y)]  # RTMW reading the other person: no judge
                if len(mine) >= 5:
                    partial = np.mean([near(x, y) for x, y in mine]) < cover_min
                    partial_frames[person] += int(partial)
            uv = archives[person]["keypoints_2d"][i]
            for k in keypoints:
                u, v = int(round(uv[k, 0])), int(round(uv[k, 1]))
                if left + margin <= u < right - margin and top + margin <= v < bottom - margin:
                    seen[person][i, k] = (partial and not other(u, v)) or near(u, v)
    out = {}
    for person, n in partial_frames.items():
        if references:
            print(f"person {person}: {n} frames with a partial mask, judged on the picture only", flush=True)
    for person, s in seen.items():
        out[person] = {part: np.stack([s[:, sides[0]].mean(1) < 0.5, s[:, sides[1]].mean(1) < 0.5], axis=1)
                       for part, sides in VISIBILITY_KEYPOINTS.items()}
    return out
