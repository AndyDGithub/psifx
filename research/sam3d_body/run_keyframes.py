"""SAM 3D Body on key frames only: the model itself decides where it is needed, the rest is interpolated.

Inside each shot, one frame every ``--start`` is computed. Then, round after round, the middle of each interval is
computed, and the interval is split again only if the model contradicts the interpolation:
- body: p95 deviation (mm) of the joints around the pelvis, global rotation removed (Kabsch), against ``--tau_body``;
- place: deviation of the pelvis in the image, in share of the body size, against ``--tau_place``.

A frame is computed at most once. An interval of at most ``--floor`` frames is never split: this bounds the cost.
At ``--max_seconds`` the refinement stops and what is left is interpolated, so a run always ends with usable archives.

Writes ``<out_dir>/<output_name>/<id>.npz``, the format of ``run_tracks.py`` on every ``--grid``-th frame, plus
``computed`` (N,), false on interpolated frames, and ``keyframes_run.json`` (times and counts).

    python research/sam3d_body/run_keyframes.py .e2e_outputs/extrait_talkshow/pipeline \\
        --repo .e2e_outputs/ext/sam-3d-body --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MOMENTUM_ENABLED", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tracks import PARAMETERS, load_estimator, mask_box  # noqa: E402

LEFT_HIP, RIGHT_HIP = 9, 10
GEOMETRY = ("keypoints_2d", "keypoints_3d", "cam_t", "focal")
ANGLES = {"global_rot": 0, "body_pose_params": 0, "mhr_model_params": 3}  # First angle index: the rest is linear.


def cut_labels(video: Path, threshold: float, min_ratio: float = 3.0) -> np.ndarray:
    """Shot number of every frame, from the mean gray difference between frames.

    A cut is a difference above ``threshold`` **and** at least ``min_ratio`` times the mean of its 2 + 2 neighbours
    (the rolling average of PySceneDetect's AdaptiveDetector). A fast camera pan, a dancer crossing the frame or a
    flash fading out give a run of high differences, not one spike: with the threshold alone, V5 frames 271-299
    became 15 one-frame shots, with no smoothing and no temporal link in between (the push out of the frame lost).
    min_ratio 0: threshold only, as before."""
    w, h = 320, 180
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                                "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    previous, differences = None, []
    while len(buffer := process.stdout.read(w * h)) == w * h:
        frame = np.frombuffer(buffer, np.uint8).astype(np.int16)
        if previous is not None:
            differences.append(np.abs(frame - previous).mean())
        previous = frame
    if process.wait() != 0:
        sys.exit(f"error: ffmpeg failed on {video}")
    d = np.array(differences)
    cut = d > threshold
    if min_ratio > 0 and len(d):
        padded = np.pad(d, 2, mode="edge")
        neighbours = (padded[:-4] + padded[1:-3] + padded[3:-1] + padded[4:]) / 4
        cut &= d >= min_ratio * np.maximum(neighbours, 1e-3)
    return np.r_[0, np.cumsum(cut)]


def split_segments(frames: np.ndarray, grid: int, labels: np.ndarray) -> list[np.ndarray]:
    """Runs of frames without a gap or a cut: nothing is interpolated across them."""
    breaks = np.where((np.diff(frames) != grid) | (labels[frames[1:]] != labels[frames[:-1]]))[0] + 1
    return np.split(frames, breaks)


def interpolate(a, b, w: float, first_angle: int | None = None):
    """Linear, by the shortest way for the angles (a wrap of 2 pi would otherwise turn the whole limb)."""
    delta = b - a
    if first_angle is not None:
        delta = delta.copy()
        delta[first_angle:] = (delta[first_angle:] + np.pi) % (2 * np.pi) - np.pi
    return a + delta * w


def densify(known: dict[int, dict], segments: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Archive arrays on all the frames of the segments, the frames in ``known`` kept as computed."""
    frames, computed, rows = [], [], []
    for segment in segments:
        keys = [int(f) for f in segment if int(f) in known]
        for f in segment:
            f = int(f)
            frames.append(f), computed.append(f in known)
            if f in known:
                rows.append(known[f])
                continue
            after = int(np.searchsorted(keys, f))
            x, y = keys[after - 1], keys[after]
            w = (f - x) / (y - x)
            rows.append({k: interpolate(known[x][k], known[y][k], w, ANGLES.get(k)) for k in known[x]})
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]} if rows else {}
    return {"frames": np.array(frames, int), "computed": np.array(computed, bool), **arrays}


def body_deviation(estimate: np.ndarray, target: np.ndarray) -> float:
    """p95 joint distance in mm around the pelvis. The rotation is removed: a moving camera must not count."""
    u, _, vt = np.linalg.svd(estimate.T @ target)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.percentile(np.linalg.norm(estimate @ (u @ np.diag([1, 1, d]) @ vt) - target, axis=1), 95) * 1000)


def contradicts(a: dict, m: dict, b: dict, w: float, tau_body: float, tau_place: float) -> bool:
    relative = [r["keypoints_3d"] - r["keypoints_3d"][[LEFT_HIP, RIGHT_HIP]].mean(0) for r in (a, m, b)]
    if body_deviation(relative[0] * (1 - w) + relative[2] * w, relative[1]) > tau_body:
        return True
    pelvis = [r["keypoints_2d"][[LEFT_HIP, RIGHT_HIP]].mean(0) for r in (a, m, b)]
    size = max(float(np.linalg.norm(m["keypoints_2d"].max(0) - m["keypoints_2d"].min(0))), 1.0)
    return float(np.linalg.norm(pelvis[0] * (1 - w) + pelvis[2] * w - pelvis[1])) / size > tau_place


def presence(mask_paths: dict[str, Path], grid: int, last_frame: int, min_area: int) -> dict[str, np.ndarray]:
    """Frames of the grid where each person has a mask big enough for a mesh."""
    found = {}
    for person, path in mask_paths.items():
        capture, frames = cv2.VideoCapture(str(path)), []
        for i in range(last_frame):
            if not capture.grab():
                break
            if i % grid == 0:
                ok, mask = capture.retrieve()
                if not ok:
                    sys.exit(f"error: cannot decode frame {i} of {path}")
                if int((mask[..., 0] > 127).sum()) >= min_area:
                    frames.append(i)
        capture.release()
        found[person] = np.array(frames, int)
    return found


class Model:
    """Runs the estimator on sorted requests, reading the videos forward once per round."""

    def __init__(self, estimator, video: Path, mask_paths: dict[str, Path], inference_type: str, deadline: float):
        self.estimator, self.video, self.mask_paths = estimator, video, mask_paths
        self.inference_type, self.deadline = inference_type, deadline
        self.calls, self.seconds = 0, 0.0

    def run(self, requests: dict[int, list[str]], results: dict[str, dict]) -> bool:
        """Fills ``results``. Returns False when the deadline stopped the round."""
        start = time.time()
        captures = {name: cv2.VideoCapture(str(path)) for name, path in
                    {"video": self.video, **self.mask_paths}.items()}
        try:
            for frame in sorted(requests):
                if time.time() > self.deadline:
                    return False
                people = sorted(requests[frame], key=int)
                images = {}
                for name in ("video", *people):
                    captures[name].set(cv2.CAP_PROP_POS_FRAMES, frame)  # Exact on these files (checked against reading).
                    ok, images[name] = captures[name].read()
                    if not ok:
                        sys.exit(f"error: cannot read frame {frame} of {name}")
                masks = [images[p][..., 0] > 127 for p in people]
                boxes = [mask_box(m, 1) for m in masks]
                if any(b is None for b in boxes):
                    sys.exit(f"error: empty mask for {people} at frame {frame}")
                inference = time.time()
                with open(os.devnull, "w") as null:  # The estimator prints a line per call.
                    stdout, sys.stdout = sys.stdout, null
                    try:
                        # cv2 gives BGR, and the psifx reader used by run_tracks.py gives RGB.
                        out = self.estimator.process_one_image(images["video"][..., ::-1].copy(), bboxes=np.stack(boxes),
                                                               masks=np.stack(masks), inference_type=self.inference_type)
                    finally:
                        sys.stdout = stdout
                self.seconds += time.time() - inference
                self.calls += 1
                if len(out) != len(people):
                    sys.exit(f"error: {len(out)} results for {len(people)} people at frame {frame}")
                for person, o in zip(people, out):
                    results[person][frame] = {
                        "keypoints_2d": o["pred_keypoints_2d"][:, :2].astype(np.float32),
                        "keypoints_3d": np.asarray(o["pred_keypoints_3d"], np.float32),
                        "cam_t": np.asarray(o["pred_cam_t"], np.float32),
                        "focal": np.asarray(float(o["focal_length"])),
                        **{name: np.asarray(o[name], np.float32) for name in PARAMETERS}}
        finally:
            for capture in captures.values():
                capture.release()
        print(f"  round: {len(requests)} frames in {time.time() - start:.0f} s", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder with input.mp4 and masks/")
    parser.add_argument("--repo", type=Path, required=True, help="clone of facebookresearch/sam-3d-body")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with model.ckpt and assets/")
    parser.add_argument("--grid", type=int, default=2, help="frames of the result: one out of N")
    parser.add_argument("--start", type=int, default=40, help="first key frames: one out of N")
    parser.add_argument("--floor", type=int, default=8, help="an interval of at most N frames is never split")
    parser.add_argument("--tau_body", type=float, default=80, help="body deviation that splits, in mm")
    parser.add_argument("--tau_place", type=float, default=0.02, help="pelvis deviation that splits, in share of the body")
    parser.add_argument("--max_seconds", type=float, default=float("inf"), help="stop refining after this time")
    parser.add_argument("--cut", type=float, default=18, help="mean gray difference that starts a new shot")
    parser.add_argument("--min_area", type=int, default=400, help="smallest mask, in pixels, worth a mesh")
    parser.add_argument("--last_frame", type=int, default=10 ** 9, help="stop there, for a quick test")
    parser.add_argument("--inference_type", default="full", choices=["full", "body"])
    parser.add_argument("--output_name", default="sam3d_keyframes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.start % args.grid or args.floor < args.grid:
        sys.exit("error: --start must be a multiple of --grid, and --floor at least --grid")
    video, mask_dir, output = args.out_dir / "input.mp4", args.out_dir / "masks", args.out_dir / args.output_name
    for path in (video, mask_dir, args.checkpoint / "model.ckpt", args.repo / "sam_3d_body"):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    mask_paths = {p.stem: p for p in sorted(mask_dir.glob("*.mp4"), key=lambda p: int(p.stem))}
    if not mask_paths:
        sys.exit(f"error: no mask in {mask_dir}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        sys.exit(f"error: {output} is not empty, pass --overwrite to replace it")
    output.mkdir(exist_ok=True)

    begin = time.time()
    labels = cut_labels(video, args.cut)
    present = presence(mask_paths, args.grid, min(args.last_frame, len(labels)), args.min_area)
    segments = {p: split_segments(f, args.grid, labels) if len(f) else [] for p, f in present.items()}
    prepare = time.time() - begin
    total = sum(len(f) for f in present.values())
    print(f"prepared in {prepare:.0f} s: {total} person-frames, {labels[-1] + 1} shots", flush=True)

    estimator = load_estimator(args.repo, args.checkpoint, args.device)
    model = Model(estimator, video, mask_paths, args.inference_type, begin + args.max_seconds)
    results: dict[str, dict] = {p: {} for p in present}
    requests: dict[int, list[str]] = {}
    intervals = []
    spacing = args.start // args.grid
    for person, parts in segments.items():
        for segment in parts:
            keys = [*segment[::spacing].tolist()]
            if keys[-1] != int(segment[-1]):
                keys.append(int(segment[-1]))
            for f in keys:
                requests.setdefault(f, []).append(person)
            intervals += [(person, x, y) for x, y in zip(keys, keys[1:])]

    def save(rounds: int, stopped: bool):
        done = 0
        for person, known in results.items():
            done += len(known)
            parts = [s for s in segments[person] if sum(int(f) in known for f in (s[0], s[-1])) == 2]
            arrays = densify(known, parts)
            np.savez_compressed(output / f"{person}.npz", faces=estimator.faces, **arrays)
        (output / "keyframes_run.json").write_text(json.dumps({
            "person_frames": total, "computed": done, "share": round(done / max(total, 1), 4),
            "model_calls": model.calls, "model_seconds": round(model.seconds), "prepare_seconds": round(prepare),
            "total_seconds": round(time.time() - begin), "rounds": rounds, "stopped_by_deadline": stopped,
            "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))

    finished = model.run(requests, results)
    rounds = 1
    if not finished:
        sys.exit("error: --max_seconds is too short for the first key frames, nothing usable was produced")
    print(f"round 0: {sum(len(r) for r in results.values())}/{total} person-frames", flush=True)
    while finished:
        todo = []
        for person, x, y in intervals:
            if y - x > args.floor:
                todo.append((person, x, x + ((y - x) // (2 * args.grid)) * args.grid, y))
        if not todo:
            break
        requests = {}
        for person, _, m, _ in todo:
            requests.setdefault(m, []).append(person)
        finished = model.run(requests, results)
        intervals = []
        for person, x, m, y in todo:
            known = results[person]
            if m in known and contradicts(known[x], known[m], known[y], (m - x) / (y - x), args.tau_body, args.tau_place):
                intervals += [(person, x, m), (person, m, y)]
        rounds += 1
        print(f"round {rounds - 1}: {sum(len(r) for r in results.values())}/{total} person-frames, "
              f"{len(intervals)} intervals to split, {time.time() - begin:.0f} s", flush=True)
        save(rounds, not finished)  # A crash later still leaves usable archives.
    save(rounds, not finished)
    print("KEYFRAMES_OK" if finished else "KEYFRAMES_OK stopped by --max_seconds", flush=True)


if __name__ == "__main__":
    main()
