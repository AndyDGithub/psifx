"""Simulation, no GPU: key frames chosen by the model itself, with a stress test for a moving camera.

Start from coarse key frames inside each shot, compute the middle of an interval, and split it again only if the model
disagrees with the interpolation. Two tests decide:
- body: largest usual joint deviation (p95, mm) around the pelvis, the global rotation removed (Kabsch), so that a camera
  moving around a still person does not count;
- place: deviation of the pelvis in the image, in share of the body size, because the render depends on it.

Every frame is computed at most once, its result being reused: the worst case is the plain run, plus the tests.
Synthetic camera: a smooth rotation and image shift added to the archives, to see what makes the method split.

    python research/scene3d/keyframe_recursive.py .e2e_outputs/extrait_talkshow --others .e2e_outputs/dev_2026-10-27/v4_dance
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / "research" / "pose_annotation")]
from evaluate import evaluate, load_ground_truth  # noqa: E402

C17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
WRISTS = [62, 41]
W, H = 320, 180


def cut_labels(video: Path) -> np.ndarray:
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                               stdout=subprocess.PIPE)
    previous, differences = None, []
    while len(buffer := process.stdout.read(W * H)) == W * H:
        frame = np.frombuffer(buffer, np.uint8).astype(np.int16)
        if previous is not None:
            differences.append(np.abs(frame - previous).mean())
        previous = frame
    if process.wait() != 0:
        sys.exit(f"error: ffmpeg failed on {video}")
    return np.r_[0, np.cumsum(np.array(differences) > 18)]


def load(folder: Path, labels: np.ndarray) -> list[dict]:
    people = []
    for path in sorted((folder / "sam3d_body").glob("*.npz")):
        a = np.load(path)
        if "frames" not in a.files:
            continue
        frames, k3 = a["frames"], a["keypoints_3d"]
        pelvis = (k3[:, 9] + k3[:, 10]) / 2
        step = int(np.median(np.diff(frames)))
        people.append(dict(name=path.stem, frames=frames, k2=a["keypoints_2d"].astype(np.float64), rel=(k3 - pelvis[:, None]).astype(np.float64), step=step))
    for p in people:
        f = p["frames"]
        ends = np.where((np.diff(f) != p["step"]) | (labels[np.minimum(f[1:], len(labels) - 1)] != labels[np.minimum(f[:-1], len(labels) - 1)]))[0] + 1
        p["segments"] = np.split(np.arange(len(f)), ends)
    return people


def camera(people: list[dict], degrees_per_s: float, shift_px: float, period_s: float, fps: float) -> list[dict]:
    """The same people seen by a camera turning smoothly: the body turns in the camera frame, and the person slides in the image."""
    out = []
    for p in people:
        t = p["frames"] / fps
        angle = np.radians(degrees_per_s) * period_s / (2 * np.pi) * np.sin(2 * np.pi * t / period_s)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.zeros((len(t), 3, 3))
        rotation[:, 0, 0], rotation[:, 0, 2], rotation[:, 1, 1], rotation[:, 2, 0], rotation[:, 2, 2] = c, s, 1, -s, c
        k2 = p["k2"].copy()
        k2[:, :, 0] += (shift_px * np.sin(2 * np.pi * t / period_s))[:, None]
        out.append({**p, "rel": np.einsum("nij,nkj->nki", rotation, p["rel"]), "k2": k2})
    return out


def body_deviation(estimate: np.ndarray, target: np.ndarray, remove_rotation: bool) -> float:
    if remove_rotation:
        u, _, vt = np.linalg.svd(estimate.T @ target)
        d = np.sign(np.linalg.det(u @ vt))
        estimate = estimate @ (u @ np.diag([1, 1, d]) @ vt)
    return float(np.percentile(np.linalg.norm(estimate - target, axis=1), 95) * 1000)


def simulate(people, start_frames: int, tau_mm: float | None, tau_place: float | None, remove_rotation: bool, floor: int = 1):
    """floor: an interval of at most this many processed frames is never split: the budget. Returns the 2D points too."""
    computed = total = 0
    wrist = []
    points = {}
    for p in people:
        k2, rel = p["k2"], p["rel"]
        out2, out3 = k2.copy(), rel.copy()
        spacing = max(start_frames // p["step"], 1)
        for seg in p["segments"]:
            total += len(seg)
            keys = list(seg[::spacing])
            if keys[-1] != seg[-1]:
                keys.append(seg[-1])
            computed += len(keys)
            stack = list(zip(keys, keys[1:]))
            while stack:
                a, b = stack.pop()
                if b - a < 2:
                    continue
                if b - a <= floor:  # Budget reached: interpolate without asking the model.
                    weights = ((np.arange(a + 1, b) - a) / (b - a))[:, None, None]
                    out2[a + 1:b] = k2[a] * (1 - weights) + k2[b] * weights
                    out3[a + 1:b] = rel[a] * (1 - weights) + rel[b] * weights
                    continue
                m = (a + b) // 2
                computed += 1
                w = (m - a) / (b - a)
                body = body_deviation(rel[a] * (1 - w) + rel[b] * w, rel[m], remove_rotation)
                size = max(float(np.linalg.norm(k2[m].max(0) - k2[m].min(0))), 1.0)
                place = np.linalg.norm((k2[a, 9:11].mean(0)) * (1 - w) + k2[b, 9:11].mean(0) * w - k2[m, 9:11].mean(0)) / size
                split = (tau_mm is None or body > tau_mm) or (tau_place is not None and place > tau_place)
                if split:
                    stack += [(a, m), (m, b)]
                    continue
                for x, y in ((a, m), (m, b)):  # Accepted: the middle is known, so the two halves are interpolated.
                    if y - x > 1:
                        weights = ((np.arange(x + 1, y) - x) / (y - x))[:, None, None]
                        out2[x + 1:y] = k2[x] * (1 - weights) + k2[y] * weights
                        out3[x + 1:y] = rel[x] * (1 - weights) + rel[y] * weights
        difference = np.linalg.norm(out3 - rel, axis=2) * 1000
        changed = (out3 != rel).any(axis=(1, 2))
        if changed.any():
            wrist.append(difference[changed][:, WRISTS].ravel())
        points[p["name"]] = out2
    wrist = np.concatenate(wrist) if wrist else np.zeros(1)
    return dict(computed=round(computed / total, 3), speedup=round(total / computed, 2), wrist_p50=round(float(np.median(wrist)), 1), wrist_p95=round(float(np.percentile(wrist, 95)), 1)), points


def oks(people, points, annotations: Path) -> float:
    predictions = {}
    with open(annotations / "frames.csv", newline="") as f:
        for row in csv.DictReader(f):
            frame, found = int(row["frame"]), []
            for p in people:
                for candidate in (frame, frame - 1):
                    i = np.searchsorted(p["frames"], candidate)
                    if i < len(p["frames"]) and p["frames"][i] == candidate:
                        found.append(np.c_[points[p["name"]][i][C17], np.ones(17)])
                        break
            predictions[row["file"]] = found
    return round(float(evaluate(predictions, load_ground_truth(annotations / "ground_truth.json"), 0.3)["oks"]), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with pipeline/ and pose_annotation/")
    parser.add_argument("--others", nargs="*", type=Path, default=[], help="other pipeline folders, without annotations")
    args = parser.parse_args()
    pipeline = args.folder / "pipeline"
    if not (pipeline / "sam3d_body").is_dir():
        sys.exit(f"error: no archives in {pipeline}")
    people = load(pipeline, cut_labels(pipeline / "input.mp4"))
    result = {}
    print("-- talk show, start 40, split only when the model disagrees")
    for tau in (40, 60, 80, 120):
        r, points = simulate(people, 40, tau, 0.02, True)
        r["oks"] = oks(people, points, args.folder / "pose_annotation")
        result[f"talkshow_tau{tau}"] = r
        print(tau, r, flush=True)
    print("-- worst case: every test says split")
    r, _ = simulate(people, 40, 0.0, None, True)
    result["worst_case"] = r
    print(r)
    print("-- worst case with a budget: never denser than 1 frame out of 4, then 1 out of 8 (of the video)")
    for floor in (2, 4):
        r, points = simulate(people, 40, 0.0, None, True, floor=floor)
        r["oks"] = oks(people, points, args.folder / "pose_annotation")
        result[f"worst_case_floor_{floor}"] = r
        print(floor, r, flush=True)
    print("-- synthetic smooth camera (tau 60)")
    for degrees, shift, period in ((30, 300, 8), (90, 300, 4), (90, 600, 2), (180, 600, 1)):
        moved = camera(people, degrees, shift, period, 60)
        with_body, _ = simulate(moved, 40, 60, 0.02, True)
        no_removal, _ = simulate(moved, 40, 60, 0.02, False)
        body_only, _ = simulate(moved, 40, 60, None, True)
        key = f"{degrees}deg_s_{shift}px_{period}s"
        result[key] = dict(body_and_place=with_body["speedup"], body_only=body_only["speedup"], without_rotation_removal=no_removal["speedup"])
        print(key, result[key], flush=True)
    for folder in args.others:
        others = load(folder, cut_labels(folder / "input.mp4"))
        step = others[0]["step"]
        lengths = [len(s) for p in others for s in p["segments"]]
        print(folder.name, "segments", len(lengths), "median length", int(np.median(lengths)), "longest", max(lengths), flush=True)
        for tau in (60, 80):
            r, _ = simulate(others, max(40, 8 * step), tau, 0.02, True)
            result[f"{folder.name}_tau{tau}"] = {**r, "archive_step": step}
            print(folder.name, tau, result[f"{folder.name}_tau{tau}"], flush=True)
    (args.folder / "keyframe_recursive.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
