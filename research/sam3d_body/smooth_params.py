"""Steadier meshes: one body per person, and the pose smoothed inside each shot, never across a cut.

Variants, each measured against the raw archives:
- ``lock``: identity (``shape_params``) and limb scales (the last 68 ``mhr_model_params``) replaced by the person's
  median over all its frames: a person does not change body;
- ``lock+sg<w>``: plus a Savitzky-Golay filter (order 2, ``w`` frames) on the pose (the first 136
  ``mhr_model_params``, angles unwrapped), the expression and the camera translation;
- ``lock+euro:<min_cutoff>:<beta>``: plus a One Euro filter (adaptive: strong on slow parts, weak on fast gestures), run forward and
  backward and averaged so that it does not lag.
A run shorter than the filter window is left as it is.

Measures, on keypoints rebuilt from the mesh (MHR gives the vertices only): each COCO keypoint is a fixed
combination of its 16 nearest vertices, fitted per person on the raw frames by least squares (the fit error is
reported). Then, per variant:
- ``jitter_mm``: median acceleration of the 17 keypoints around the pelvis;
- ``change_mm`` p50 / p95: distance to the raw keypoints, on all frames and on the 10 % fastest wrist frames
  (``gestures``): smoothing must not erase gestures;
- ``oks``: the rebuilt keypoints projected with the archive camera, against the talk-show annotations; ``raw`` also
  gives the OKS of the archive's own 2D keypoints, to see what the rebuild costs.

Writes ``<pipeline>/smoothing.json`` and, for ``--keep``, the archives of that variant in ``<pipeline>/<keep>/``
(same format, so that ``render_avatars.py --params`` takes them).

    python research/sam3d_body/smooth_params.py .e2e_outputs/extrait_talkshow/pipeline \\
        --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith --annotations .e2e_outputs/extrait_talkshow/pose_annotation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_eval import annotations  # noqa: E402
from run_keyframes import cut_labels, split_segments  # noqa: E402
from run_tracks import MHR70_TO_COCO17  # noqa: E402

POSE, FIRST_ANGLE = 136, 3  # mhr_model_params: 3 translations, 133 angles, then 68 scales.
LEFT_HIP, RIGHT_HIP = 11, 12  # COCO indices.
WRISTS = [9, 10]
NEIGHBORS = 16


def one_euro(x: np.ndarray, rate: float, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    """One Euro filter along axis 0, every column at once (Casiez et al., CHI 2012)."""
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * rate)
    out, previous, speed = np.empty_like(x), x[0].copy(), np.zeros_like(x[0])
    out[0] = previous
    for t in range(1, len(x)):
        a_d = alpha(d_cutoff)
        speed = a_d * (x[t] - previous) * rate + (1 - a_d) * speed
        a = alpha(min_cutoff + beta * np.abs(speed))
        previous = a * x[t] + (1 - a) * previous
        out[t] = previous
    return out


def smooth(a: dict, segments: list[np.ndarray], method: str, rate: float) -> dict:
    """Locked body, then the pose, expression and camera filtered run by run."""
    b = {k: v.copy() for k, v in a.items()}
    b["shape_params"][:] = np.median(a["shape_params"], axis=0)
    b["mhr_model_params"][:, POSE:] = np.median(a["mhr_model_params"][:, POSE:], axis=0)
    if method == "lock":
        return b
    position = {int(f): i for i, f in enumerate(a["frames"])}
    for segment in segments:
        rows = np.array([position[int(f)] for f in segment])
        for key, first_angle, last in (("mhr_model_params", FIRST_ANGLE, POSE), ("expr_params", None, None),
                                       ("cam_t", None, None)):
            x = a[key][rows][:, :last].astype(np.float64)
            if first_angle is not None:
                x[:, first_angle:] = np.unwrap(x[:, first_angle:], axis=0)  # No 2 pi jump inside a run.
            if method.startswith("lock+sg"):
                window = int(method[len("lock+sg"):])
                if len(rows) < window:
                    continue
                y = savgol_filter(x, window, 2, axis=0)
            else:
                _, min_cutoff, beta = method.split(":")
                forward = one_euro(x, rate, float(min_cutoff), float(beta))
                backward = one_euro(x[::-1], rate, float(min_cutoff), float(beta))[::-1]
                y = (forward + backward) / 2
            b[key][rows, :last] = y
    return b


@torch.inference_mode()
def meshes(a: dict, mhr, device: str, batch: int = 256) -> np.ndarray:
    out = []
    for s in range(0, len(a["frames"]), batch):
        v = mhr(*(torch.from_numpy(a[k][s:s + batch]).to(device).float()
                  for k in ("shape_params", "mhr_model_params", "expr_params")))[0].cpu().numpy() / 100
        v[..., [1, 2]] *= -1  # MHR axes to the camera axes, as in mesh_eval.py.
        out.append(v.astype(np.float32))
    return np.concatenate(out)


def fit_regressor(vertices: np.ndarray, keypoints: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, float]:
    """Per keypoint, the 16 nearest vertices (median frame) and their least-squares weights; returns the fit error."""
    sample = rng.choice(len(vertices), min(400, len(vertices)), replace=False)
    rest_v, rest_k = np.median(vertices[sample], axis=0), np.median(keypoints[sample], axis=0)
    nearest = np.argsort(np.linalg.norm(rest_v[None] - rest_k[:, None], axis=2), axis=1)[:, :NEIGHBORS]
    weights = np.zeros((len(rest_k), NEIGHBORS))
    for k in range(len(rest_k)):
        design = vertices[sample][:, nearest[k]].transpose(0, 2, 1).reshape(-1, NEIGHBORS)  # (frames*3, 16)
        weights[k] = np.linalg.lstsq(design, keypoints[sample][:, k].reshape(-1), rcond=None)[0]
    error = np.linalg.norm(apply_regressor(vertices, nearest, weights) - keypoints, axis=2)
    return nearest, weights, float(np.median(error) * 1000)


def apply_regressor(vertices: np.ndarray, nearest: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("fknc,kn->fkc", vertices[:, nearest], weights)


def measures(k3: np.ndarray, raw: np.ndarray, fast: np.ndarray) -> dict:
    relative = k3 - k3[:, [LEFT_HIP, RIGHT_HIP]].mean(1, keepdims=True)
    raw_relative = raw - raw[:, [LEFT_HIP, RIGHT_HIP]].mean(1, keepdims=True)
    accel = np.linalg.norm(relative[2:] - 2 * relative[1:-1] + relative[:-2], axis=2).mean(1) * 1000
    change = np.linalg.norm(relative - raw_relative, axis=2) * 1000
    return {"jitter": accel, "change": change.ravel(), "gestures": change[fast][:, WRISTS].ravel()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4 and the archives")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with assets/mhr_model.pt")
    parser.add_argument("--params", default="sam3d_params")
    parser.add_argument("--annotations", type=Path, help="folder with frames.csv and ground_truth.json")
    parser.add_argument("--variants", nargs="+", default=["lock", "lock+sg7", "lock+sg15",
                                                          "lock+euro:1.0:0.05", "lock+euro:0.5:0.02"])
    parser.add_argument("--keep", help="variant to write as archives, in <pipeline>/sam3d_smooth")
    parser.add_argument("--grid", type=int, default=2, help="frame step of the archives")
    parser.add_argument("--cut", type=float, default=18)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    folder, video = args.pipeline / args.params, args.pipeline / "input.mp4"
    model = args.checkpoint / "assets" / "mhr_model.pt"
    for path in (folder, video, model):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    if args.keep and args.keep not in args.variants:
        sys.exit(f"error: --keep {args.keep} is not in --variants")
    archives = {}
    for path in sorted(folder.glob("*.npz"), key=lambda p: int(p.stem)):
        with np.load(path) as npz:
            archives[path.stem] = {k: npz[k] for k in npz.files}
    mhr = torch.jit.load(str(model), map_location=args.device)
    labels = cut_labels(video, args.cut)
    import cv2
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    rate = fps / args.grid  # Samples per second in the archives.
    rng = np.random.default_rng(0)

    pooled: dict[str, dict[str, list]] = {v: {"jitter": [], "change": [], "gestures": []} for v in ["raw"] + args.variants}
    projected: dict[str, dict] = {v: {} for v in ["raw"] + args.variants}
    report: dict = {"fit_error_mm_p50": {}, "rate_hz": rate}
    for person, a in archives.items():
        frames = a["frames"].astype(int)
        if frames.max() >= len(labels):
            sys.exit(f"error: person {person} has frame {frames.max()}, the video has {len(labels)}")
        segments = split_segments(frames, args.grid, labels)
        raw_vertices = meshes(a, mhr, args.device)
        target = a["keypoints_3d"][:, MHR70_TO_COCO17]
        nearest, weights, error = fit_regressor(raw_vertices, target, rng)
        report["fit_error_mm_p50"][person] = round(error, 1)
        raw = apply_regressor(raw_vertices, nearest, weights)
        relative = raw - raw[:, [LEFT_HIP, RIGHT_HIP]].mean(1, keepdims=True)
        speed = np.linalg.norm(np.diff(relative[:, WRISTS], axis=0, prepend=relative[:1, WRISTS]), axis=2).max(1)
        fast = speed >= np.percentile(speed, 90)
        del raw_vertices
        for variant in ["raw"] + args.variants:
            b = a if variant == "raw" else smooth(a, segments, variant, rate)
            k3 = raw if variant == "raw" else apply_regressor(meshes(b, mhr, args.device), nearest, weights)
            for key, value in measures(k3, raw, fast).items():
                pooled[variant][key].append(value)
            camera = k3 + b["cam_t"][:, None]
            focal = b["focal"][:, None]
            z = np.clip(camera[..., 2], 1e-3, None)
            two_d = np.full((len(k3), 70, 2), np.nan, np.float32)
            two_d[:, MHR70_TO_COCO17, 0] = focal * camera[..., 0] / z + 1920 / 2
            two_d[:, MHR70_TO_COCO17, 1] = focal * camera[..., 1] / z + 1080 / 2
            projected[variant][person] = {"frames": a["frames"], "keypoints_2d": two_d}
            if variant == args.keep:
                out = args.pipeline / "sam3d_smooth"
                out.mkdir(exist_ok=True)
                np.savez_compressed(out / f"{person}.npz", **b)
        print(person, "done", flush=True)

    for variant, values in pooled.items():
        jitter, change, gestures = (np.concatenate(values[k]) for k in ("jitter", "change", "gestures"))
        entry = {"jitter_mm_p50": round(float(np.median(jitter)), 1),
                 "change_mm_p50_p95": [round(float(v), 1) for v in np.percentile(change, [50, 95])],
                 "gestures_change_mm_p50_p95": [round(float(v), 1) for v in np.percentile(gestures, [50, 95])]}
        if args.annotations:
            entry["oks"] = round(annotations(projected[variant], args.annotations)["oks"], 3)
        report[variant] = entry
        print(variant, json.dumps(entry), flush=True)
    if args.annotations:
        report["raw_archive_keypoints_oks"] = round(annotations(archives, args.annotations)["oks"], 3)
    (args.pipeline / "smoothing.json").write_text(json.dumps(report, indent=1))
    print("SMOOTH_OK")


if __name__ == "__main__":
    main()
