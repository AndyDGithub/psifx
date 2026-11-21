"""Avatar hands fitted to the 2D hand detectors, over the smoothed SAM 3D Body parameters.

SAM 3D Body's hands are noisy, and the robust smoother (``robust_smooth.py``) that removes the noise also makes the
hands lag on fast gestures (up to about 100 px for a few frames). This script moves only the arm, wrist and finger
angles of MHR so that the 21 projected points of each hand land on the 2D reference of ``hand_reference.py``, frame
by frame, with:

- a **data term**: robust (Geman-McClure) distance of each projected hand point to the reference, in hand sizes,
  weighted by the detector's confidence (points under ``--min_score`` do not count, nor points out of the picture
  or off the person's SAM3 mask: detectors put hands out of the frame on its border, ``keypoint_view.py``);
- a **prior**: the change from the smoothed angles costs, more for the arms than for the fingers, so a hand the
  detectors do not see stays where the smoother put it;
- a **temporal term**: the acceleration of that change costs, so the detectors' own jitter is not copied.

The MHR70 hand points are skeleton joints (one joint per row of SAM 3D Body's ``keypoint_mapping``), and the MHR
skeleton does not depend on the body shape, so the fit runs on the skeleton alone, without skinning the mesh.

``--source`` picks the reference the fit uses (``hand``: RTMPose hand on the crop where it is confident, else RTMW;
``rtmw``: RTMW only, the default; ``hand_only``) and ``--evaluate`` measures input and output against each source:
fitted on RTMW, the hand model stays an independent judge (on the talk-show, frames 8900-9100, the error to it falls
from 0.28 / 0.14 to 0.07 / 0.05 hand size with ``--temporal 2e4``, the hands as still as the smoother's at 1e5).

    python research/sam3d_body/hand_refit.py .e2e_outputs/extrait_talkshow/pipeline --checkpoint \\
        .e2e_outputs/ext/ckpt/sam-3d-body-vith --params sam3d_final --output sam3d_hands
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_smooth import parameter_parts  # noqa: E402
from run_keyframes import cut_labels  # noqa: E402
import keypoint_view  # noqa: E402

POSE = 136
# Skeleton joint of each MHR70 hand point in COCO-WholeBody hand order (wrist, then thumb, index, middle, ring, pinky
# from base to tip); found from the rows of SAM 3D Body's keypoint_mapping (one joint each).
HAND_JOINTS = {
    "l": [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83],
    "r": [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47],
}
RTMW_HANDS = {"l": slice(91, 112), "r": slice(112, 133)}
PRIOR = {"arms": 4.0, "wrists": 1.0, "fingers": 0.3}  # cost of a change of 1 rad, per part
GM_SCALE = 0.15  # Geman-McClure scale, in hand sizes: farther points count less and less (detector errors)


def reference(ref: dict, source: str, min_score: float, view: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2 sides l/r, 21, 2) target points and (n, 2, 21) weights from the hand_reference.py archive.

    ``view``: {"rtmw": (n, 2, 21) bool, "hand": (n, 2, 21) bool} from keypoint_view.in_view; points out of view
    (out of the picture or off the person's mask) get a zero weight."""
    rtmw = np.stack([ref["body"][:, RTMW_HANDS[s]] for s in "lr"], 1)
    rtmw_w = np.stack([ref["body_score"][:, RTMW_HANDS[s]] for s in "lr"], 1)
    hand, hand_w = ref["hands"], ref["hand_score"]
    if view is not None:
        rtmw_w, hand_w = rtmw_w * view["rtmw"], hand_w * view["hand"]
    if source == "rtmw":
        points, weights = rtmw, rtmw_w
    elif source == "hand_only":
        points, weights = hand, hand_w
    else:  # hand model where it is confident, RTMW elsewhere
        use_hand = hand_w > min_score
        points = np.where(use_hand[..., None], hand, rtmw)
        weights = np.where(use_hand, hand_w, rtmw_w)
    weights = np.where((weights > min_score) & np.isfinite(points).all(-1), weights, 0.0)
    return np.nan_to_num(points), weights.astype(np.float32)


def skeleton(mhr, params: torch.Tensor) -> torch.Tensor:
    """(n, 127, 3) joint positions in cm, MHR axes, without skinning the mesh (the skeleton ignores the shape)."""
    c = mhr.character_torch
    joint_params = c.model_parameters_to_joint_parameters(
        torch.cat([params, torch.zeros(len(params), 45, device=params.device)], 1))
    return c.joint_parameters_to_skeleton_state(joint_params)[..., :3]


def project(joints_cm: torch.Tensor, cam_t: torch.Tensor, focal: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    j = joints_cm / 100 * torch.tensor([1.0, -1.0, -1.0], device=joints_cm.device) + cam_t[:, None]
    return focal[:, None, None] * j[..., :2] / j[..., 2:3].clamp(min=1e-3) + center


def hand_points_2d(mhr, a: dict, device: str, center: np.ndarray, batch: int = 2048) -> np.ndarray:
    """(n, 2, 21, 2) projected hand points of an archive."""
    out = []
    idx = HAND_JOINTS["l"] + HAND_JOINTS["r"]
    with torch.inference_mode():
        for s in range(0, len(a["frames"]), batch):
            p = torch.from_numpy(a["mhr_model_params"][s:s + batch]).float().to(device)
            uv = project(skeleton(mhr, p)[:, idx], torch.from_numpy(a["cam_t"][s:s + batch]).float().to(device),
                         torch.from_numpy(a["focal"][s:s + batch]).float().to(device),
                         torch.tensor(center, device=device, dtype=torch.float32))
            out.append(uv.cpu().numpy().reshape(-1, 2, 21, 2))
    return np.concatenate(out)


def fit_chunk(mhr, params, cam_t, focal, center, target, weight, size, link, free, prior_w, args):
    """Changes of the free angles for one chunk of consecutive samples. link[i]: sample i follows i-1."""
    dev = params.device
    delta = torch.zeros(len(params), len(free), device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    idx = HAND_JOINTS["l"] + HAND_JOINTS["r"]
    accel_ok = (link[1:-1] & link[2:]).float()  # samples i-1, i, i+1 all consecutive
    for _ in range(args.iterations):
        p = params.clone()
        p[:, free] = p[:, free] + delta
        uv = project(skeleton(mhr, p)[:, idx], cam_t, focal, center).reshape(-1, 2, 21, 2)
        r2 = ((uv - target) ** 2).sum(-1) / size[..., None] ** 2
        data = (weight * r2 / (r2 + GM_SCALE ** 2)).sum() / len(params)
        prior = (prior_w * delta ** 2).sum() / len(params)
        acc = delta[2:] - 2 * delta[1:-1] + delta[:-2]
        temporal = ((acc ** 2).sum(1) * accel_ok).sum() / len(params)
        loss = data + args.prior * prior + args.temporal * temporal
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return delta.detach()


def errors(points: np.ndarray, target: np.ndarray, weight: np.ndarray, size: np.ndarray) -> dict:
    """Median distance in hand sizes and PCK at 0.2 / 0.1 hand size, over the confident reference points."""
    e = np.linalg.norm(points - target, axis=-1) / size[..., None]
    e = e[weight > 0]
    return {"median": round(float(np.median(e)), 4), "pck@0.2": round(float((e < 0.2).mean()), 4),
            "pck@0.1": round(float((e < 0.1).mean()), 4), "points": int(len(e))}


def jitter(points: np.ndarray, frames: np.ndarray, size: np.ndarray) -> float:
    """p95 acceleration of the projected hand points, in hand sizes, over consecutive frames."""
    ok = (np.diff(frames)[1:] == 1) & (np.diff(frames)[:-1] == 1)
    acc = np.linalg.norm(points[2:] - 2 * points[1:-1] + points[:-2], axis=-1) / size[1:-1, :, None]
    return round(float(np.percentile(acc[ok], 95)), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with assets/mhr_model.pt")
    parser.add_argument("--params", default="sam3d_final", help="smoothed person archives to start from")
    parser.add_argument("--reference", default="hand_reference", help="output of hand_reference.py")
    parser.add_argument("--source", choices=("hand", "rtmw", "hand_only"), default="rtmw")
    parser.add_argument("--output", help="output folder (omit with --evaluate to only measure --params)")
    parser.add_argument("--evaluate", nargs="*", default=None, help="also measure these archive folders")
    parser.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--parts", nargs="+", default=["arms", "wrists", "fingers"])
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--people", type=Path, help="people.json (default: <pipeline>/people.json), for the masks")
    parser.add_argument("--no_view_filter", action="store_true",
                        help="keep 2D points out of the picture or off the person's mask (old behaviour)")
    parser.add_argument("--prior", type=float, default=0.05)
    parser.add_argument("--temporal", type=float, default=5e4,
                        help="weight of the acceleration of the change (2e4: closer to the detectors, 1e5: as still as the smoother)")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--cut", type=float, default=18)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.pipeline
    model = args.checkpoint / "assets" / "mhr_model.pt"
    for path in (d / "input.mp4", d / args.params, d / args.reference, model):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    if args.output and (d / args.output).exists() and any((d / args.output).iterdir()):
        sys.exit(f"error: {d / args.output} is not empty")
    people_path = args.people or d / "people.json"
    if not args.no_view_filter and not people_path.is_file():
        sys.exit(f"error: missing {people_path} (or pass --no_view_filter)")
    people = json.loads(people_path.read_text()) if not args.no_view_filter else None
    import cv2

    capture = cv2.VideoCapture(str(d / "input.mp4"))
    center = np.array([capture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2])
    capture.release()
    labels = cut_labels(d / "input.mp4", args.cut)
    mhr = torch.jit.load(str(model), map_location=args.device)
    parts = parameter_parts(mhr, args.device)
    free = [j for j, (part, side) in enumerate(parts) if part in args.parts and side is not None]
    prior_w = torch.tensor([PRIOR[parts[j][0]] for j in free], device=args.device)
    print(f"{len(free)} free angles: " + ", ".join(f"{p} {sum(parts[j][0] == p for j in free)}" for p in args.parts))

    report = {"args": {k: str(v) for k, v in vars(args).items()}, "people": {}}
    out_dir = d / args.output if args.output else None
    if out_dir:
        out_dir.mkdir(exist_ok=True)
    start = time.time()
    for path in sorted((d / args.params).glob("*.npz")):
        person = path.stem
        with np.load(path) as npz:
            a = {k: npz[k] for k in npz.files}
        ref_path = d / args.reference / f"{person}.npz"
        if not ref_path.is_file():
            sys.exit(f"error: missing {ref_path}")
        with np.load(ref_path) as npz:
            ref = {k: npz[k] for k in npz.files}
        rows = np.searchsorted(a["frames"], ref["frames"])
        if (rows >= len(a["frames"])).any() or (a["frames"][np.minimum(rows, len(a["frames"]) - 1)] != ref["frames"]).any():
            sys.exit(f"error: person {person}: reference frames are not frames of {args.params}")
        if args.range:
            keep = (ref["frames"] >= args.range[0]) & (ref["frames"] <= args.range[1])
            ref = {k: v[keep] for k, v in ref.items()}
            rows = rows[keep]
        frames = a["frames"][rows].astype(int)
        size = np.maximum(ref["hand_box"] / 1.25, 20).astype(np.float32)  # hand crop side without rtmlib's margin
        view = None
        if not args.no_view_filter:
            # Detectors place hands out of the frame on its border, with confidence: such points are not data.
            n = len(frames)
            rtmw_pts = np.stack([ref["body"][:, RTMW_HANDS[s]] for s in "lr"], 1).reshape(n, 42, 2)
            both = keypoint_view.in_view(d, people, person, frames, np.concatenate([rtmw_pts, ref["hands"].reshape(n, 42, 2)], 1),
                                         cover=(ref["body"][:, :23], ref["body_score"][:, :23]))
            view = {"rtmw": both[:, :42].reshape(n, 2, 21), "hand": both[:, 42:].reshape(n, 2, 21)}
            print(f"person {person}: {100 * (1 - view['rtmw'].mean()):.1f}% of RTMW hand points out of view", flush=True)
        targets = {s: reference(ref, s, args.min_score, view) for s in ("hand", "rtmw", "hand_only")}
        target, weight = targets[args.source]
        link = np.r_[False, (np.diff(frames) == 1) & (labels[frames[1:]] == labels[frames[:-1]])]

        params = a["mhr_model_params"][rows].astype(np.float32).copy()
        if out_dir:
            dev = args.device
            new = params.copy()
            # Chunks of consecutive samples, cut at shot changes and gaps; long runs overlap by 32 samples and the
            # middle of each chunk is kept, so a chunk border sees data on both sides.
            breaks = np.r_[np.where(~link)[0], len(frames)]
            for b0, b1 in zip(breaks[:-1], breaks[1:]):
                s = b0
                while s < b1:
                    lo, hi = max(b0, s - 32), min(b1, s + args.chunk + 32)
                    t = lambda x: torch.from_numpy(np.ascontiguousarray(x[lo:hi])).float().to(dev)  # noqa: E731
                    delta = fit_chunk(mhr, t(params), t(a["cam_t"][rows]), t(a["focal"][rows]),
                                      torch.tensor(center, device=dev, dtype=torch.float32), t(target), t(weight),
                                      t(size), torch.from_numpy(link[lo:hi]).to(dev), free, prior_w, args)
                    keep_to = min(b1, s + args.chunk)
                    new[s:keep_to, free] += delta[s - lo:keep_to - lo].cpu().numpy()
                    s = keep_to
            a_out = dict(a)
            a_out["mhr_model_params"] = a["mhr_model_params"].copy()
            a_out["mhr_model_params"][rows] = new
            np.savez_compressed(out_dir / f"{person}.npz", **a_out)
            print(f"person {person}: {len(rows)} frames fitted, {time.time() - start:.0f} s", flush=True)

        # Measures: projected hand points of each archive against each reference.
        folders = [args.params] + ([args.output] if args.output else []) + list(args.evaluate or [])
        report["people"][person] = {}
        for folder in folders:
            with np.load(d / folder / f"{person}.npz") as npz:
                b = {k: npz[k] for k in ("frames", "mhr_model_params", "cam_t", "focal")}
            r = np.searchsorted(b["frames"], frames)
            b = {k: v[r] for k, v in b.items()}
            uv = hand_points_2d(mhr, b, args.device, center)
            report["people"][person][folder] = {
                **{f"vs_{s}": errors(uv, *targets[s][:2], size) for s in ("hand_only", "rtmw")},
                "jitter_p95": jitter(uv, frames, size)}
            print(person, folder, json.dumps(report["people"][person][folder]), flush=True)

    name = f"hand_refit_{args.output or args.params}.json"
    (d / name).write_text(json.dumps(report, indent=1))
    print(f"wrote {d / name}")
    print("REFIT_OK")


if __name__ == "__main__":
    main()
