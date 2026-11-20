"""Avatar body fitted to the 2D body points of RTMW, over the smoothed SAM 3D Body parameters.

The robust smoother removes SAM 3D Body's spikes, but on fast moves (a dance, a quick gesture) it trails behind the
image, and nothing brings the body back onto it: on the dance, the error to the 2D body grows from 0.86 to 0.94 torso
on the fastest frames. This script moves the pose angles (root rotation, spine and head, arms, wrists, legs) so that
the 17 projected COCO points of the avatar land on RTMW, with:

- a **data term**: robust (Geman-McClure) distance in torso sizes, weighted by the confidence; points out of the
  picture or off the person's SAM3 mask do not count (``keypoint_view.py``);
- a **prior**: the change from the smoothed angles costs (more for the root than for the limbs);
- a **temporal term**: the acceleration of the change costs, **less where the 2D points move fast**, so a fast move
  is followed and a still pose stays still (weight ``1 / (1 + (speed / --fast_speed)^2)``, speed in torsos per
  second, from the 2D points).

The points are SAM 3D Body's own MHR70 keypoints, the ones the annotations are compared to. Shoulders, wrists, hips
and knees are skeleton joints; the face, elbows and ankles are mixes of mesh vertices. Each of these is attached to
a joint (head, ``lowarm``, ``foot``) by its offset in the joint's frame, taken per frame on the starting pose: the
fit then runs on the skeleton without skinning, as ``hand_refit.py`` (0.3 cm off after a 55 cm move of the elbow).
Fitting bare joints instead, with the head free and no face target, let the head drift: OKS 0.816 -> 0.747 on the
talk-show annotations although closer to RTMW. Run ``hand_refit.py`` afterwards on the output for the fingers.

``--holdout K`` leaves the targets of every K-th frame out of the fit and measures the error there: an honest
measure of how well the fitted motion follows the image between the frames it was fitted on.

    python research/sam3d_body/body_refit.py .e2e_outputs/v4_dance --checkpoint \\
        .e2e_outputs/ext/ckpt/sam-3d-body-vith --params sam3d_robust --output sam3d_body_refit
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
import keypoint_view  # noqa: E402
from hand_refit import project  # noqa: E402
from robust_smooth import parameter_parts  # noqa: E402
from run_keyframes import cut_labels  # noqa: E402

# COCO 0-16 -> MHR70 keypoint and the skeleton joint it is attached to.
COCO = list(range(17))
MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
JOINTS = ["c_head"] * 5 + ["l_uparm", "r_uparm", "l_lowarm", "r_lowarm", "l_wrist", "r_wrist",
                           "l_upleg", "r_upleg", "l_lowleg", "r_lowleg", "l_foot", "r_foot"]
PRIOR = {"root": 8.0, "torso_head": 4.0, "arms": 1.0, "wrists": 1.0, "legs": 1.0}  # cost of a change of 1 rad
GM_SCALE = 0.1  # Geman-McClure scale, in torso sizes


def skeleton_state(mhr, params: torch.Tensor) -> torch.Tensor:
    """(n, 127, 8) joint states: position (cm, MHR axes), rotation quaternion (x, y, z, w), scale."""
    c = mhr.character_torch
    joint_params = c.model_parameters_to_joint_parameters(
        torch.cat([params, torch.zeros(len(params), 45, device=params.device)], 1))
    return c.joint_parameters_to_skeleton_state(joint_params)


def rotate(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Rotates x by the unit quaternions q (x, y, z, w)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2 * torch.cross(u, x, dim=-1)
    return x + w * t + torch.cross(u, t, dim=-1)


def attached(state: torch.Tensor, joints: list[int], local: torch.Tensor) -> torch.Tensor:
    """(n, k, 3) keypoints from their joints' states and their local offsets."""
    s = state[:, joints]
    return s[..., :3] + s[..., 7:8] * rotate(s[..., 3:7], local)


@torch.inference_mode()
def keypoints_cm(mhr, mapping: torch.Tensor, a: dict, rows: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, 17, 3) MHR70 keypoints (cm, MHR axes) from the full mesh, and (n, 127, 8) joint states."""
    kps, states = [], []
    for s in range(0, len(rows), 256):
        r = rows[s:s + 256]
        v, _ = mhr(*(torch.from_numpy(a[k][r]).float().to(device) for k in ("shape_params", "mhr_model_params", "expr_params")))
        state = skeleton_state(mhr, torch.from_numpy(a["mhr_model_params"][r]).float().to(device))
        kps.append(torch.einsum("kv,bvc->bkc", mapping, torch.cat([v, state[..., :3]], 1)))
        states.append(state)
    return torch.cat(kps), torch.cat(states)


def torso_size(points: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """(n,) shoulder-hip distance from the 2D points, filled from neighbours where the torso is not seen."""
    sh, hip = points[:, [0, 1]].mean(1), points[:, [6, 7]].mean(1)
    ok = (weight[:, [0, 1, 6, 7]] > 0).all(1)
    size = np.linalg.norm(sh - hip, axis=-1)
    if not ok.any():
        return np.full(len(points), 100.0, np.float32)
    idx = np.arange(len(points))
    return np.maximum(np.interp(idx, idx[ok], size[ok]), 20).astype(np.float32)


def moved_camera(cam_t: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    """cam_t changed by cam = (log depth factor, image shift x, y in focal units): scaling the translation moves the
    person along the viewing ray (same place in the image, size divided by the factor), then the shift moves it
    across the image at its new depth."""
    t = cam_t * torch.exp(cam[:, :1])
    return torch.cat([t[:, :2] + cam[:, 1:] * t[:, 2:], t[:, 2:]], 1)


def fit_chunk(mhr, params, cam_t, focal, center, target, weight, size, link, tweight, free, prior_w, joints, local, args):
    """Changes of the free angles (and of the camera, with --depth) for one chunk of consecutive samples.
    link[i]: sample i follows i-1."""
    delta = torch.zeros(len(params), len(free), device=params.device, requires_grad=True)
    cam = torch.zeros(len(params), 3, device=params.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta, cam] if args.depth else [delta], lr=args.lr)
    accel_ok = (link[1:-1] & link[2:]).float() * tweight[1:-1]
    for _ in range(args.iterations):
        p = params.clone()
        p[:, free] = p[:, free] + delta
        uv = project(attached(skeleton_state(mhr, p), joints, local), moved_camera(cam_t, cam), focal, center)
        r2 = ((uv - target) ** 2).sum(-1) / size[:, None] ** 2
        data = (weight * r2 / (r2 + GM_SCALE ** 2)).sum() / len(params)
        prior = (prior_w * delta ** 2).sum() / len(params) + args.depth_prior * (cam ** 2).sum() / len(params)
        acc = delta[2:] - 2 * delta[1:-1] + delta[:-2]
        cam_acc = cam[2:] - 2 * cam[1:-1] + cam[:-2]
        temporal = (((acc ** 2).sum(1) + args.depth_temporal * (cam_acc ** 2).sum(1)) * accel_ok).sum() / len(params)
        loss = data + args.prior * prior + args.temporal * temporal
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return delta.detach(), cam.detach()


def points_2d(mhr, mapping, a: dict, rows: np.ndarray, center: np.ndarray, device: str) -> np.ndarray:
    """(n, 17, 2) projected MHR70 keypoints from the full mesh (the honest measure, not the attached offsets)."""
    k, _ = keypoints_cm(mhr, mapping, a, rows, device)
    t = lambda key: torch.from_numpy(a[key][rows]).float().to(device)  # noqa: E731
    with torch.inference_mode():
        return project(k, t("cam_t"), t("focal"), torch.tensor(center, device=device, dtype=torch.float32)).cpu().numpy()


def measures(uv, target, weight, size, frames, fast) -> dict:
    e = np.linalg.norm(uv - target, axis=-1) / size[:, None]
    ok = weight > 0
    out = {"median": round(float(np.median(e[ok])), 4), "pck@0.1": round(float((e[ok] < 0.1).mean()), 4),
           "median_fast": round(float(np.median(e[ok & fast[:, None]])), 4) if (ok & fast[:, None]).any() else None,
           "points": int(ok.sum())}
    cons = (np.diff(frames)[1:] == 1) & (np.diff(frames)[:-1] == 1)
    acc = np.linalg.norm(uv[2:] - 2 * uv[1:-1] + uv[:-2], axis=-1) / size[1:-1, None]
    out["accel_p95"] = round(float(np.percentile(acc[cons], 95)), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with model.ckpt and assets/mhr_model.pt")
    parser.add_argument("--params", default="sam3d_robust", help="smoothed person archives to start from")
    parser.add_argument("--reference", default="hand_reference", help="output of hand_reference.py (RTMW body)")
    parser.add_argument("--people", type=Path, help="people.json (default: <pipeline>/people.json), for the masks")
    parser.add_argument("--output", help="output folder (omit to only measure)")
    parser.add_argument("--evaluate", nargs="*", default=[], help="also measure these archive folders")
    parser.add_argument("--parts", nargs="+", default=["root", "torso_head", "arms", "wrists", "legs"])
    parser.add_argument("--min_score", type=float, default=0.5)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--temporal", type=float, default=1e5,
                        help="weight of the acceleration (per frame) of the change. 1e5 keeps the robust smoother's "
                             "jitter at 25 frames/s. Scaling it by (fps / 25)^4 for the same smoothness in seconds "
                             "made the fit too stiff to converge at 60 frames/s (OKS 0.70): the detector's noise is "
                             "per frame, so the weight is kept per frame")
    parser.add_argument("--fast_speed", type=float, default=1.0,
                        help="2D speed (torsos per second) at which the temporal weight is halved")
    parser.add_argument("--depth", action="store_true",
                        help="also refit the depth and image position of the person (off by default: on the talk-show "
                             "the avatar's head size already matches the image within 0.87-1.07 in every shot, the "
                             "x2.35 close-up error measured before came from detector points invented out of the "
                             "frame; refitting the depth kept OKS (0.843 vs 0.841) and tripled the jitter)")
    parser.add_argument("--depth_prior", type=float, default=1.0, help="cost of a log depth change of 1")
    parser.add_argument("--depth_temporal", type=float, default=10.0,
                        help="extra weight of the camera's acceleration (depth jumps were the worst spikes)")
    parser.add_argument("--holdout", type=int, default=0, help="leave every K-th frame out of the fit, measure there")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--cut", type=float, default=18)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.pipeline
    model = args.checkpoint / "assets" / "mhr_model.pt"
    people_path = args.people or d / "people.json"
    for path in (d / "input.mp4", d / args.params, d / args.reference, model, people_path):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    if args.output and (d / args.output).exists() and any((d / args.output).iterdir()):
        sys.exit(f"error: {d / args.output} is not empty")
    people = json.loads(people_path.read_text())
    import cv2

    capture = cv2.VideoCapture(str(d / "input.mp4"))
    center = np.array([capture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2])
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not fps:
        sys.exit("error: no frame rate in input.mp4")
    labels = cut_labels(d / "input.mp4", args.cut)
    mhr = torch.jit.load(str(model), map_location=args.device)
    state = torch.load(args.checkpoint / "model.ckpt", map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    if "head_pose.keypoint_mapping" not in state:
        sys.exit("error: head_pose.keypoint_mapping not in the checkpoint")
    mapping = state["head_pose.keypoint_mapping"][MHR70].float().to(args.device)
    del state
    names = list(mhr.character_torch.skeleton.joint_names)
    joints = [names.index(n) for n in JOINTS]
    parts = parameter_parts(mhr, args.device)
    free = [j for j, (part, _) in enumerate(parts) if part in args.parts]
    prior_w = torch.tensor([PRIOR[parts[j][0]] for j in free], device=args.device)
    print(f"{len(free)} free values: " + ", ".join(f"{p} {sum(parts[j][0] == p for j in free)}" for p in args.parts))

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
            ref = {k: npz[k] for k in ("frames", "body", "body_score")}
        rows = np.searchsorted(a["frames"], ref["frames"])
        if (rows >= len(a["frames"])).any() or (a["frames"][np.minimum(rows, len(a["frames"]) - 1)] != ref["frames"]).any():
            sys.exit(f"error: person {person}: reference frames are not frames of {args.params}")
        frames = a["frames"][rows].astype(int)
        target = np.nan_to_num(ref["body"][:, COCO]).astype(np.float32)
        view = keypoint_view.in_view(d, people, person, frames, ref["body"][:, COCO])
        weight = np.where((ref["body_score"][:, COCO] > args.min_score) & view, ref["body_score"][:, COCO], 0)
        weight = weight.astype(np.float32)
        size = torso_size(target, weight)
        link = np.r_[False, (np.diff(frames) == 1) & (labels[frames[1:]] == labels[frames[:-1]])]
        # 2D speed of the seen points, torsos per second: where the image moves fast, the fit may move fast.
        step = np.linalg.norm(np.diff(target, axis=0), axis=-1) / size[1:, None]
        both = (weight[1:] > 0) & (weight[:-1] > 0)
        speed = np.r_[0, np.where(both.any(1), np.nanmedian(np.where(both, step, np.nan), axis=1), 0)] * fps
        speed = np.where(link, np.nan_to_num(speed), 0)
        tweight = (1 / (1 + (speed / args.fast_speed) ** 2)).astype(np.float32)
        fast = speed >= np.percentile(speed[link], 90) if link.any() else np.zeros(len(frames), bool)
        held = np.zeros(len(frames), bool)
        if args.holdout:
            held[::args.holdout] = True
        fit_weight = np.where(held[:, None], 0, weight).astype(np.float32)
        print(f"person {person}: {len(frames)} frames, {100 * (weight > 0).mean():.1f}% of the points used, "
              f"{100 * (1 - view.mean()):.1f}% out of view", flush=True)

        if out_dir:
            dev = args.device
            params = a["mhr_model_params"][rows].astype(np.float32).copy()
            # Offsets of the keypoints in their joints' frames, on the starting pose.
            k0, s0 = keypoints_cm(mhr, mapping, a, rows, dev)
            sj = s0[:, joints]
            q_inv = torch.cat([-sj[..., 3:6], sj[..., 6:7]], -1)
            local_all = rotate(q_inv, k0 - sj[..., :3]) / sj[..., 7:8]
            del k0, s0, sj, q_inv
            new = params.copy()
            new_cam = a["cam_t"][rows].astype(np.float32).copy()
            breaks = np.r_[np.where(~link)[0], len(frames)]
            for b0, b1 in zip(breaks[:-1], breaks[1:]):
                s = b0
                while s < b1:
                    lo, hi = max(b0, s - 32), min(b1, s + args.chunk + 32)
                    t = lambda x: torch.from_numpy(np.ascontiguousarray(x[lo:hi])).float().to(dev)  # noqa: E731
                    delta, cam = fit_chunk(mhr, t(params), t(a["cam_t"][rows]), t(a["focal"][rows]),
                                      torch.tensor(center, device=dev, dtype=torch.float32), t(target),
                                      t(fit_weight), t(size), torch.from_numpy(link[lo:hi]).to(dev), t(tweight),
                                      free, prior_w, joints, local_all[lo:hi], args)
                    keep_to = min(b1, s + args.chunk)
                    new[s:keep_to, free] += delta[s - lo:keep_to - lo].cpu().numpy()
                    new_cam[s:keep_to] = moved_camera(t(a["cam_t"][rows]), cam)[s - lo:keep_to - lo].cpu().numpy()
                    s = keep_to
            a_out = dict(a)
            a_out["mhr_model_params"] = a["mhr_model_params"].copy()
            a_out["mhr_model_params"][rows] = new
            a_out["cam_t"] = a["cam_t"].copy()
            a_out["cam_t"][rows] = new_cam
            np.savez_compressed(out_dir / f"{person}.npz", **a_out)
            print(f"person {person}: fitted, {time.time() - start:.0f} s", flush=True)

        report["people"][person] = {}
        for folder in [args.params] + ([args.output] if args.output else []) + args.evaluate:
            with np.load(d / folder / f"{person}.npz") as npz:
                b = {k: npz[k] for k in ("frames", "mhr_model_params", "cam_t", "focal")}
            r = np.searchsorted(b["frames"], frames)
            with np.load(d / folder / f"{person}.npz") as npz:
                b["shape_params"], b["expr_params"] = npz["shape_params"], npz["expr_params"]
            uv = points_2d(mhr, mapping, b, r, center, args.device)
            entry = {"all": measures(uv, target, weight, size, frames, fast)}
            if args.holdout:
                entry["held_out"] = measures(uv, target, np.where(held[:, None], weight, 0), size, frames, fast)
            report["people"][person][folder] = entry
            print(person, folder, json.dumps(entry), flush=True)

    name = f"body_refit_{args.output or args.params}.json"
    (d / name).write_text(json.dumps(report, indent=1))
    print(f"wrote {d / name}")
    print("BODY_REFIT_OK")


if __name__ == "__main__":
    main()
