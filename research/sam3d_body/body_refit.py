"""Avatar body fitted to the 2D body points of RTMW, over the smoothed SAM 3D Body parameters.

The robust smoother removes SAM 3D Body's spikes, but on fast moves (a dance, a quick gesture) it trails behind the
image, and nothing brings the body back onto it: on the dance, the error to the 2D body grows from 0.86 to 0.94 torso
on the fastest frames. This script moves the pose angles (root rotation, spine and head, arms, wrists, legs) so that
the 17 projected COCO points and the 6 feet points (heels, big and small toes) of the avatar land on RTMW, with:

- a **data term**: robust (Geman-McClure) distance in torso sizes, weighted by the confidence; points out of the
  picture or off the person's SAM3 mask do not count (``keypoint_view.py``);
- a **prior**: the change from the smoothed angles costs (more for the root than for the limbs);
- a **temporal term**: the acceleration of the change costs, **less where the 2D points move fast**, so a fast move
  is followed and a still pose stays still (weight ``1 / (1 + (speed / --fast_speed)^2)``, speed in torsos per
  second, from the 2D points);
- a **silhouette term**, on the frames where the person's mask touches the picture's border (entering or leaving
  the frame): the 23 points and the middles of the limbs must project on a SAM3 mask (anyone's: a limb may pass
  behind another dancer) or outside the picture; the cost is the squared distance to such a place, in torsos. With
  only the 2D points, the parts out of the frame were free: V5's man leaving the frame folded an arm and the legs
  into the empty room, the girl pushed out lay down on the floor. On these frames the avatar may also move across
  the image (``--edge_shift``), so it can leave the frame. Not used elsewhere: a part on no mask may be hidden by a
  thing (legs under a table), not absent.

The points are SAM 3D Body's own MHR70 keypoints, the ones the annotations are compared to. Shoulders, wrists, hips
and knees are skeleton joints; the face, elbows, ankles and feet are mixes of mesh vertices. Each point is attached
to its nearest joint (median over the person's frames) by its offset in the joint's frame, taken per frame on the
starting pose: the fit then runs on the skeleton without skinning, as ``hand_refit.py`` (0.3 cm off after a 55 cm
move of the elbow). The feet points (heel, toes) tell where the foot points, hence a foot behind the other.
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
from run_keyframes import cut_labels, repeated_frames  # noqa: E402

# COCO-WholeBody 0-22 (body, then left big toe, small toe, heel, right big toe, small toe, heel) -> MHR70 keypoint.
COCO = list(range(23))
MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
SHOULDERS, HIPS = [COCO.index(5), COCO.index(6)], [COCO.index(11), COCO.index(12)]
PRIOR = {"root": 8.0, "torso_head": 4.0, "arms": 1.0, "wrists": 1.0, "legs": 1.0}  # cost of a change of 1 rad
# COCO points that tell how fast each limb moves (knees, ankles, feet; elbows, wrists).
LEG_POINTS = [COCO.index(k) for k in (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)]
ARM_POINTS = [COCO.index(k) for k in (7, 8, 9, 10)]
GM_SCALE = 0.1  # Geman-McClure scale, in torso sizes
# Limb segments (COCO indices) whose middle is also checked by the silhouette term, besides the 23 points.
SEGMENTS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12), (5, 12), (6, 11)]


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
    sh, hip = points[:, SHOULDERS].mean(1), points[:, HIPS].mean(1)
    ok = (weight[:, SHOULDERS + HIPS] > 0).all(1)
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


def steadiness(target: np.ndarray, frames: np.ndarray, repeated: np.ndarray, labels: np.ndarray,
               level: float) -> np.ndarray:
    """(n, k) factor in (0, 1] per 2D point: 1 / (1 + (a / level)^2), a = median over 7 new frames of the point's own
    2D acceleration (torso sizes per frame^2, from the shoulders-hips distance of the target).

    A detector's guess of a hidden part (legs under a table) jumps from frame to frame (RTMW's ankles on the
    talk-show: 2 torsos per frame^2), a real fast move is smooth (V5's stamp: about 0.12). With the robust loss
    annealed, guesses that fell on the mask were followed; this factor lets the real moves through and not them."""
    new = np.where(~repeated[frames])[0]
    out = np.ones(target.shape[:2])
    if len(new) < 7:
        return out
    t = target[new]
    torso = np.linalg.norm(t[:, [5, 6]].mean(1) - t[:, [11, 12]].mean(1), axis=-1)
    torso = np.maximum(np.median(torso), 1.0)
    f, lab = frames[new], labels[frames[new]]
    acc = np.full(t.shape[:2], np.nan)
    ok = (lab[2:] == lab[1:-1]) & (lab[1:-1] == lab[:-2]) & (np.diff(f)[1:] <= 3) & (np.diff(f)[:-1] <= 3)
    a = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=-1) / torso
    acc[1:-1][ok] = a[ok]
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(acc, ((3, 3), (0, 0)), constant_values=np.nan), 7, axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(windows, axis=-1)
    factor = 1 / (1 + (np.nan_to_num(med, nan=0.0) / level) ** 2)
    # Repeated frames take the factor of their last new frame (their own weight is 0 anyway).
    last_new = np.maximum.accumulate(np.where(~repeated[frames], np.arange(len(frames)), 0))
    full = np.ones(target.shape[:2])
    full[new] = factor
    return full[last_new]


def gm_scale(it: int, args) -> float:
    """Geman-McClure scale at iteration it (graduated non-convexity): from --gm_start torsos, divided by 1.4 per
    step, down to GM_SCALE by 60 % of the iterations, then held. A fixed 0.1 torso gives a point 0.4 torso away 4 %
    of the pull of a near one: a stamp or a sharp arm move that the smoother rounded off was taken for an outlier
    and never brought back."""
    if args.gm_start <= GM_SCALE:
        return GM_SCALE
    steps = int(np.ceil(np.log(args.gm_start / GM_SCALE) / np.log(1.4)))
    k = min(steps, int(it / max(1, 0.6 * args.iterations) * (steps + 1)))
    return max(GM_SCALE, args.gm_start / 1.4 ** k)


def silhouette_points(uv: torch.Tensor) -> torch.Tensor:
    """(n, 23 + segments, 2): the 23 projected points and the middles of the limb segments."""
    a, b = zip(*SEGMENTS)
    return torch.cat([uv, 0.5 * (uv[:, list(a)] + uv[:, list(b)])], 1)


def off_silhouette(uv: torch.Tensor, dist: torch.Tensor, scale: int) -> torch.Tensor:
    """(n, k) distance in full-size pixels from each point to an allowed place (dist: (n, h, w) at 1/scale; outside
    the map counts as allowed, bilinear so the gradient flows)."""
    h, w = dist.shape[1:]
    grid = torch.stack([2 * uv[..., 0] / scale / w - 1, 2 * uv[..., 1] / scale / h - 1], -1)[:, None]
    return torch.nn.functional.grid_sample(dist[:, None], grid, align_corners=False, padding_mode="zeros")[:, 0, 0] * scale


def fit_chunk(mhr, params, cam_t, focal, center, target, weight, size, link, tweight, free, prior_w, temporal_w, joints,
              local, args, dist=None, edge=None):
    """Changes of the free angles (and of the camera, with --depth, or on edge frames with --edge_shift) for one chunk
    of consecutive samples. link[i]: sample i follows i-1. dist, edge: allowed_distance() maps (float, 1/scale) and
    the frames where the silhouette term is used."""
    delta = torch.zeros(len(params), len(free), device=params.device, requires_grad=True)
    cam = torch.zeros(len(params), 3, device=params.device, requires_grad=True)
    use_sil = dist is not None and bool(edge.any())
    # Camera prior per frame and value: free (depth_prior) with --depth; with --edge_shift, the image shift is free
    # on edge frames only; everything else is held by a large cost.
    cam_prior = torch.full((len(params), 3), 1e4, device=params.device)
    if args.depth:
        cam_prior[:] = args.depth_prior
    elif use_sil and args.edge_shift:
        cam_prior[edge, 1:] = args.depth_prior
    move_cam = args.depth or (use_sil and args.edge_shift)
    optimizer = torch.optim.Adam([delta, cam] if move_cam else [delta], lr=args.lr)
    accel_ok = (link[1:-1] & link[2:]).float()[:, None] * tweight[1:-1]  # (n-2, free values)
    for it in range(args.iterations):
        gm = gm_scale(it, args)
        # Cosine decay of the step to 1 %: at a constant step Adam hovers around the optimum, frame by frame
        # differently (V1: the avatar's face shook 5x more than RTMW's at 600 iterations, 1.4x at 300).
        if args.lr_decay:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * it / args.iterations)))
        p = params.clone()
        p[:, free] = p[:, free] + delta
        uv = project(attached(skeleton_state(mhr, p), joints, local), moved_camera(cam_t, cam), focal, center)
        r2 = ((uv - target) ** 2).sum(-1) / size[:, None] ** 2
        # sigma^2 r^2 / (sigma^2 + r^2), divided by GM_SCALE^2: quadratic near 0 whatever sigma (SMPLify's form), and
        # the former r^2 / (r^2 + GM_SCALE^2) at the end of the annealing.
        data = (weight * gm ** 2 * r2 / (r2 + gm ** 2)).sum() / len(params) / GM_SCALE ** 2
        prior = (prior_w * delta ** 2).sum() / len(params) + (cam_prior * cam ** 2).sum() / len(params)
        acc = delta[2:] - 2 * delta[1:-1] + delta[:-2]
        cam_acc = cam[2:] - 2 * cam[1:-1] + cam[:-2]
        temporal = ((temporal_w * acc ** 2 * accel_ok).sum() + (args.temporal * args.depth_temporal
                    * (cam_acc ** 2).sum(1) * accel_ok.mean(1)).sum()) / len(params)
        loss = data + args.prior * prior + temporal
        if use_sil:
            off = off_silhouette(silhouette_points(uv), dist, args.sil_scale) / size[:, None]
            loss = loss + args.silhouette * (edge[:, None] * off ** 2).sum() / len(params)
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
    parser.add_argument("--min_score", type=float, default=1.5, help="RTMW score under which a point is not used")
    parser.add_argument("--full_score", type=float, default=3.0, help="RTMW score from which a point counts fully")
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--temporal", type=float, default=1e5,
                        help="weight of the acceleration (per frame) of the change. 1e5 keeps the robust smoother's "
                             "jitter at 25 frames/s. Scaling it by (fps / 25)^4 for the same smoothness in seconds "
                             "made the fit too stiff to converge at 60 frames/s (OKS 0.70): the detector's noise is "
                             "per frame, so the weight is kept per frame")
    parser.add_argument("--legs_temporal", type=float, default=5e3,
                        help="temporal weight of the leg values (dance: 1e5 kept 0.87 of the feet amplitude, 5e3 0.92, "
                             "for the same jitter of the feet)")
    parser.add_argument("--arms_temporal", type=float, default=5e3,
                        help="temporal weight of the arm and wrist values (V5's arm crosses: 1e5 kept 0.47-0.59 of the "
                             "wrists' peak speed, 5e3 0.73-0.95)")
    parser.add_argument("--jerk_level", type=float, default=0.3,
                        help="2D acceleration (torsos per frame^2, local median) at which a point's weight is halved; "
                             "0: off")
    parser.add_argument("--frames", type=int, nargs=2, help="first and last frame to fit (a test on an extract)")
    parser.add_argument("--repeat_level", type=float, default=0.15,
                        help="mean gray difference under which a frame repeats the previous one (0: off)")
    parser.add_argument("--no_lr_decay", dest="lr_decay", action="store_false",
                        help="constant learning rate (before: Adam hovered around the optimum)")
    parser.add_argument("--part_speed", choices=("max", "median", "off"), default="max",
                        help="speed that relaxes each limb's temporal term: its fastest or median point, or off (the "
                             "body median for every value, as before)")
    parser.add_argument("--gm_start", type=float, default=1.0,
                        help="Geman-McClure scale (torsos) at the start, annealed to 0.1 (graduated non-convexity); "
                             "0.1 or less: fixed scale, as before")
    parser.add_argument("--fast_speed", type=float, default=1.0,
                        help="2D speed (torsos per second) at which the temporal weight is halved")
    parser.add_argument("--depth", action="store_true",
                        help="also refit the depth and image position of the person (off by default: on the talk-show "
                             "the avatar's head size already matches the image within 0.87-1.07 in every shot, the "
                             "x2.35 close-up error measured before came from detector points invented out of the "
                             "frame; refitting the depth kept OKS (0.843 vs 0.841) and tripled the jitter)")
    parser.add_argument("--depth_prior", type=float, default=1.0, help="cost of a log depth change of 1")
    parser.add_argument("--silhouette", type=float, default=100.0,
                        help="weight of the silhouette term on frames where the person touches the picture's border "
                             "(squared distance in torsos to a mask or the outside, per point); 0: off")
    parser.add_argument("--sil_scale", type=int, default=4, help="the masks are read at 1/sil_scale")
    parser.add_argument("--no_edge_shift", dest="edge_shift", action="store_false",
                        help="keep the avatar's image position on edge frames (by default it may move, to leave)")
    parser.add_argument("--depth_temporal", type=float, default=10.0,
                        help="extra weight of the camera's acceleration (depth jumps were the worst spikes)")
    parser.add_argument("--holdout", type=int, default=0, help="leave every K-th frame out of the fit, measure there")
    parser.add_argument("--iterations", type=int, default=600)
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
    # A repeated frame (frame-rate conversion) brings no new image: no 2D target there, the temporal term interpolates;
    # fitted, the avatar followed the detector's stairs (talk-show: 2D acceleration x5).
    repeated = repeated_frames(d / "input.mp4") if args.repeat_level > 0 else np.zeros(len(labels), bool)
    print(f"{100 * repeated.mean():.1f}% of the frames repeat the previous one: no 2D target there", flush=True)
    mhr = torch.jit.load(str(model), map_location=args.device)
    state = torch.load(args.checkpoint / "model.ckpt", map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    if "head_pose.keypoint_mapping" not in state:
        sys.exit("error: head_pose.keypoint_mapping not in the checkpoint")
    mapping = state["head_pose.keypoint_mapping"][MHR70].float().to(args.device)
    del state
    names = list(mhr.character_torch.skeleton.joint_names)
    parts = parameter_parts(mhr, args.device)
    free = [j for j, (part, _) in enumerate(parts) if part in args.parts]
    prior_w = torch.tensor([PRIOR[parts[j][0]] for j in free], device=args.device)
    # Per value: the legs keep less of the smoother's calm, so foot taps and stomps keep their amplitude.
    limb_w = {"legs": args.legs_temporal, "arms": args.arms_temporal, "wrists": args.arms_temporal}
    temporal_w = torch.tensor([limb_w.get(parts[j][0], args.temporal) for j in free],
                              device=args.device)
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
        if args.frames:
            keep = (ref["frames"] >= args.frames[0]) & (ref["frames"] <= args.frames[1])
            ref = {k: v[keep] for k, v in ref.items()}
        rows = np.searchsorted(a["frames"], ref["frames"])
        if (rows >= len(a["frames"])).any() or (a["frames"][np.minimum(rows, len(a["frames"]) - 1)] != ref["frames"]).any():
            sys.exit(f"error: person {person}: reference frames are not frames of {args.params}")
        frames = a["frames"][rows].astype(int)
        target = np.nan_to_num(ref["body"][:, COCO]).astype(np.float32)
        view_stats = {}
        view = keypoint_view.in_view(d, people, person, frames, ref["body"][:, COCO],
                                     cover=(ref["body"][:, COCO], ref["body_score"][:, COCO]), stats=view_stats)
        # RTMW's scores: seen points 2.5-7, guesses of hidden legs 1-2 (talk-show, AVA). A ramp from --min_score to
        # --full_score: at 0.5, guessed ankles on the torso's mask were targets, followed once the loss is annealed.
        score = ref["body_score"][:, COCO]
        ramp = np.clip((score - args.min_score) / max(args.full_score - args.min_score, 1e-6), 0, 1)
        weight = np.where(view, score * ramp, 0)
        weight = weight.astype(np.float32)
        weight = np.where(repeated[frames][:, None], 0, weight).astype(np.float32)
        if args.jerk_level > 0:
            weight = (weight * steadiness(target, frames, repeated, labels, args.jerk_level)).astype(np.float32)
        size = torso_size(target, weight)
        dist = edge = None
        if args.silhouette > 0:
            dist, edge = keypoint_view.allowed_distance(d, people, person, frames,
                                                        view_stats.get("partial_rows", np.zeros(0, int)),
                                                        scale=args.sil_scale)
            print(f"person {person}: {int(edge.sum())} frames at the picture's border (silhouette term)", flush=True)
        link = np.r_[False, (np.diff(frames) == 1) & (labels[frames[1:]] == labels[frames[:-1]])]
        # 2D speed of the seen points, torsos per second: where the image moves fast, the fit may move fast. Per part:
        # the median over the whole body stays low when only a foot stamps or an arm strikes, and held them back.
        # Step from the last new frame (repeats skipped), per frame of gap.
        last_new = np.maximum.accumulate(np.where(~repeated[frames], np.arange(len(frames)), -1))
        prev = np.r_[-1, last_new[:-1]]
        ok_prev = (prev >= 0) & ~repeated[frames]
        prev = np.maximum(prev, 0)
        gap = np.maximum(np.arange(len(frames)) - prev, 1)[:, None]
        step = (np.linalg.norm(target - target[prev], axis=-1) / size[:, None] / gap)[1:]
        both = ((weight > 0) & (weight[prev] > 0) & ok_prev[:, None])[1:]

        def part_speed(points):
            st, bo = step[:, points], both[:, points]
            if args.part_speed == "max":  # the fastest seen point of the part
                v = np.where(bo, st, -np.inf).max(1)
            else:  # the median seen point: one noisy point does not relax the whole limb
                v = np.nanmedian(np.where(bo, st, np.nan), axis=1) if bo.any() else np.zeros(len(st))
            sp = np.r_[0, np.where(bo.any(1), v, 0)] * fps
            return spread(np.where(link, np.nan_to_num(sp, neginf=0), 0))

        # A repeated frame has no speed of its own: it takes the one of the next new frame (else it stays stiff).
        rep = repeated[frames]
        next_new = np.minimum.accumulate(np.where(~rep, np.arange(len(frames)), len(frames) - 1)[::-1])[::-1]

        def spread(sp):
            return np.where(rep, sp[next_new], sp)


        speed = np.r_[0, np.where(both.any(1), np.nanmedian(np.where(both, step, np.nan), axis=1), 0)] * fps
        speed = spread(np.where(link, np.nan_to_num(speed), 0))
        speeds = {"legs": part_speed(LEG_POINTS), "arms": part_speed(ARM_POINTS), "wrists": part_speed(ARM_POINTS)}
        per_value = np.stack([speed if args.part_speed == "off" else speeds.get(parts[j][0], speed) for j in free], 1)
        tweight = (1 / (1 + (per_value / args.fast_speed) ** 2)).astype(np.float32)
        fast = speed >= np.percentile(speed[link], 90) if link.any() else np.zeros(len(frames), bool)
        held = np.zeros(len(frames), bool)
        if args.holdout:
            held[::args.holdout] = True
        fit_weight = np.where(held[:, None], 0, weight).astype(np.float32)
        print(f"person {person}: {len(frames)} frames, {100 * (weight > 0).mean():.1f}% of the points used, "
              f"{100 * (1 - view.mean()):.1f}% out of view, {view_stats.get('partial_mask_frames', 0)} frames with a "
              f"partial mask", flush=True)

        if out_dir:
            dev = args.device
            params = a["mhr_model_params"][rows].astype(np.float32).copy()
            # Offsets of the keypoints in their joints' frames, on the starting pose.
            k0, s0 = keypoints_cm(mhr, mapping, a, rows, dev)
            # Each keypoint rides on its nearest joint (median over the frames): the most rigid attachment.
            joints = torch.linalg.norm(k0[:, :, None] - s0[:, None, :, :3], dim=-1).median(0).values.argmin(1).tolist()
            if person == sorted(p.stem for p in (d / args.params).glob("*.npz"))[0]:
                print("joints:", ", ".join(f"{MHR70[i]}->{names[j]}" for i, j in enumerate(joints)), flush=True)
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
                                      free, prior_w, temporal_w, joints, local_all[lo:hi], args,
                                      None if dist is None else t(dist), None if edge is None else
                                      torch.from_numpy(edge[lo:hi]).to(dev))
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
