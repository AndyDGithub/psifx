"""Where does the avatar shake? Per body part: tremble, spikes ("teleports") and depth pumping, from MHR joints.

The user sees three defects on the rendered avatars: hands that tremble, extreme values that make the avatar jump for
a frame or two, crossed legs badly rendered. This script puts numbers on them so that each fix can be measured.

Joints come from the MHR skeleton (127 joints, the second output of ``mhr_model.pt``), in camera space (camera
translation added), so a depth error shows as a jump. Inside each run (no gap, no shot cut, as in
``run_keyframes.split_segments``), per part:

- ``accel_mm``: p50 / p95 / p99 of the second difference of the part's joints (relative to their parent part, so a
  hand trembling is not hidden by the body moving), per sample;
- ``spike``: per sample, distance between the joint and the mean of its two neighbours; a "teleport" is a spike
  over ``--spike_mm`` (a joint away for one sample, then back). Counted per minute, and the worst ones listed
  (frame and person) for a visual check;
- ``root_depth``: same for the pelvis depth alone (size pumping).

Parts: ``root`` (pelvis in camera space), ``torso_head`` and ``legs`` / ``wrists`` (relative to the pelvis),
``fingers`` (relative to their wrist).

Writes ``<pipeline>/motion_diag_<params>.json``.

    python research/sam3d_body/motion_diag.py .e2e_outputs/extrait_talkshow/pipeline \\
        --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith --params sam3d_people sam3d_smooth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_keyframes import cut_labels, split_segments  # noqa: E402


def joint_groups(names: list[str]) -> dict[str, list[int]]:
    def pick(*keys, side=("l_", "r_")):
        return [i for i, n in enumerate(names) if n.startswith(side) and any(k in n for k in keys)
                and not n.endswith(("_null", "_proc"))]
    return {
        "pelvis": [names.index("root")],
        "torso_head": [names.index(n) for n in ("c_spine1", "c_spine3", "c_neck", "c_head")],
        "legs": pick("lowleg", "foot", "ball"),
        "wrists": [names.index("l_wrist"), names.index("r_wrist")],
        "elbows_shoulders": pick("uparm", "lowarm", "clavicle"),
        "fingers_l": pick("pinky", "ring", "middle", "index", "thumb", side=("l_",)),
        "fingers_r": pick("pinky", "ring", "middle", "index", "thumb", side=("r_",)),
    }


@torch.inference_mode()
def camera_joints(a: dict, mhr, device: str, batch: int = 512) -> np.ndarray:
    """(n, 127, 3) joints in meters, camera axes, camera translation added."""
    out = []
    for s in range(0, len(a["frames"]), batch):
        _, skel = mhr(*(torch.from_numpy(a[k][s:s + batch]).to(device).float()
                        for k in ("shape_params", "mhr_model_params", "expr_params")))
        j = skel[..., :3].cpu().numpy() / 100
        j[..., [1, 2]] *= -1  # MHR axes to the camera axes, as in mesh_eval.py.
        out.append(j + a["cam_t"][s:s + batch, None])
    return np.concatenate(out)


def part_series(j: np.ndarray, g: dict) -> dict[str, np.ndarray]:
    pelvis = j[:, g["pelvis"]]
    return {
        "root": pelvis,
        "torso_head": j[:, g["torso_head"]] - pelvis,
        "legs": j[:, g["legs"]] - pelvis,
        "elbows_shoulders": j[:, g["elbows_shoulders"]] - pelvis,
        "wrists": j[:, g["wrists"]] - pelvis,
        "fingers": np.concatenate([j[:, g["fingers_l"]] - j[:, [g["wrists"][0]]],
                                   j[:, g["fingers_r"]] - j[:, [g["wrists"][1]]]], axis=1),
        "root_depth": pelvis[..., 2:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with assets/mhr_model.pt")
    parser.add_argument("--params", nargs="+", default=["sam3d_people", "sam3d_smooth"])
    parser.add_argument("--spike_mm", type=float, default=40.0, help="a joint this far from its neighbours' mean")
    parser.add_argument("--cut", type=float, default=18)
    parser.add_argument("--worst", type=int, default=400, help="worst spikes listed per variant")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    video, model = args.pipeline / "input.mp4", args.checkpoint / "assets" / "mhr_model.pt"
    for path in [video, model, *(args.pipeline / p for p in args.params)]:
        if not path.exists():
            sys.exit(f"error: missing {path}")
    mhr = torch.jit.load(str(model), map_location=args.device)
    groups = joint_groups(list(mhr.character_torch.skeleton.joint_names))
    labels = cut_labels(video, args.cut)
    import cv2
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    for params in args.params:
        accel, spikes, events, seconds = {}, {}, [], 0.0
        for path in sorted((args.pipeline / params).glob("*.npz"), key=lambda p: int(p.stem)):
            with np.load(path) as npz:
                a = {k: npz[k] for k in npz.files}
            frames = a["frames"].astype(int)
            grid = int(np.median(np.diff(frames)))
            parts = part_series(camera_joints(a, mhr, args.device), groups)
            position = {int(f): i for i, f in enumerate(frames)}
            for segment in split_segments(frames, grid, labels):
                if len(segment) < 3:
                    continue
                rows = np.array([position[int(f)] for f in segment])
                seconds += len(segment) * grid / fps
                for part, x in parts.items():
                    x = x[rows]
                    second = x[2:] - 2 * x[1:-1] + x[:-2]  # (n-2, joints, 3)
                    accel.setdefault(part, []).append(np.linalg.norm(second, axis=2).mean(1) * 1000)
                    spike = np.linalg.norm(second / 2, axis=2).max(1) * 1000  # Worst joint of the part.
                    spikes.setdefault(part, []).append(spike)
                    for k in np.where(spike > args.spike_mm)[0]:
                        events.append((float(spike[k]), path.stem, int(segment[k + 1]), part))
        minutes = seconds / 60
        report = {"minutes": round(minutes, 1), "spike_mm": args.spike_mm, "parts": {}}
        for part in accel:
            a_all, s_all = np.concatenate(accel[part]), np.concatenate(spikes[part])
            report["parts"][part] = {
                "accel_mm_p50_95_99": [round(float(v), 1) for v in np.percentile(a_all, [50, 95, 99])],
                "spikes_per_min": round(float((s_all > args.spike_mm).sum() / minutes), 1),
            }
        events.sort(reverse=True)
        report["worst"] = [{"mm": round(mm), "person": p, "frame": f, "part": part}
                           for mm, p, f, part in events[:args.worst]]
        (args.pipeline / f"motion_diag_{params}.json").write_text(json.dumps(report, indent=1))
        print(params, json.dumps(report["parts"]), flush=True)
        print(" worst:", report["worst"][:8], flush=True)
    print("DIAG_OK")


if __name__ == "__main__":
    main()
