"""What do interpolated frames lose? Key frame archives against a run of the model on every frame.

For each archive set (``run_keyframes.py``, or a fixed 1-out-of-N taken from ``--dense``):
- share of the person-frames computed, so the expected gain;
- ``computed``: do the computed frames equal the dense run? Below that level, an error is only run-to-run variation;
- ``interpolated``: 3D keypoints around the pelvis (p95 per frame, wrists), and the mesh rebuilt from the interpolated
  MHR parameters against the stored vertices (a sample of frames, ``--sample``);
- ``gestures``: the same wrist error on the 10 % fastest wrist frames of the dense run, where interpolation should fail;
- ``annotations``: 2D accuracy on the hand annotations.

    python research/sam3d_body/keyframe_eval.py .e2e_outputs/extrait_talkshow/pipeline \\
        --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith --annotations .e2e_outputs/extrait_talkshow/pose_annotation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_eval import LEFT_HIP, RIGHT_HIP, annotations  # noqa: E402
from run_keyframes import cut_labels, densify, split_segments  # noqa: E402

WRISTS = [62, 41]
KEYS = ("keypoints_2d", "keypoints_3d", "cam_t", "focal", "global_rot", "body_pose_params", "hand_pose_params",
        "scale_params", "shape_params", "expr_params", "mhr_model_params")


def relative(k3: np.ndarray) -> np.ndarray:
    return k3 - k3[:, [LEFT_HIP, RIGHT_HIP]].mean(1, keepdims=True)


def fixed(dense, step: int, grid: int, labels: np.ndarray) -> dict:
    """One frame out of ``step`` per segment, plus its end, the others interpolated by the code of the real run."""
    known, segments = {}, split_segments(dense["frames"], grid, labels)
    position = {int(f): i for i, f in enumerate(dense["frames"])}
    for segment in segments:
        for f in {*segment[::step // grid].tolist(), int(segment[-1])}:
            known[f] = {k: dense[k][position[f]] for k in KEYS}
    return densify(known, segments)


def rounded(values, q) -> list[float]:
    return [round(float(v), 2) for v in np.percentile(values, q)] if len(values) else []


def measure(archive, dense, vertices, mhr, device: str, sample: int) -> dict:
    position = {int(f): i for i, f in enumerate(dense["frames"])}
    mine = np.array([i for i, f in enumerate(archive["frames"]) if int(f) in position])
    theirs = np.array([position[int(archive["frames"][i])] for i in mine])
    computed = archive["computed"][mine]
    k_a, k_d = relative(archive["keypoints_3d"][mine]), relative(dense["keypoints_3d"][theirs])
    errors = np.linalg.norm(k_a - k_d, axis=2) * 1000
    body, wrist = np.percentile(errors, 95, axis=1), errors[:, WRISTS]
    dense_rel = relative(dense["keypoints_3d"])
    speed = np.linalg.norm(np.diff(dense_rel[:, WRISTS], axis=0, prepend=dense_rel[:1, WRISTS]), axis=2) * 1000
    fast = speed[theirs] >= np.percentile(speed, 90)
    result = {"frames": len(mine), "computed": int(computed.sum()),
              "computed_body_p95_mm_p50_p99": rounded(body[computed], [50, 99]),
              "interpolated_body_p95_mm_p50_p95": rounded(body[~computed], [50, 95]),
              "interpolated_wrist_mm_p50_p95": rounded(wrist[~computed].ravel(), [50, 95]),
              "gestures_wrist_mm_p50_p95": rounded(wrist[~computed][fast[~computed]], [50, 95])}
    if vertices is not None and (~computed).any():
        sampled = mine[~computed][::sample]
        stored = {int(f): i for i, f in enumerate(vertices[0])}
        per_frame, hand = [], []
        tie = None
        for i in sampled:
            frame = int(archive["frames"][i])
            if frame not in stored:
                continue
            with torch.inference_mode():
                rebuilt = mhr(*(torch.from_numpy(archive[k][i][None]).to(device).float() for k in
                                ("shape_params", "mhr_model_params", "expr_params")))[0][0].cpu().numpy() / 100
            rebuilt[..., [1, 2]] *= -1
            reference = vertices[1][stored[frame]].astype(np.float64)
            if tie is None:  # The vertices closest to the wrists: the mesh error where gestures happen.
                wrists = dense["keypoints_3d"][position[frame]][WRISTS]
                tie = np.linalg.norm(reference[:, None] - wrists[None], axis=2).argmin(0)
            distance = np.linalg.norm(rebuilt - reference, axis=1) * 1000
            per_frame.append(np.percentile(distance, 95)), hand.append(distance[tie])
        result["mesh_frames"] = len(per_frame)
        result["mesh_p95_mm_p50_p95"] = rounded(per_frame, [50, 95])
        result["mesh_wrist_mm_p50_p95"] = rounded(np.ravel(hand), [50, 95])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4 and the archive folders")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with assets/mhr_model.pt")
    parser.add_argument("--keyframes", default="sam3d_keyframes", help="archives of run_keyframes.py")
    parser.add_argument("--dense", default="sam3d_params", help="run on every frame, with the MHR parameters")
    parser.add_argument("--vertices", default="sam3d_body", help="same frames with the stored vertices")
    parser.add_argument("--fixed", type=int, nargs="*", default=[10], help="fixed key frames to compare: 1 out of N")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--sample", type=int, default=5, help="mesh rebuilt on one interpolated frame out of N")
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--cut", type=float, default=18)
    parser.add_argument("--output", type=Path, help="default: <keyframes>/keyframe_eval.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    folders = {name: args.pipeline / getattr(args, name) for name in ("keyframes", "dense", "vertices")}
    model = args.checkpoint / "assets" / "mhr_model.pt"
    for path in (*folders.values(), model):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    people = sorted(p.stem for p in folders["keyframes"].glob("*.npz"))
    if not people:
        sys.exit(f"error: no archive in {folders['keyframes']}")
    mhr = torch.jit.load(str(model), map_location=args.device)
    labels = cut_labels(args.pipeline / "input.mp4", args.cut)
    dense = {p: dict(np.load(folders["dense"] / f"{p}.npz")) for p in people}
    methods = {"keyframes": {p: dict(np.load(folders["keyframes"] / f"{p}.npz")) for p in people}}
    for step in args.fixed:
        if step % args.grid:
            sys.exit(f"error: --fixed {step} is not a multiple of --grid")
        methods[f"fixed_{step}"] = {p: fixed(dense[p], step, args.grid, labels) for p in people}

    report: dict = {}
    for name, archives in methods.items():
        report[name] = {"share": round(sum(int(a["computed"].sum()) for a in archives.values())
                                       / sum(len(a["frames"]) for a in archives.values()), 4), "people": {}}
    for person in people:
        # Loaded once and kept for all the methods: reading from the npz would decompress it at each access.
        stored = np.load(folders["vertices"] / f"{person}.npz")
        vertices = (stored["frames"], stored["vertices"])
        for name, archives in methods.items():
            report[name]["people"][person] = measure(archives[person], dense[person], vertices, mhr, args.device, args.sample)
            print(name, person, report[name]["people"][person], flush=True)
    if args.annotations:
        report["dense_oks"] = annotations(dense, args.annotations)["oks"]
        for name, archives in methods.items():
            report[name]["annotations"] = annotations(archives, args.annotations)
    run = folders["keyframes"] / "keyframes_run.json"
    if run.is_file():
        report["run"] = json.loads(run.read_text())
    (args.output or folders["keyframes"] / "keyframe_eval.json").write_text(json.dumps(report, indent=1))
    print({k: v["annotations"]["oks"] for k, v in report.items() if isinstance(v, dict) and "annotations" in v},
          "dense", report.get("dense_oks"))
    print("KEYFRAME_EVAL_OK")


if __name__ == "__main__":
    main()
