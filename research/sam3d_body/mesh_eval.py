"""Measures the SAM 3D Body archives of ``run_tracks.py``, the reference for the temporal and camera work.

- ``rebuild``: is the mesh recovered from the MHR parameters alone? Rebuilt vertices against the vertices of
  ``--reference`` (a previous run with ``--save_vertices``), in millimeters, on a sample of frames.
- ``repeat``: are two runs the same? 3D keypoints against ``--reference``.
- ``annotations``: 2D accuracy on the hand annotations of ``--annotations``. An annotated frame not processed
  (with ``--stride``) uses the frame before it. There is no score per point: phantom measures do not apply.
- ``per person``: shoulder width, speed relative to the pelvis, acceleration, depth. Speeds include the cuts.

Writes ``<archives>/mesh_eval.json`` and stops with an error when ``--max_rebuild_mm`` is exceeded.

    python research/sam3d_body/mesh_eval.py .e2e_outputs/extrait_talkshow/pipeline/sam3d_params \\
        --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith \\
        --reference .e2e_outputs/extrait_talkshow/pipeline/sam3d_body \\
        --annotations .e2e_outputs/extrait_talkshow/pose_annotation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research" / "pose_annotation"), str(Path(__file__).resolve().parent)]
from evaluate import evaluate, load_ground_truth  # noqa: E402
from run_tracks import MHR70_TO_COCO17  # noqa: E402

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 9, 10
FRAME_STEP_SAMPLES = 200


def rebuild(archive, reference, mhr, device: str) -> dict:
    """Vertices from the parameters alone, against the stored ones (same frames only)."""
    index = {int(f): i for i, f in enumerate(reference["frames"])}
    common = [i for i, f in enumerate(archive["frames"]) if int(f) in index]
    if not common:
        return {"frames": 0}
    errors = []
    for i in np.array(common)[np.linspace(0, len(common) - 1, min(FRAME_STEP_SAMPLES, len(common))).astype(int)]:
        shape, params, expr = (torch.from_numpy(archive[k][i][None]).to(device) for k in
                               ("shape_params", "mhr_model_params", "expr_params"))
        with torch.inference_mode():
            vertices = mhr(shape, params, expr)[0][0].cpu().numpy() / 100  # Centimeters to meters.
        vertices[..., [1, 2]] *= -1  # Camera system difference, as in the SAM 3D Body head.
        stored = reference["vertices"][index[int(archive["frames"][i])]].astype(np.float64)
        errors.append(np.linalg.norm(vertices - stored, axis=1))
    errors = np.stack(errors) * 1000
    return {"frames": errors.shape[0], "mm_median": float(np.median(errors)), "mm_p99": float(np.percentile(errors, 99)),
            "mm_max": float(errors.max())}


def per_person(a) -> dict:
    k3, frames, depth = a["keypoints_3d"], a["frames"], a["cam_t"][:, 2]
    shoulders = np.linalg.norm(k3[:, LEFT_SHOULDER] - k3[:, RIGHT_SHOULDER], axis=1) * 100
    relative = k3 - ((k3[:, LEFT_HIP] + k3[:, RIGHT_HIP]) / 2)[:, None]
    step = np.diff(frames) == np.median(np.diff(frames))
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=2)[step] * 1000
    both = step[1:] & step[:-1]
    acceleration = np.linalg.norm(relative[2:] - 2 * relative[1:-1] + relative[:-2], axis=2)[both] * 1000
    percentiles = lambda values, q: [round(float(v), 2) for v in np.percentile(values, q)]  # noqa: E731
    return {"frames": len(frames), "shoulder_cm_p5_50_95": percentiles(shoulders, [5, 50, 95]),
            "speed_mm_p50_95": percentiles(speed, [50, 95]), "acceleration_mm_p50_95": percentiles(acceleration, [50, 95]),
            "depth_m_p1_50_99": percentiles(depth, [1, 50, 99])}


def annotations(archives: dict, folder: Path) -> dict:
    index = {person: {int(f): i for i, f in enumerate(a["frames"])} for person, a in archives.items()}
    predictions = {}
    with open(folder / "frames.csv", newline="") as f:
        for row in csv.DictReader(f):
            frame, people = int(row["frame"]), []
            for person, a in archives.items():
                i = index[person].get(frame, index[person].get(frame - 1))
                if i is not None:
                    points = a["keypoints_2d"][i][MHR70_TO_COCO17].astype(float)
                    people.append(np.concatenate([points, np.ones((len(points), 1))], axis=1))
            predictions[row["file"]] = people
    result = evaluate(predictions, load_ground_truth(folder / "ground_truth.json"), min_score=0.3)
    return {k: float(result[k]) for k in ("oks", "found", "pck@0.05", "pck@0.1", "extra_people")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archives", type=Path, help="folder of <person>.npz from run_tracks.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with assets/mhr_model.pt")
    parser.add_argument("--reference", type=Path, help="earlier archives with vertices")
    parser.add_argument("--annotations", type=Path, help="folder with frames.csv and ground_truth.json")
    parser.add_argument("--max_rebuild_mm", type=float, default=1.0, help="largest accepted median rebuild error")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = sorted(args.archives.glob("*.npz"))
    model = args.checkpoint / "assets" / "mhr_model.pt"
    if not paths or not model.is_file():
        sys.exit(f"error: need {args.archives}/*.npz and {model}")
    # A TorchScript model from the SAM 3D Body checkpoint, also what the estimator loads.
    mhr = torch.jit.load(str(model), map_location=args.device)
    archives = {p.stem: np.load(p) for p in paths if len(np.load(p).files) > 1 and "frames" in np.load(p).files}
    report = {"people": {person: per_person(a) for person, a in archives.items()}}
    if args.annotations:
        report["annotations"] = annotations(archives, args.annotations)
    worst = 0.0
    if args.reference:
        for person, a in archives.items():
            path = args.reference / f"{person}.npz"
            if not path.is_file():
                sys.exit(f"error: missing reference {path}")
            r = np.load(path)
            result = rebuild(a, r, mhr, args.device)
            index = {int(f): i for i, f in enumerate(r["frames"])}
            shared = [(i, index[int(f)]) for i, f in enumerate(a["frames"]) if int(f) in index]
            if shared:
                ours, theirs = (np.array(x) for x in zip(*shared))
                difference = np.linalg.norm(a["keypoints_3d"][ours] - r["keypoints_3d"][theirs], axis=2) * 1000
                result["repeat_keypoints_mm_median"] = float(np.median(difference))
            report["people"][person]["rebuild"] = result
            worst = max(worst, result.get("mm_median", 0.0))
    (args.archives / "mesh_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    if worst > args.max_rebuild_mm:
        sys.exit(f"error: the parameters do not rebuild the mesh, median error {worst:.2f} mm > {args.max_rebuild_mm} mm")


if __name__ == "__main__":
    main()
