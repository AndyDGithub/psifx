"""Compares SAM 3D Body with RTMO on the same tracks: coverage and 2D agreement.

For each person and each frame SAM 3D Body processed, its MHR-70 keypoints are reduced to COCO-17 and
compared with the RTMO keypoints of ``poses_body/`` (confidence >= ``--min_conf``). The distance is
divided by the RTMO torso height (shoulder center to hip center), so it does not depend on the
person's size in the image. Writes ``sam3d_body/compare_rtmo.json``.

    python research/sam3d_body/compare_rtmo.py .e2e_outputs/dev_2026-10-27/v3_talkshow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
from render import load_people  # noqa: E402
from run_tracks import MHR70_TO_COCO17  # noqa: E402

COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
              "right_knee", "left_ankle", "right_ankle"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min_conf", type=float, default=0.3)
    args = parser.parse_args()

    d = args.out_dir
    archives = {p.stem: np.load(p) for p in sorted((d / "sam3d_body").glob("*.npz"))}
    if not archives:
        sys.exit(f"error: no archive in {d / 'sam3d_body'}")
    n_frames = max(int(a["frames"].max()) + 1 for a in archives.values() if len(a.get("frames", [])))
    rtmo = load_people(d / "poses_body", n_frames)

    report = {}
    for person, a in archives.items():
        if "frames" not in a or person not in rtmo:
            report[person] = {"frames": int(len(a.get("frames", [])))}
            continue
        frames = a["frames"]
        ours = a["keypoints_2d"][:, MHR70_TO_COCO17]  # (N, 17, 2)
        ref = rtmo[person]["pose_keypoints_2d"][frames]  # (N, 17, 3)
        valid = ~np.isnan(ref[..., 2]) & (ref[..., 2] >= args.min_conf)
        torso = np.linalg.norm((ref[:, 5, :2] + ref[:, 6, :2]) / 2 - (ref[:, 11, :2] + ref[:, 12, :2]) / 2, axis=1)
        torso_ok = valid[:, [5, 6, 11, 12]].all(axis=1) & (torso > 1)
        dist = np.linalg.norm(ours - ref[..., :2], axis=2) / np.where(torso_ok, torso, np.nan)[:, None]
        dist = np.where(valid & torso_ok[:, None], dist, np.nan)
        per_joint = {COCO_NAMES[k]: round(float(np.nanmedian(dist[:, k])), 3)
                     for k in range(17) if np.isfinite(dist[:, k]).any()}
        report[person] = {
            "frames": int(len(frames)),
            "frames_with_rtmo_torso": int(torso_ok.sum()),
            "median_distance_torso": round(float(np.nanmedian(dist)), 3) if np.isfinite(dist).any() else None,
            "pck_0.2_torso": round(float(np.mean(dist[np.isfinite(dist)] < 0.2)), 3) if np.isfinite(dist).any() else None,
            "median_per_joint": per_joint,
        }
    path = d / "sam3d_body" / "compare_rtmo.json"
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
