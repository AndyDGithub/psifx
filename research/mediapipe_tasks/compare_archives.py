"""Compares MediaPipe pose archives of the legacy Solutions API and of the Tasks API, part by part.

For each part: share of frames where each API finds it, and, on frames where both do, the median
distance between the points, divided by the person's torso height (shoulders-hips midpoints) so
that it reads the same on a close-up and a wide shot.

    python research/mediapipe_tasks/compare_archives.py .e2e_outputs/v1_example/poses_mp \
        .e2e_outputs/mp_tasks/v1_example/poses_mp
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

PARTS = ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d")
SHOULDERS, HIPS = (11, 12), (23, 24)


def load(path: Path) -> dict[str, dict]:
    frames = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if member.isfile() and name.endswith(".json") and name != "edges.json":
                frames[name] = json.load(archive.extractfile(member))
    return frames


def points(frame: dict, part: str) -> np.ndarray | None:
    values = np.asarray(frame.get(part, []), dtype=np.float64).reshape(-1, 3)
    return values[:, :2] if values.size and np.abs(values[:, :2]).sum() > 0 else None


def torso(frame: dict) -> float | None:
    body = points(frame, "pose_keypoints_2d")
    if body is None:
        return None
    height = np.linalg.norm(body[list(SHOULDERS)].mean(0) - body[list(HIPS)].mean(0))
    return height if height > 1 else None


def compare(old_path: Path, new_path: Path) -> dict:
    old, new = load(old_path), load(new_path)
    if set(old) != set(new):
        sys.exit(f"error: {old_path} and {new_path} have different frames")
    out = {"frames": len(old)}
    for part in PARTS:
        found_old = found_new = 0
        distances = []
        for name in old:
            a, b = points(old[name], part), points(new[name], part)
            found_old += a is not None
            found_new += b is not None
            scale = torso(old[name])
            if a is not None and b is not None and scale:
                n = min(len(a), len(b))  # the legacy face had 478 points when found, 468 zeros otherwise
                distances.append(np.median(np.linalg.norm(a[:n] - b[:n], axis=1)) / scale)
        out[part] = {
            "found_legacy": round(found_old / len(old), 3),
            "found_tasks": round(found_new / len(old), 3),
            "median_dist_torso": round(float(np.median(distances)), 3) if distances else None,
            "p90_dist_torso": round(float(np.percentile(distances, 90)), 3) if distances else None,
        }
    lengths = {len(f.get("face_keypoints_2d", [])) // 3 for f in new.values()}
    out["face_points_tasks"] = sorted(lengths)
    out["face_points_legacy"] = sorted({len(f.get("face_keypoints_2d", [])) // 3 for f in old.values()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legacy_dir", type=Path)
    parser.add_argument("tasks_dir", type=Path)
    args = parser.parse_args()
    result = {}
    for path in sorted(args.legacy_dir.glob("*.tar.gz")):
        other = args.tasks_dir / path.name
        if not other.exists():
            sys.exit(f"error: {other} is missing")
        result[path.name] = compare(path, other)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
