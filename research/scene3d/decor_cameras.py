"""Which camera filmed each shot? Keypoints on the decor answer it, without any 3D.

Shots come from frame differences. SIFT keypoints are taken on the decor only: the SAM3 masks of the people
and the ``--ignore`` boxes (channel logos) are removed. Two shots share a camera when a RANSAC homography keeps
enough inliers and barely moves the image center. For each camera group, the median of the frames over the
visible decor gives a background plate, the people removed.

    python research/scene3d/decor_cameras.py .e2e_outputs/extrait_talkshow/pipeline \\
        --ignore 0,0,.14,.14 .86,0,1,.14
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 960, 540, 5  # Enough for the decor, and it keeps a 5 min video in memory.
RATIO, RANSAC_PX = 0.75, 3.0


def frames(path: Path, pix_fmt: str):
    """Yields the frames of a video, resized, at FPS."""
    channels = {"gray": 1, "bgr24": 3}[pix_fmt]
    size = W * H * channels
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo",
           "-pix_fmt", pix_fmt, "-"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while len(buffer := process.stdout.read(size)) == size:
        yield np.frombuffer(buffer, np.uint8).reshape((H, W, channels) if channels > 1 else (H, W))
    if process.wait() != 0:
        sys.exit(f"error: ffmpeg failed on {path}")


def match(a: tuple, b: tuple, matcher) -> tuple[int, float, np.ndarray | None]:
    """Inliers of the homography between two sets of keypoints, how far it moves the image center, inlier points."""
    (points_a, desc_a), (points_b, desc_b) = a, b
    if desc_a is None or desc_b is None or len(points_a) < 8 or len(points_b) < 8:
        return 0, np.nan, None
    good = [m[0] for m in matcher.knnMatch(desc_a, desc_b, k=2) if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    if len(good) < 12:
        return 0, np.nan, None
    source, target = points_a[[g.queryIdx for g in good]], points_b[[g.trainIdx for g in good]]
    homography, status = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PX)
    if homography is None:
        return 0, np.nan, None
    center = np.array([[[W / 2, H / 2]]], np.float32)
    shift = float(np.linalg.norm(cv2.perspectiveTransform(center, homography)[0, 0] - center[0, 0]))
    keep = status.ravel().astype(bool)
    return int(keep.sum()), shift, np.hstack([source[keep], target[keep]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="pipeline folder with input.mp4, masks/ and masks_dropped/")
    parser.add_argument("--ignore", nargs="*", default=[], help="boxes x0,y0,x1,y1 in [0, 1] without decor (logos)")
    parser.add_argument("--cut", type=float, default=18, help="mean gray difference above which a frame starts a shot")
    parser.add_argument("--same_inliers", type=int, default=60, help="inliers for two shots to share a camera")
    parser.add_argument("--same_shift", type=float, default=12, help="center shift (px at 960 wide) for the same")
    parser.add_argument("--link_inliers", type=int, default=40, help="inliers for two cameras to see the same decor")
    parser.add_argument("--plate_samples", type=int, default=60, help="frames per camera group for its plate")
    parser.add_argument("--output", type=Path, default=None, help="default: out_dir/decor_cameras")
    args = parser.parse_args()

    video_path = args.out_dir / "input.mp4"
    mask_paths = sorted(args.out_dir.glob("masks/*.mp4")) + sorted(args.out_dir.glob("masks_dropped/*.mp4"))
    if not video_path.is_file() or not mask_paths:
        sys.exit(f"error: {args.out_dir} needs input.mp4 and masks/*.mp4")
    output = args.output or args.out_dir / "decor_cameras"
    output.mkdir(parents=True, exist_ok=True)

    gray = np.stack(list(frames(video_path, "gray")))
    count = len(gray)
    people = np.zeros_like(gray, dtype=bool)
    for path in mask_paths:
        for i, mask in enumerate(frames(path, "gray")):
            if i < count:
                people[i] |= mask > 127

    difference = np.abs(gray[1:].astype(np.int16) - gray[:-1]).mean(axis=(1, 2))
    cuts = [0, *(np.where(difference > args.cut)[0] + 1).tolist(), count]
    shots = [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= FPS]  # Under 1 s: fades and flashes.
    middles = [(a + b) // 2 for a, b in shots]

    not_ignored = np.ones((H, W), bool)
    for box in args.ignore:
        x0, y0, x1, y1 = (float(v) for v in box.split(","))
        not_ignored[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = False
    kernel = np.ones((25, 25), np.uint8)  # Mask borders are imprecise: stay away from the people.
    decor = np.stack([(cv2.dilate(people[m].astype(np.uint8), kernel) == 0) & not_ignored for m in middles])

    sift = cv2.SIFT_create(nfeatures=3000)
    features = []
    for m, region in zip(middles, decor):
        points, descriptors = sift.detectAndCompute(gray[m], region.astype(np.uint8) * 255)
        features.append((np.array([p.pt for p in points], np.float32).reshape(-1, 2), descriptors))
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})

    n = len(shots)
    inliers, shifts = np.zeros((n, n), int), np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            inliers[i, j], shifts[i, j], _ = match(features[i], features[j], matcher)
            inliers[j, i], shifts[j, i] = inliers[i, j], shifts[i, j]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    same = (inliers >= args.same_inliers) & (shifts < args.same_shift)
    for i, j in zip(*np.where(np.triu(same, 1))):
        parent[find(int(i))] = find(int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: -sum(shots[i][1] - shots[i][0] for i in g))

    links = []
    for x in range(len(ordered)):
        for y in range(x + 1, len(ordered)):
            best = max((inliers[i, j], i, j) for i in ordered[x] for j in ordered[y])
            if best[0] >= args.link_inliers:
                links.append({"groups": [x, y], "inliers": int(best[0]), "shots": [int(best[1]), int(best[2])]})
    links.sort(key=lambda link: -link["inliers"])

    # Second pass in color, only on the frames of the plates, evenly taken inside each large group.
    wanted: dict[int, int] = {}
    for g, group in enumerate(ordered[:8]):
        pool = [f for i in group for f in range(*shots[i])]
        for f in np.array(pool)[np.linspace(0, len(pool) - 1, min(args.plate_samples, len(pool))).astype(int)]:
            wanted[int(f)] = g
    stacks: dict[int, list[np.ndarray]] = {}
    for f, image in enumerate(frames(video_path, "bgr24")):
        if f in wanted:
            sample = image.astype(np.float32)
            sample[cv2.dilate(people[f].astype(np.uint8), kernel) > 0] = np.nan
            stacks.setdefault(wanted[f], []).append(sample)

    report = []
    for g, group in enumerate(ordered):
        pairs = [(i, j) for k, i in enumerate(group) for j in group[k + 1:] if same[i, j]]
        entry = {"group": g, "shots": len(group), "seconds": sum(shots[i][1] - shots[i][0] for i in group) / FPS,
                 "start_s": [shots[i][0] / FPS for i in group],
                 "inliers_median": float(np.median([inliers[p] for p in pairs])) if pairs else None,
                 "shift_px_median": float(np.median([shifts[p] for p in pairs])) if pairs else None}
        if g in stacks:
            samples = np.stack(stacks.pop(g))
            seen = np.isfinite(samples[..., 0]).sum(axis=0)
            plate = np.nan_to_num(np.nanmedian(samples, axis=0))  # Never-seen pixels stay black.
            cv2.imwrite(str(output / f"plate_{g}.jpg"), plate.astype(np.uint8))
            entry["plate_unseen_share"] = float((seen < 3).mean())
        report.append(entry)

    # One row per large group: a shot, then its plate.
    rows = []
    for g, group in enumerate(ordered[:8]):
        shot = cv2.resize(cv2.cvtColor(gray[middles[group[0]]], cv2.COLOR_GRAY2BGR), (480, 270))
        cv2.putText(shot, f"group {g}, {report[g]['seconds']:.0f} s", (8, 260), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        rows.append(np.hstack([shot, cv2.resize(cv2.imread(str(output / f"plate_{g}.jpg")), (480, 270))]))
    cv2.imwrite(str(output / "groups_and_plates.jpg"), np.vstack(rows))

    summary = {"shots": n, "cuts": len(cuts) - 2, "groups": len(ordered), "keypoints_per_shot_p10_p50_p90":
               np.percentile([len(f[0]) for f in features], [10, 50, 90]).astype(int).tolist(),
               "links_between_groups": links[:20], "cameras": report}
    (output / "decor_cameras.json").write_text(json.dumps(summary, indent=2))
    print({k: summary[k] for k in ("shots", "cuts", "groups", "keypoints_per_shot_p10_p50_p90")})
    for entry in report[:12]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items() if k != "start_s"})
    print("links between groups", links[:12])


if __name__ == "__main__":
    main()
