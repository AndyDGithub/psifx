"""Contact sheet of SAM 3D Body meshes projected over the original frames, one color per person.

Needs ``run_tracks.py --save_vertices``. The projection uses the camera SAM 3D Body assumed: focal from
the archive, principal point at the image center.

    python research/sam3d_body/contact_sheet.py .e2e_outputs/dev_2026-10-27/v1_example --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PALETTE = [(46, 204, 113), (231, 76, 60), (52, 152, 219), (241, 196, 15), (155, 89, 182)]


def project(points: np.ndarray, cam_t: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    p = points.astype(np.float32) + cam_t
    z = np.clip(p[:, 2], 1e-3, None)
    return np.stack([focal * p[:, 0] / z + width / 2, focal * p[:, 1] / z + height / 2], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n", type=int, default=8, help="number of frames on the sheet")
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--width", type=int, default=640, help="width of each tile")
    args = parser.parse_args()

    d = args.out_dir
    archives = {p.stem: np.load(p) for p in sorted((d / "sam3d_body").glob("*.npz"))}
    archives = {p: a for p, a in archives.items() if "vertices" in a}
    if not archives:
        sys.exit(f"error: no archive with vertices in {d / 'sam3d_body'} (run run_tracks.py --save_vertices)")
    frames = sorted(set(np.concatenate([a["frames"] for a in archives.values()]).tolist()))
    wanted = set(np.array(frames)[np.linspace(0, len(frames) - 1, args.n).astype(int)].tolist())
    order = sorted(archives, key=int)

    from psifx.io import video

    tiles = []
    with video.VideoReader(path=d / "input.mp4") as reader:
        for i, frame in enumerate(reader):
            if i > max(wanted):
                break
            if i not in wanted:
                continue
            h, w = frame.shape[:2]
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay = img.copy()
            for k, person in enumerate(order):
                a = archives[person]
                hit = np.nonzero(a["frames"] == i)[0]
                if not len(hit):
                    continue
                j = hit[0]
                uv = project(a["vertices"][j], a["cam_t"][j], float(a["focal"][j]), w, h)
                color = PALETTE[k % len(PALETTE)][::-1]
                # Filled triangles give a solid silhouette; edges alone are unreadable at 18k vertices.
                tri = uv[a["faces"]].astype(np.int32)
                cv2.fillPoly(overlay, list(tri), color, lineType=cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
            cv2.putText(img, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (args.width, int(args.width * h / w))))

    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + args.cols]) for r in range(0, len(tiles), args.cols)]
    path = d / "sam3d_body" / "contact_sheet.jpg"
    cv2.imwrite(str(path), np.vstack(rows))
    print(path)


if __name__ == "__main__":
    main()
