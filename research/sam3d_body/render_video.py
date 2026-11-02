"""Video of the SAM 3D Body meshes over the original video, with the subtitles colored by speaker.

Needs ``run_tracks.py --save_vertices``. Each mesh is projected with the camera SAM 3D Body assumed
(focal from the archive, principal point at the image center) and shaded by the angle of each
triangle to the camera, so the body reads as 3D. With ``--stride N``, a mesh stays on screen for the
N frames until the next one. A person keeps the color of ``research/e2e/render.py``; the subtitles
are its ``subtitles.ass``, burnt in with ffmpeg, and the original sound is kept.

    python research/sam3d_body/render_video.py .e2e_outputs/talkshow_2026-11-02/pipeline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
from render import PALETTE  # noqa: E402

DEPTH_SLICES, SHADES = 48, 6  # Painter's algorithm by depth slice, triangles grouped by shade level.


def project(points: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    z = np.clip(points[:, 2], 1e-3, None)
    return np.stack([focal * points[:, 0] / z + width / 2, focal * points[:, 1] / z + height / 2], axis=1)


def draw_mesh(img: np.ndarray, vertices: np.ndarray, faces: np.ndarray, focal: float, color) -> None:
    """Fills the mesh triangles far to near, darker as they turn away from the camera."""
    h, w = img.shape[:2]
    uv = project(vertices, focal, w, h)
    tri = vertices[faces]  # (F, 3, 3), camera frame
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    shade = np.abs(normals[:, 2])  # Facing the camera: 1, seen edge-on: 0.
    level = np.minimum((shade * SHADES).astype(int), SHADES - 1)
    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(-depth)
    polygons = uv[faces].astype(np.int32)
    base = np.asarray(color, dtype=np.float32)
    for chunk in np.array_split(order, DEPTH_SLICES):
        for k in range(SHADES):
            sel = chunk[level[chunk] == k]
            if len(sel):
                c = base * (0.35 + 0.65 * (k + 0.5) / SHADES)
                cv2.fillPoly(img, list(polygons[sel]), tuple(int(v) for v in c), lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="end-to-end folder with input.mp4 and sam3d_body/")
    parser.add_argument("--stride", type=int, default=None,
                        help="frames a mesh stays on screen (default: the run_tracks.py stride, read from the archives)")
    parser.add_argument("--alpha", type=float, default=0.65, help="mesh opacity")
    parser.add_argument("--output", type=Path, default=None, help="default: <out_dir>/render_sam3d_body.mp4")
    args = parser.parse_args()

    d = args.out_dir
    src = d / "input.mp4"
    # Read each array once: indexing an NpzFile decompresses the whole array again at every access.
    archives = {}
    for path in sorted((d / "sam3d_body").glob("*.npz")):
        with np.load(path) as npz:
            archives[path.stem] = {key: npz[key] for key in npz.files}
    archives = {p: a for p, a in archives.items() if "vertices" in a and len(a["frames"])}
    if not src.is_file():
        sys.exit(f"error: missing {src}")
    if not archives:
        sys.exit(f"error: no archive with vertices in {d / 'sam3d_body'} (run run_tracks.py --save_vertices)")
    out = args.output or d / "render_sam3d_body.mp4"
    subtitles = d / "subtitles.ass"

    # Same colors as render.py: one per person, in the order of the pose archives.
    ids = {p.name.removesuffix(".tar.gz") for sub in ("poses_body", "poses_mp") for p in (d / sub).glob("*.tar.gz")}
    order = sorted(ids | set(archives), key=int)
    colors = {p: PALETTE[k % len(PALETTE)][::-1] for k, p in enumerate(order)}  # BGR for OpenCV

    stride = args.stride or min(int(np.diff(a["frames"]).min()) if len(a["frames"]) > 1 else 1
                                for a in archives.values())
    from psifx.io import video

    with video.VideoReader(path=src) as reader:
        n_frames, fps = len(reader), float(Fraction(reader.frame_rate))
    # For each person and frame, the archive row to draw (-1: none), held for ``stride`` frames.
    rows = {}
    for person, a in archives.items():
        r = np.full(n_frames + stride, -1, dtype=np.int64)
        for j, f in enumerate(a["frames"]):
            r[f:f + stride] = j
        rows[person] = r[:n_frames]

    tmp = out.with_name(out.stem + "_noaudio.mp4")
    with (
        video.VideoReader(path=src) as reader,
        video.VideoWriter(path=tmp, input_dict={"-r": str(fps)},
                          output_dict={"-c:v": "libx264", "-crf": "18", "-pix_fmt": "yuv420p"},
                          overwrite=True) as writer,
    ):
        for i, frame in enumerate(tqdm(reader, total=n_frames, desc="Rendering meshes")):
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            overlay = img.copy()
            labels = []
            for person in sorted(archives, key=int):
                j = rows[person][i]
                if j < 0:
                    continue
                a = archives[person]
                vertices = a["vertices"][j].astype(np.float32) + a["cam_t"][j]
                draw_mesh(overlay, vertices, a["faces"], float(a["focal"][j]), colors[person])
                head = a["keypoints_2d"][j][0]
                labels.append((f"P{person}", (int(head[0]) - 20, int(head[1]) - 40), colors[person]))
            img = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)
            size = img.shape[0] / 600  # Label height about 4 % of the image.
            for text, (x, y), color in labels:
                pos = (x, y - int(20 * size))
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), int(6 * size), cv2.LINE_AA)
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, int(2.5 * size), cv2.LINE_AA)
            writer.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Original sound back, subtitles burnt in. Run from the subtitles' folder so the ass filter gets a
    # plain file name (its argument would otherwise need escaping); the other paths are absolute.
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp.resolve()), "-i", str(src.resolve()),
               "-map", "0:v", "-map", "1:a?"]
    if subtitles.is_file():
        command += ["-vf", f"ass={subtitles.name}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        print(f"warning: no {subtitles}, run research/e2e/render.py first for the subtitles")
        command += ["-c:v", "copy"]
    command += ["-c:a", "aac", "-shortest", str(out.resolve())]
    if subprocess.run(command, cwd=subtitles.parent).returncode != 0:
        sys.exit("error: ffmpeg could not add the sound and subtitles")
    tmp.unlink()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
