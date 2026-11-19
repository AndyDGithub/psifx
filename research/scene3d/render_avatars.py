"""Video with the people replaced by 3D avatars, rebuilt from the MHR parameters, with sound and subtitles.

The appearance levels of the plan (``13_plan_reproduction_3d.md``, section 4) are two independent choices:
- ``--background``: ``original`` (the video itself, avatars half transparent: a check), ``plate`` (the decor of the
  shot's camera group from ``decor_cameras.py``, people removed; grey when the shot has no plate) or ``grey``;
- ``--body``: ``own`` (the person's identity and scale parameters) or ``neutral`` (identity 0 and the mean scale of
  the model: everyone gets the same body, only the pose and expression are kept).
A2 = plate + own, A3 = plate + neutral, A4 = grey + neutral.

The mesh is rebuilt by MHR from ``shape_params``, ``mhr_model_params`` (136 pose then 68 scale values) and
``expr_params``, placed with ``cam_t`` and projected with the archive focal, then drawn by ``render_video.draw_mesh``.
A person keeps one color. Between two computed frames of the same shot, the parameters are interpolated
(``--hold`` keeps the old behaviour: the mesh stays until the next computed frame, steps at the computed rate).

A plate is trusted only if its camera comes back (``--min_plate_shots``) and its shots share enough decor keypoints
(``--min_plate_inliers``): a single shot does not separate the decor from a zoom transition, and an inserted video
(incrustation) only shares its logos with itself. Black pixels of a plate (decor never seen behind a person) are
filled by OpenCV inpainting: the avatar covers most of them.

    python research/scene3d/render_avatars.py .e2e_outputs/extrait_talkshow/pipeline \\
        --checkpoint .e2e_outputs/ext/ckpt/sam-3d-body-vith --background plate --body own
    # Stills for the report, original | render, no video:
    ... --still 3000 9000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sam3d_body"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
from render import PALETTE  # noqa: E402
from render_video import draw_mesh  # noqa: E402
from run_keyframes import cut_labels  # noqa: E402

GREY = (128, 128, 128)
POSE = 136  # mhr_model_params: 136 pose values, then 68 scale values.


def plates_by_frame(labels: np.ndarray, fps: float, decor: Path, size: tuple[int, int], min_shots: int,
                    min_inliers: int) -> tuple[list, int]:
    """Plate of each frame (None: no plate), matching each shot start to the shot starts of ``decor_cameras.json``."""
    report = json.loads((decor / "decor_cameras.json").read_text())
    starts = {}  # Shot start in seconds -> camera group.
    for camera in report["cameras"]:
        if camera["shots"] < min_shots or (camera["inliers_median"] or 0) < min_inliers:
            continue
        for s in camera["start_s"]:
            starts[round(s, 1)] = camera["group"]
    images = {}
    for g in set(starts.values()):
        path = decor / f"plate_{g}.jpg"
        if path.is_file():
            plate = cv2.imread(str(path))
            unseen = (plate.max(axis=2) < 8).astype(np.uint8)  # Never seen: stored black.
            images[g] = cv2.resize(cv2.inpaint(plate, unseen, 5, cv2.INPAINT_TELEA), size)
    first = np.r_[0, np.where(np.diff(labels))[0] + 1]  # First frame of each shot.
    by_shot, matched = {}, 0
    for f in first:
        # decor_cameras.py finds the cuts on other frames sizes: allow 0.1 s of difference.
        group = next((starts[k] for k in (round(f / fps + d, 1) for d in (0, -0.1, 0.1)) if k in starts), None)
        by_shot[labels[f]] = images.get(group)
        matched += by_shot[labels[f]] is not None
    return [by_shot[label] for label in labels], matched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4, subtitles.ass and the archives")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with model.ckpt and assets/mhr_model.pt")
    parser.add_argument("--params", default="sam3d_params", help="archives with the MHR parameters")
    parser.add_argument("--decor", default="decor_cameras", help="output of decor_cameras.py")
    parser.add_argument("--background", choices=("original", "plate", "grey"), default="plate")
    parser.add_argument("--body", choices=("own", "neutral"), default="own")
    parser.add_argument("--min_plate_shots", type=int, default=2)
    parser.add_argument("--min_plate_inliers", type=int, default=100)
    parser.add_argument("--cut", type=float, default=18, help="shot cut threshold, as in run_keyframes.py")
    parser.add_argument("--still", type=int, nargs="*", help="only write these frames, original | render, as jpg")
    parser.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="only render these frames (a clip to check), side by side with the original")
    parser.add_argument("--hold", action="store_true", help="hold each computed frame instead of interpolating")
    parser.add_argument("--output", type=Path, help="default: <pipeline>/avatars_<background>_<body>.mp4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.pipeline
    src, model = d / "input.mp4", args.checkpoint / "assets" / "mhr_model.pt"
    for path in (src, model, d / args.params):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    archives = {}
    for path in sorted((d / args.params).glob("*.npz"), key=lambda p: int(p.stem)):
        with np.load(path) as npz:  # Read once: an npz decompresses the whole array at each access.
            archives[path.stem] = {k: npz[k] for k in npz.files}
    if not archives:
        sys.exit(f"error: no archive in {d / args.params}")
    colors = {p: PALETTE[k % len(PALETTE)][::-1] for k, p in enumerate(archives)}  # BGR, order of render.py.

    if args.body == "neutral":
        state = torch.load(args.checkpoint / "model.ckpt", map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        if "head_pose.scale_mean" not in state:
            sys.exit("error: head_pose.scale_mean not in the checkpoint")
        scale_mean = state["head_pose.scale_mean"].numpy().astype(np.float32)
        del state
        for a in archives.values():
            a["shape_params"] = np.zeros_like(a["shape_params"])
            a["mhr_model_params"] = a["mhr_model_params"].copy()
            a["mhr_model_params"][:, POSE:] = scale_mean
    mhr = torch.jit.load(str(model), map_location=args.device)

    capture = cv2.VideoCapture(str(src))
    n_frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    labels = cut_labels(src, args.cut)[:n_frames]
    n_frames = len(labels)  # ffmpeg may decode one frame less than the container announces.
    plates = [None] * n_frames
    if args.background == "plate":
        if not (d / args.decor / "decor_cameras.json").is_file():
            sys.exit(f"error: missing {d / args.decor / 'decor_cameras.json'}")
        plates, matched = plates_by_frame(labels, fps, d / args.decor, (width, height),
                                          args.min_plate_shots, args.min_plate_inliers)
        print(f"{matched} of {labels[-1] + 1} shots have a plate; the others are grey", flush=True)
    # Row of each person's archive to draw on each frame (-1: none), held until its next computed frame, never
    # across a cut; weight of the next row for the interpolation (0 where the next row is too far or in another
    # shot).
    rows, weights = {}, {}
    for p, a in archives.items():
        r = np.full(n_frames, -1, np.int64)
        w = np.zeros(n_frames)
        frames = a["frames"].astype(int)
        for j, f in enumerate(frames):
            end = frames[j + 1] if j + 1 < len(frames) else n_frames
            r[f:min(end, f + 8)] = j
            if not args.hold and j + 1 < len(frames) and end - f <= 8 and labels[end] == labels[f]:
                w[f:end] = (np.arange(f, end) - f) / (end - f)
        r[(r >= 0) & (labels != labels[frames[np.clip(r, 0, None)]])] = -1
        rows[p], weights[p] = r, w

    def at(a: dict, key: str, j: int, w: float) -> np.ndarray:
        """Value of row j moved by w towards row j + 1: linear, the shortest way for the pose angles."""
        x = a[key][j].astype(np.float64)
        if w == 0:
            return x
        delta = a[key][j + 1] - x
        if key == "mhr_model_params":
            delta[3:POSE] = (delta[3:POSE] + np.pi) % (2 * np.pi) - np.pi
        return x + w * delta

    @torch.inference_mode()
    def vertices(a: dict, j: int, w: float) -> np.ndarray:
        v = mhr(*(torch.from_numpy(at(a, k, j, w)[None]).to(args.device).float()
                  for k in ("shape_params", "mhr_model_params", "expr_params")))[0][0].cpu().numpy() / 100
        v[:, [1, 2]] *= -1  # MHR axes to the camera axes of SAM 3D Body, as in mesh_eval.py.
        return v + at(a, "cam_t", j, w)

    def render(i: int, frame: np.ndarray) -> np.ndarray:
        if args.background == "original":
            image = frame.copy()
        elif plates[i] is not None:
            image = plates[i].copy()
        else:
            image = np.full_like(frame, GREY)
        canvas = image.copy()
        # Far people first, so a near one covers them.
        drawn = [(p, rows[p][i]) for p in archives if rows[p][i] >= 0]
        for p, j in sorted(drawn, key=lambda x: -archives[x[0]]["cam_t"][x[1]][2]):
            a = archives[p]
            w = weights[p][i]
            draw_mesh(canvas, vertices(a, j, w), a["faces"], float(at(a, "focal", j, w)), colors[p])
        if args.background == "original":  # Half transparent: the check of the fit.
            canvas = cv2.addWeighted(canvas, 0.65, image, 0.35, 0)
        return canvas

    name = f"avatars_{args.background}_{args.body}"
    if args.still:
        folder = d / "avatar_stills"
        folder.mkdir(exist_ok=True)
        for i in args.still:
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = capture.read()
            if not ok:
                sys.exit(f"error: cannot read frame {i}")
            cv2.imwrite(str(folder / f"{name}_{i}.jpg"), np.hstack([frame, render(i, frame)]))
        print(f"wrote {len(args.still)} stills in {folder}")
        return

    if args.range:
        first, last = max(0, args.range[0]), min(n_frames - 1, args.range[1])
        out = args.output or d / f"{name}_{first}_{last}.mp4"
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        writer = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                   f"{width}x{height // 2}", "-r", str(Fraction(fps).limit_denominator(1001)), "-i",
                                   "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
                                  stdin=subprocess.PIPE)
        for i in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            pair = np.hstack([frame, render(i, frame)])
            writer.stdin.write(cv2.resize(pair, (width, height // 2)).tobytes())
        writer.stdin.close()
        if writer.wait() != 0:
            sys.exit("error: ffmpeg could not write the clip")
        print(f"wrote {out}")
        print("RENDER_OK")
        return

    out = args.output or d / f"{name}.mp4"
    tmp = out.with_name(out.stem + "_noaudio.mp4")
    writer = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                               f"{width}x{height}", "-r", str(Fraction(fps).limit_denominator(1001)), "-i", "-",
                               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)],
                              stdin=subprocess.PIPE)
    start = time.time()
    for i in range(n_frames):
        ok, frame = capture.read()
        if not ok:
            break
        writer.stdin.write(render(i, frame).tobytes())
        if i % 1000 == 0:
            print(f"{i}/{n_frames} frames, {(time.time() - start) / (i + 1):.3f} s/frame", flush=True)
    writer.stdin.close()
    if writer.wait() != 0:
        sys.exit("error: ffmpeg could not write the video")
    seconds = time.time() - start

    # Original sound back, subtitles burnt in (run from their folder: the ass filter gets a plain name).
    subtitles = d / "subtitles.ass"
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp.resolve()), "-i", str(src.resolve()), "-map", "0:v",
               "-map", "1:a?"]
    if subtitles.is_file():
        command += ["-vf", f"ass={subtitles.name}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        print(f"warning: no {subtitles}, video without subtitles")
        command += ["-c:v", "copy"]
    command += ["-c:a", "aac", "-shortest", str(out.resolve())]
    if subprocess.run(command, cwd=d).returncode != 0:
        sys.exit("error: ffmpeg could not add the sound and subtitles")
    tmp.unlink()
    (d / f"{name}.json").write_text(json.dumps({"frames": n_frames, "seconds": round(seconds),
                                                "seconds_per_frame": round(seconds / n_frames, 4)}, indent=1))
    print(f"wrote {out} in {seconds / 60:.0f} min")
    print("RENDER_OK")


if __name__ == "__main__":
    main()
