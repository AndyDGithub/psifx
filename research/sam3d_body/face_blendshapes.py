"""Facial expression of each merged person, frame by frame: the 52 ARKit blendshapes of MediaPipe Face Landmarker.

SAM 3D Body does not estimate the face (``mhr_head.py`` sets the expression and the jaw to 0), so the avatars keep a
still face. MediaPipe Face Landmarker gives, per face, 52 blendshape scores in [0, 1] named as Apple ARKit
(``jawOpen``, ``mouthSmileLeft``, ``browInnerUp``, ``eyeBlinkLeft``...), which ``face_expression.py`` turns into
the 72 FACS expressions of MHR.

The faces are those already found by psifx (``poses_mp``: MediaPipe Holistic face mesh per SAM3 track), given to
the merged people by ``people.json`` (pieces left out, such as an inserted video, are not used). Each face is cut
from the frame as a square of ``--margin`` times its landmark box, resized to ``--size`` pixels and read alone
(image mode: a face is never mixed with the face of another shot).

``--video`` reads another video at the same places: the avatar video, to measure what of the expression the
avatar's face carries (``face_expression.py --reread``).

Writes ``<pipeline>/<output>/<person>.npz``: ``frames``, ``scores`` (n, 52, NaN where no face was read),
``names``, ``box`` (face size in pixels), ``mouth`` (inner-lip gap over mouth width), ``source`` (0: MediaPipe Holistic face, 1: SAM 3D Body head).

    python research/sam3d_body/face_blendshapes.py .e2e_outputs/extrait_talkshow/pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from psifx.io import video
from psifx.utils.constants import MEDIAPIPE_FACE_PATH
from psifx.video.pose.mediapipe.tool import bundle_path
from psifx.video.pose.tool import PoseEstimationTool


def face_boxes(poses: dict, n_frames: int) -> np.ndarray:
    """(n_frames, 4) x1, y1, x2, y2 of the MediaPipe face landmarks, NaN where no face."""
    boxes = np.full((n_frames, 4), np.nan)
    for i, pose in poses.items():
        if i >= n_frames:
            continue
        face = np.asarray(pose.get("face_keypoints_2d", []), dtype=float).reshape(-1, 3)
        points = face[(face[:, :2] > 0).all(1), :2]
        if len(points) >= 100:  # A real face mesh, not a few stray points.
            boxes[i] = (*points.min(0), *points.max(0))
    return boxes


def head_boxes(params: Path, person: str, n_frames: int) -> np.ndarray:
    """(n_frames, 4) face box from SAM 3D Body's head keypoints (MHR70: nose 0, eyes 1-2, ears 3-4), found on
    nearly every frame where MediaPipe Holistic misses the face; a frame between two computed ones takes the
    previous one. The box is sized like a face landmark box: 1.2 x the larger of the ear distance and twice the
    eye distance, its center a bit below the eyes."""
    boxes = np.full((n_frames, 4), np.nan)
    path = params / f"{person}.npz"
    if not path.is_file():
        return boxes
    with np.load(path) as npz:
        frames, k = npz["frames"].astype(int), npz["keypoints_2d"][:, :5].astype(float)
    side = 1.2 * np.maximum(np.linalg.norm(k[:, 3] - k[:, 4], axis=1), 2 * np.linalg.norm(k[:, 1] - k[:, 2], axis=1))
    center = k[:, :3].mean(1) + np.stack([np.zeros_like(side), 0.15 * side], 1)
    for j, f in enumerate(frames):
        end = min(frames[j + 1] if j + 1 < len(frames) else f + 1, f + 2, n_frames)
        boxes[f:end] = (*(center[j] - side[j] / 2), *(center[j] + side[j] / 2))
    return boxes


def square_crop(image: np.ndarray, box: np.ndarray, margin: float, size: int) -> np.ndarray:
    """Square around the box center, side margin x the box's longer side, padded with grey outside the image."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1]) * margin / 2
    x1, y1, x2, y2 = (int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    pad = max(0, -x1, -y1, x2 - image.shape[1], y2 - image.shape[0])
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    crop = image[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path, help="folder with input.mp4, poses_mp/ and people.json")
    parser.add_argument("--people", type=Path, help="default: <pipeline>/people.json")
    parser.add_argument("--video", type=Path, help="video to read the faces from (default: <pipeline>/input.mp4)")
    parser.add_argument("--output", default="face_blendshapes", help="output folder, inside the pipeline")
    parser.add_argument("--margin", type=float, default=2.4, help="crop side over the face landmark box side")
    parser.add_argument("--size", type=int, default=256, help="crop side in pixels")
    parser.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="only these frames")
    parser.add_argument("--video_start", type=int, default=0,
                        help="frame of input.mp4 shown on the first frame of --video (a clip of render_avatars --range)")
    parser.add_argument("--head_params", help="person archives of SAM 3D Body (e.g. sam3d_people): face boxes from "
                                              "their head keypoints where MediaPipe Holistic found no face")
    args = parser.parse_args()

    d = args.pipeline
    people_path, source = args.people or d / "people.json", args.video or d / "input.mp4"
    for path in (d / "input.mp4", d / "poses_mp", people_path, source):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    people = json.loads(people_path.read_text())
    with video.VideoReader(path=d / "input.mp4") as reader:
        n_frames = len(reader)

    decoder = PoseEstimationTool(device="cpu", verbose=False)
    boxes = {a.name.removesuffix(".tar.gz"): face_boxes(decoder.decode(a)[0], n_frames)
             for a in sorted((d / "poses_mp").glob("*.tar.gz"))}
    first, last = args.range or (0, n_frames - 1)
    heads = {}
    if args.head_params:
        if not (d / args.head_params).is_dir():
            sys.exit(f"error: missing {d / args.head_params}")
        heads = {p: head_boxes(d / args.head_params, str(p), n_frames) for p in range(people["people"])}
    todo: dict[int, list] = {}  # frame -> [(person, box, source)]
    for piece in people["assignment"]:
        if not isinstance(piece["person"], int):
            continue
        track, person = str(piece["track"]), piece["person"]
        for f in range(max(first, piece["frames"][0]), min(last, piece["frames"][1] + 1, n_frames - 1) + 1):
            if track in boxes and not np.isnan(boxes[track][f]).any():
                todo.setdefault(f, []).append((person, boxes[track][f], 0))
            elif person in heads and not np.isnan(heads[person][f]).any():
                todo.setdefault(f, []).append((person, heads[person][f], 1))

    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(bundle_path(MEDIAPIPE_FACE_PATH))),
                                    running_mode=RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
                                    min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
    results: dict[int, dict[str, list]] = {p: {"frames": [], "scores": [], "box": [], "source": [], "mouth": []}
                                           for p in range(people["people"])}
    names = None
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        sys.exit(f"error: cannot open {source}")
    if first < args.video_start:
        sys.exit(f"error: --video starts at frame {args.video_start}, frame {first} is asked")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first - args.video_start)
    read = 0
    with FaceLandmarker.create_from_options(options) as landmarker:
        for f in range(first, (max(todo) if todo else first) + 1):
            ok, image = capture.read()
            if not ok:
                sys.exit(f"error: cannot read frame {f} of {source}")
            for person, box, source in todo.get(f, []):
                crop = cv2.cvtColor(square_crop(image, box, args.margin, args.size), cv2.COLOR_BGR2RGB)
                out = landmarker.detect(Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
                scores = np.full(52, np.nan, np.float32)
                mouth = np.nan
                if out.face_landmarks:
                    # Inner-lip gap over mouth width (landmarks 13/14 and 61/291), as in link_speakers.py: the
                    # mouth opening of speech, which jawOpen barely shows on these faces.
                    lm = np.array([[p.x, p.y] for p in out.face_landmarks[0]])
                    width = np.linalg.norm(lm[61] - lm[291])
                    if width > 1e-3:
                        mouth = float(np.linalg.norm(lm[13] - lm[14]) / width)
                if out.face_blendshapes:
                    shapes = out.face_blendshapes[0]
                    if names is None:
                        names = [s.category_name for s in shapes]
                    scores[:] = [s.score for s in shapes]
                    read += 1
                r = results[person]
                r["frames"].append(f), r["scores"].append(scores), r["box"].append(max(box[2] - box[0], box[3] - box[1]))
                r["source"].append(source)
                r["mouth"].append(mouth)
            if f % 2000 == 0:
                print(f"frame {f}: {read} faces read", flush=True)
    capture.release()
    if names is None:
        sys.exit("error: no face was read")

    out_dir = d / args.output
    out_dir.mkdir(exist_ok=True)
    for person, r in results.items():
        scores = np.array(r["scores"], np.float32).reshape(-1, 52)
        np.savez_compressed(out_dir / f"{person}.npz", frames=np.array(r["frames"], np.int64), scores=scores,
                            names=np.array(names), box=np.array(r["box"], np.float32),
                            source=np.array(r["source"], np.int8), mouth=np.array(r["mouth"], np.float32))
        read_share = float(np.mean(~np.isnan(scores[:, 0]))) if len(scores) else 0.0
        source = np.array(r["source"])
        from_head = float(np.mean(~np.isnan(scores[source == 1, 0]))) if (source == 1).any() else 0.0
        print(f"person {person}: {len(scores)} faces, {read_share:.1%} read; {int((source == 1).sum())} placed by "
              f"the SAM 3D Body head, {from_head:.1%} of them read")
    print("BLENDSHAPES_OK")


if __name__ == "__main__":
    main()
