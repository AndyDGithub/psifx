"""2D hand keypoints of each merged person, from detectors made for it: the reference the avatar hands are measured
and refitted against (``hand_refit.py``).

There is no hand annotation on the talk-show (``pose_annotation`` has the 17 COCO points), so the reference is the
best open 2D detectors, run top-down on each person of the SAM 3D Body archives:

1. **RTMW-x** (whole body, 133 COCO-WholeBody points, 70.2 mAP, the best open whole-body model) on the person box
   (from the archive's upper-body keypoints);
2. **RTMPose-m hand** (21 points, trained on five hand datasets) on a square crop around each hand found at step 1:
   at 384x288 for the whole person, a hand is about 30 pixels wide, so step 2 reads it again at 256x256.

Writes ``<pipeline>/<output>/<person>.npz``: ``frames``, ``body`` (n, 133, 2), ``body_score`` (n, 133),
``hands`` (n, 2, 21, 2) and ``hand_score`` (n, 2, 21), hand 0 the person's left, points in COCO-WholeBody hand order
(wrist, then thumb, index, middle, ring, pinky from base to tip), ``hand_box`` (n, 2) crop side in pixels.

    python research/sam3d_body/hand_reference.py .e2e_outputs/extrait_talkshow/pipeline --params sam3d_people60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RTMW = ("https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip", (288, 384))
HAND = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip", (256, 256))
# MHR70 keypoints of the upper body (head, shoulders, elbows, hips, both hands): the person box. The legs are left
# out, the model invents them under the table.
UPPER = [*range(0, 11), *range(21, 63)]
# COCO-WholeBody: left hand 91-111, right hand 112-132.
HAND_SLICES = (slice(91, 112), slice(112, 133))
# MHR70 hand keypoints of the person's left and right hand, used when RTMW does not see the hand.
MHR70_HANDS = (range(42, 63), range(21, 42))
MIN_HAND_SCORE = 0.3


def box_of(points: np.ndarray, image_size: tuple[int, int], grow: float = 1.0, min_side: float = 0) -> list | None:
    """x1, y1, x2, y2 around the points, grown about its center, clipped to the image; None if empty."""
    if len(points) == 0:
        return None
    (x1, y1), (x2, y2) = points.min(0), points.max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = max((x2 - x1) * grow, min_side) / 2, max((y2 - y1) * grow, min_side) / 2
    w, h = image_size
    box = [max(0, cx - half_w), max(0, cy - half_h), min(w, cx + half_w), min(h, cy + half_h)]
    return box if box[2] - box[0] > 4 and box[3] - box[1] > 4 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--params", default="sam3d_people60", help="person archives of SAM 3D Body")
    parser.add_argument("--output", default="hand_reference")
    parser.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d = args.pipeline
    for path in (d / "input.mp4", d / args.params):
        if not path.exists():
            sys.exit(f"error: missing {path}")
    from rtmlib import RTMPose

    body_model = RTMPose(RTMW[0], model_input_size=RTMW[1], backend="onnxruntime", device=args.device)
    hand_model = RTMPose(HAND[0], model_input_size=HAND[1], backend="onnxruntime", device=args.device)

    archives = {}
    for path in sorted((d / args.params).glob("*.npz")):
        with np.load(path) as npz:
            archives[path.stem] = {"frames": npz["frames"].astype(int), "k2d": npz["keypoints_2d"][:, :, :2]}
    if not archives:
        sys.exit(f"error: no archive in {d / args.params}")
    capture = cv2.VideoCapture(str(d / "input.mp4"))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first, last = args.range or (0, n_frames - 1)
    todo: dict[int, list] = {}  # frame -> [(person, row)]
    for person, a in archives.items():
        for row, f in enumerate(a["frames"]):
            if first <= f <= last:
                todo.setdefault(int(f), []).append((person, row))
    if not todo:
        sys.exit("error: no person in the asked frames")
    out = {p: {"frames": [], "body": [], "body_score": [], "hands": [], "hand_score": [], "hand_box": []}
           for p in archives}

    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    start = time.time()
    for f in range(first, max(todo) + 1):
        ok, image = capture.read()  # BGR, as rtmlib expects
        if not ok:
            sys.exit(f"error: cannot read frame {f}")
        for person, row in todo.get(f, []):
            k2d = archives[person]["k2d"][row]
            inside = k2d[UPPER][(k2d[UPPER] >= 0).all(1) & (k2d[UPPER, 0] < size[0]) & (k2d[UPPER, 1] < size[1])]
            box = box_of(inside, size, grow=1.1)
            body, body_score = np.full((133, 2), np.nan, np.float32), np.zeros(133, np.float32)
            hands, hand_score = np.full((2, 21, 2), np.nan, np.float32), np.zeros((2, 21), np.float32)
            hand_box = np.zeros(2, np.float32)
            if box is not None:
                k, s = body_model(image, bboxes=[box])
                body, body_score = k[0], s[0]
                for side in (0, 1):
                    seen = body_score[HAND_SLICES[side]] > MIN_HAND_SCORE
                    points = body[HAND_SLICES[side]][seen] if seen.sum() >= 5 else k2d[list(MHR70_HANDS[side])]
                    # Square crop: the hand's extent, at least a fifth of the person box height (a closed fist
                    # gives few spread points); rtmlib adds 25 % around it.
                    hb = box_of(points, size, grow=1.2, min_side=0.2 * (box[3] - box[1]))
                    if hb is None:
                        continue
                    side_px = max(hb[2] - hb[0], hb[3] - hb[1])
                    cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                    k, s = hand_model(image, bboxes=[[cx - side_px / 2, cy - side_px / 2,
                                                      cx + side_px / 2, cy + side_px / 2]])
                    hands[side], hand_score[side], hand_box[side] = k[0], s[0], side_px
            o = out[person]
            o["frames"].append(f), o["body"].append(body), o["body_score"].append(body_score)
            o["hands"].append(hands), o["hand_score"].append(hand_score), o["hand_box"].append(hand_box)
        if (f - first) % 1000 == 0:
            print(f"frame {f}: {time.time() - start:.0f} s", flush=True)
    capture.release()

    out_dir = d / args.output
    out_dir.mkdir(exist_ok=True)
    for person, o in out.items():
        arrays = {k: np.asarray(v, np.int64 if k == "frames" else np.float32) for k, v in o.items()}
        np.savez_compressed(out_dir / f"{person}.npz", **arrays)
        seen = (arrays["hand_score"] > MIN_HAND_SCORE).mean() if len(arrays["frames"]) else 0
        print(f"person {person}: {len(arrays['frames'])} frames, {seen:.1%} hand points above {MIN_HAND_SCORE}")
    print(f"HANDS_OK in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
