"""Evaluate a psifx pose CLI tool (``psifx video pose <tool> single-inference``)
on COCO person-keypoint images.

Unlike ``run_accuracy.py`` (which calls each backend's Python API directly on
individual images), this goes through psifx's real pipeline end to end:
COCO images are letterboxed into one video (one image per frame), the psifx
tool writes its usual ``.tar.gz`` archive, and the archive is scored against
the COCO ground truth with the same OKS/PCK as ``metrics.py``. Every model
branch that adds a psifx pose tool is therefore measured on exactly the same
input and output format as MediaPipe, the model it is meant to replace.

Used by ``tests/integration/video/test_pose_coco_accuracy.py`` and by
``run_psifx_cli_benchmark.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image

from common import COCO_KEYPOINT_NAMES, MEDIAPIPE_TO_COCO17, NUM_COCO_KEYPOINTS, PersonPose
from metrics import object_keypoint_similarity, pck


@dataclass
class GroundTruth:
    """COCO ground truth for one frame, already in letterboxed video coordinates."""

    keypoints: np.ndarray  # (17, 2)
    visible: np.ndarray  # (17,) bool
    area: float
    bbox_diagonal: float


def load_coco_samples(annotations_path: Path, images_dir: Path) -> List[dict]:
    """Return ``[{"image_path", "annotation"}]``, one per image, sorted by image id.

    Raises if an image has several annotated people: matching would then be
    ambiguous for single-person tools.
    """
    coco = json.loads(Path(annotations_path).read_text())
    images = {img["id"]: img for img in coco["images"]}
    by_image: Dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    samples = []
    for image_id in sorted(by_image):
        anns = by_image[image_id]
        if len(anns) != 1:
            raise ValueError(f"image {image_id} has {len(anns)} annotated people, expected exactly 1")
        image_path = Path(images_dir) / images[image_id]["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"missing COCO image {image_path}")
        samples.append({"image_path": image_path, "annotation": anns[0]})
    if not samples:
        raise ValueError(f"no annotations found in {annotations_path}")
    return samples


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize ``image`` to fit in a ``size``x``size`` black canvas, keeping aspect ratio.

    Returns ``(canvas, scale, offset_x, offset_y)`` so that a point ``(x, y)``
    of the original image lands at ``(x * scale + offset_x, y * scale + offset_y)``.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    ox, oy = (size - new_w) // 2, (size - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas, scale, float(ox), float(oy)


def build_coco_video(samples: List[dict], video_path: Path, size: int = 640) -> List[GroundTruth]:
    """Write one letterboxed frame per COCO image and return the matching ground truth."""
    # Imported here so the pure metric helpers above stay usable without ffmpeg.
    from psifx.io.video import VideoWriter

    if size % 2:
        raise ValueError("size must be even (yuv420p needs even dimensions)")

    ground_truth = []
    with VideoWriter(
        path=video_path,
        input_dict={"-r": "1"},
        # Near-lossless so codec artifacts don't blur the accuracy comparison.
        output_dict={"-c:v": "libx264", "-crf": "10", "-pix_fmt": "yuv420p"},
        overwrite=True,
    ) as writer:
        for sample in samples:
            image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
            frame, scale, ox, oy = letterbox(image, size)
            writer.write(frame)

            ann = sample["annotation"]
            kp = np.asarray(ann["keypoints"], dtype=np.float64).reshape(NUM_COCO_KEYPOINTS, 3)
            _, _, bw, bh = ann["bbox"]
            ground_truth.append(
                GroundTruth(
                    keypoints=kp[:, :2] * scale + [ox, oy],
                    visible=kp[:, 2] > 0,
                    area=float(ann["area"]) * scale ** 2,
                    bbox_diagonal=float(np.hypot(bw, bh)) * scale,
                )
            )
    return ground_truth


# --- psifx archive -> COCO-17 adapters --------------------------------------
# Each takes one decoded frame of a psifx pose archive ({part: flat [x, y, c, ...]})
# and returns a (17, 3) array (x, y, confidence), NaN where undetected.
# Model branches register their own adapter here.

def _mediapipe_to_coco17(frame: dict) -> np.ndarray:
    points = np.asarray(frame["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
    out = np.full((NUM_COCO_KEYPOINTS, 3), np.nan)
    # psifx writes all-zero landmarks when MediaPipe detects nobody.
    if not np.any(points):
        return out
    for name, mp_idx in MEDIAPIPE_TO_COCO17.items():
        out[COCO_KEYPOINT_NAMES.index(name)] = points[mp_idx]
    return out


def _coco17(frame: dict) -> np.ndarray:
    """Tools built on psifx.video.pose.coco17 already write COCO-17 order."""
    points = np.asarray(frame["pose_keypoints_2d"], dtype=np.float64).reshape(NUM_COCO_KEYPOINTS, 3)
    if not np.any(points):
        return np.full((NUM_COCO_KEYPOINTS, 3), np.nan)
    return points


ADAPTERS: Dict[str, Callable[[dict], np.ndarray]] = {
    "mediapipe": _mediapipe_to_coco17,
    "coco17": _coco17,
<<<<<<< HEAD
    "rtmo": _coco17,
=======
    "vitpose": _coco17,
>>>>>>> 0a5727f (Add ViTPose pose estimation tool)
}


def read_archive(poses_path: Path) -> List[dict]:
    """Decode a psifx pose archive into a list of per-frame dicts, in frame order."""
    from psifx.io.tar import TarReader

    files = TarReader.read(poses_path, verbose=False)
    files.pop("edges.json", None)
    frames = sorted(files.items(), key=lambda kv: int(kv[0].replace(".json", "")))
    return [json.loads(content) for _, content in frames]


def evaluate_archive(poses_path: Path, ground_truth: List[GroundTruth], tool: str) -> Dict[str, float]:
    """Score a psifx pose archive against ``ground_truth`` (one entry per frame).

    Undetected frames count as OKS = PCK = 0, so a tool can't look better by
    skipping hard images.
    """
    if tool not in ADAPTERS:
        raise KeyError(f"no COCO-17 adapter registered for tool '{tool}' (known: {sorted(ADAPTERS)})")
    frames = read_archive(poses_path)
    if len(frames) != len(ground_truth):
        raise ValueError(f"archive has {len(frames)} frames, expected {len(ground_truth)}")

    oks_values, pck_values, detected = [], [], 0
    for frame, gt in zip(frames, ground_truth):
        coco = ADAPTERS[tool](frame)
        if np.all(np.isnan(coco[:, 0])):
            oks_values.append(0.0)
            pck_values.append(0.0)
            continue
        detected += 1
        pred = PersonPose(keypoints=coco[:, :2], scores=coco[:, 2])
        oks_values.append(object_keypoint_similarity(pred, gt.keypoints, gt.visible, gt.area) or 0.0)
        pck_values.append(pck(pred, gt.keypoints, gt.visible, gt.bbox_diagonal) or 0.0)

    return {
        "oks": float(np.mean(oks_values)),
        "pck": float(np.mean(pck_values)),
        "detection_rate": detected / len(ground_truth),
        "n_images": len(ground_truth),
    }
