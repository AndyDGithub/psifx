"""Pose models to compare on the annotated frames, behind one interface: ``load(spec, device)``
returns a function from an RGB frame to the people found, each a (17, 3) COCO-17 array
``[x, y, score]`` in pixels.

A spec is ``family:name`` with optional ``@option,option``, for example
``vitpose:usyd-community/vitpose-plus-base@fp16`` or ``rtmlib:wholebody-performance@fp16``:

- ``vitpose``: ViTPose, ViTPose++ (the COCO expert, ``dataset_index`` 0) and SynthPose (its first
  17 points are COCO), from ``transformers``, on the RT-DETR boxes psifx uses;
- ``sapiens2``: Sapiens2 pose (308 points, the 17 COCO ones kept), same boxes;
- ``rtmlib``: ``rtmo-<mode>`` (one stage), ``body-<mode>`` (YOLOX + RTMPose) and ``wholebody-<mode>``
  (YOLOX + RTMW, the first 17 of its 133 points are COCO), with rtmlib's own detector;
- ``rfdetr``: RF-DETR Keypoint preview (one stage);
- ``yolo``: Ultralytics YOLO pose (AGPL-3.0: measured for reference only).

Options: ``fp16`` (half precision; for the top-down models the detector stays in fp32),
``compile`` (``torch.compile`` of the pose model), ``optimize`` (RF-DETR's own inference optimization).
"""

from __future__ import annotations

import tempfile
from typing import Callable

import numpy as np
# Before rtmlib: the CUDA libraries it preloads for onnxruntime break a later torch import.
import torch

from tests.pose_eval.coco import COCO_KEYPOINT_NAMES

N = len(COCO_KEYPOINT_NAMES)
Predictor = Callable[[np.ndarray], list[np.ndarray]]
DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"  # the person detector of psifx's ViTPose tool
PERSON_LABEL = 0


def parse(spec: str) -> tuple[str, str, set[str]]:
    family, _, rest = spec.partition(":")
    name, _, options = rest.partition("@")
    return family, name, set(filter(None, options.split(",")))


def check_options(spec: str, options: set[str], allowed: set[str]):
    if unknown := options - allowed:
        raise ValueError(f"{spec}: unknown options {sorted(unknown)}, allowed {sorted(allowed)}")


class PersonDetector:
    """RT-DETR person boxes ``[x1, y1, x2, y2]``, as in psifx's ViTPose tool."""

    def __init__(self, device: str, threshold: float = 0.3):
        from transformers import AutoProcessor, RTDetrForObjectDetection

        self.device, self.threshold = device, threshold
        self.processor = AutoProcessor.from_pretrained(DETECTOR)
        self.model = RTDetrForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        result = self.processor.post_process_object_detection(
            self.model(**inputs), target_sizes=[image.shape[:2]], threshold=self.threshold)[0]
        return result["boxes"][result["labels"] == PERSON_LABEL].cpu().numpy()


def top_down(spec: str, name: str, options: set[str], device: str, coco_indices: Callable,
             batch: int = 64) -> Predictor:
    """A ``transformers`` pose model on RT-DETR boxes; ``coco_indices(config)`` picks the COCO points."""
    from transformers import AutoModelForPoseEstimation, AutoProcessor

    check_options(spec, options, {"fp16", "compile"})
    dtype = torch.float16 if "fp16" in options else torch.float32
    detector = PersonDetector(device)
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForPoseEstimation.from_pretrained(name, dtype=dtype).to(device).eval()
    if "compile" in options:
        model = torch.compile(model)
    indices = coco_indices(model.config)
    # ViTPose++ is a mixture of experts, one per training dataset: 0 is COCO.
    moe = getattr(getattr(model.config, "backbone_config", None), "num_experts", 1) > 1

    @torch.inference_mode()
    def predict(image: np.ndarray) -> list[np.ndarray]:
        boxes = detector(image)
        if len(boxes) == 0:
            return []
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        out = []
        # Sapiens2 works at 1024x768: several people at once do not fit in 8 GB, so one at a time.
        for start in range(0, len(boxes_xywh), batch):
            chunk = boxes_xywh[start:start + batch]
            inputs = processor(image, boxes=[chunk], return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            if moe:
                inputs["dataset_index"] = torch.zeros(len(chunk), dtype=torch.long, device=device)
            outputs = model(**inputs)
            if hasattr(outputs, "heatmaps"):
                outputs.heatmaps = outputs.heatmaps.float()
            for p in processor.post_process_pose_estimation(outputs, boxes=[chunk])[0]:
                out.append(np.concatenate([p["keypoints"].cpu().numpy()[indices],
                                           p["scores"].cpu().numpy()[indices, None]], axis=1).astype(np.float64))
        return out

    return predict


def first_17(config) -> list[int]:
    return list(range(N))


def sapiens2_coco(config) -> list[int]:
    """The COCO points among Sapiens2's 308 (unnamed in its config): ids of ``keypoints308.py`` in the
    sapiens2 repository, where the wrists come after the right hand (41) and the left hand (62)."""
    if config.num_labels != 308:
        raise ValueError(f"expected 308 Sapiens2 keypoints, the model has {config.num_labels}")
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def rtmlib_model(spec: str, name: str, options: set[str], device: str) -> Predictor:
    import cv2
    from rtmlib import Body, Wholebody

    from psifx.video.pose.onnx_optimization import optimize_rtmlib_models, preload_cuda_libraries

    check_options(spec, options, {"fp16"})
    kind, _, mode = name.partition("-")
    if device != "cpu":
        preload_cuda_libraries()
    if kind == "rtmo":
        model = Body(pose="rtmo", mode=mode, backend="onnxruntime", device=device)
        sessions = {"pose": model.pose_model}
    elif kind in ("body", "wholebody"):
        model = (Body if kind == "body" else Wholebody)(mode=mode, backend="onnxruntime", device=device)
        sessions = {"det": model.det_model, "pose": model.pose_model}
    else:
        raise ValueError(f"{spec}: rtmlib name must be rtmo-<mode>, body-<mode> or wholebody-<mode>")
    work_dir = tempfile.TemporaryDirectory()
    optimize_rtmlib_models(sessions, "fp16" if "fp16" in options else "fp32", device, work_dir.name)

    def predict(image: np.ndarray) -> list[np.ndarray]:
        keypoints, scores = model(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        predict.work_dir = work_dir  # keeps the converted models alive with the predictor
        return [np.concatenate([np.asarray(k, float)[:N], np.asarray(s, float)[:N, None]], axis=1)
                for k, s in zip(keypoints, scores)]

    return predict


def rfdetr_model(spec: str, name: str, options: set[str], device: str) -> Predictor:
    from rfdetr import RFDETRKeypointPreview

    check_options(spec, options, {"optimize", "fp16"})
    model = RFDETRKeypointPreview(device=device)
    if "optimize" in options or "fp16" in options:
        # RF-DETR's own optimization: a torch.jit trace of the model, cast to fp16 if asked.
        model.inference(compile=True, dtype=torch.float16 if "fp16" in options else torch.float32)

    def predict(image: np.ndarray) -> list[np.ndarray]:
        # Low threshold: the person score is kept apart from the keypoint scores, which are measured.
        people = model.predict(image, threshold=0.3)
        return [np.concatenate([xy.astype(float), conf.astype(float)[:, None]], axis=1)
                for xy, conf in zip(people.xy, people.keypoint_confidence)]

    return predict


def yolo_model(spec: str, name: str, options: set[str], device: str) -> Predictor:
    from ultralytics import YOLO

    check_options(spec, options, {"fp16"})
    model = YOLO(f"{name}.pt")
    # ``half`` is deprecated in Ultralytics 8.4: only passed when asked.
    extra = {"half": True} if "fp16" in options else {}

    def predict(image: np.ndarray) -> list[np.ndarray]:
        result = model.predict(image[..., ::-1], device=device, conf=0.3, verbose=False, **extra)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return []
        data = result.keypoints.data.cpu().numpy().astype(float)  # (people, 17, 3)
        return list(data)

    return predict


def load(spec: str, device: str) -> Predictor:
    family, name, options = parse(spec)
    if family == "vitpose":
        return top_down(spec, name, options, device, first_17)
    if family == "sapiens2":
        return top_down(spec, name, options, device, sapiens2_coco, batch=1)
    if family == "rtmlib":
        return rtmlib_model(spec, name, options, device)
    if family == "rfdetr":
        return rfdetr_model(spec, name, options, device)
    if family == "yolo":
        return yolo_model(spec, name, options, device)
    raise ValueError(f"unknown model family {family!r} in {spec!r}")
