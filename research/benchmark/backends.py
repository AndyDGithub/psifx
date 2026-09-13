"""One adapter per candidate backend, each implementing common.PoseBackend.

Design choice, documented here rather than hidden in code: MediaPipe and
RTMPose (via rtmlib's `Body`) both ship their own person detector, so they
run standalone. ViTPose (via transformers) is top-down only -- it needs a
person bounding box as input, it does not detect people itself. Rather than
adding a fourth dependency (a separate object detector) just for this
benchmark, ViTPose reuses whichever boxes are passed to `predict_with_boxes`
by the caller:
  - on the COCO subset, the ground-truth bounding boxes (a standard
    "oracle-box" top-down evaluation practice, isolating pose accuracy from
    detection accuracy);
  - on Video.mp4 (no ground truth), the boxes produced by the RTMPose
    detector for that same frame (documented simplification in
    03_resultats.md, not a property of ViTPose itself).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from common import (
    COCO_KEYPOINT_NAMES,
    FramePrediction,
    MEDIAPIPE_TO_COCO17,
    NUM_COCO_KEYPOINTS,
    PersonPose,
    PoseBackend,
    empty_person_pose,
)


class MediaPipeBackend(PoseBackend):
    name = "mediapipe"

    def __init__(self, static_image_mode: bool = False):
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        import cv2

        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        results = self._pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        dt = time.perf_counter() - t0

        if not results.pose_landmarks:
            return FramePrediction(people=[], inference_seconds=dt)

        kp = np.full((NUM_COCO_KEYPOINTS, 2), np.nan)
        sc = np.full((NUM_COCO_KEYPOINTS,), np.nan)
        landmarks = results.pose_landmarks.landmark
        for name, mp_idx in MEDIAPIPE_TO_COCO17.items():
            i = COCO_KEYPOINT_NAMES.index(name)
            lm = landmarks[mp_idx]
            kp[i] = [lm.x * w, lm.y * h]
            sc[i] = lm.visibility
        person = PersonPose(keypoints=kp, scores=sc)
        return FramePrediction(people=[person], inference_seconds=dt)

    def close(self) -> None:
        self._pose.close()


class RTMPoseBackend(PoseBackend):
    name = "rtmpose"

    def __init__(self, device: str = "cuda", mode: str = "balanced"):
        from rtmlib import Body

        # 'performance' (the heaviest preset: YOLOX-x + RTMPose-x) turned
        # out slower and more GPU-hungry than published RTMPose numbers in
        # the first benchmark pass (18.5 FPS, 792 MB) -- 'balanced' is
        # rtmlib's default and the one closer to the "real-time" reputation
        # cited in 01_*/02_*. 'lightweight' is the third option if an even
        # faster/smaller preset is wanted later.
        self._body = Body(
            mode=mode,
            backend="onnxruntime",
            device=device,
        )

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        # Body.__call__ doesn't expose the intermediate detection boxes, and
        # we need them to drive ViTPose on video (see module docstring), so
        # the two-stage pipeline is replicated manually here instead of
        # calling self._body(image_bgr) directly.
        t0 = time.perf_counter()
        bboxes = self._body.det_model(image_bgr)
        keypoints, scores = self._body.pose_model(image_bgr, bboxes=bboxes)
        dt = time.perf_counter() - t0
        self._last_boxes = np.asarray(bboxes)[:, :4] if len(bboxes) else np.empty((0, 4))

        people: List[PersonPose] = []
        for kp_person, sc_person in zip(keypoints, scores):
            # rtmlib's Body (coco17 preset) already emits COCO-17 order.
            person = PersonPose(
                keypoints=np.asarray(kp_person, dtype=np.float64)[:NUM_COCO_KEYPOINTS],
                scores=np.asarray(sc_person, dtype=np.float64)[:NUM_COCO_KEYPOINTS],
            )
            people.append(person)
        return FramePrediction(people=people, inference_seconds=dt)

    def last_boxes(self) -> Optional[np.ndarray]:
        """Person boxes rtmlib's internal detector found on the last
        `predict` call, in (N, 4) xyxy -- reused to drive ViTPoseBackend on
        Video.mp4 (see module docstring)."""
        return getattr(self, "_last_boxes", None)


class RTMOBackend(PoseBackend):
    name = "rtmo"

    def __init__(self, device: str = "cuda", mode: str = "balanced"):
        from rtmlib import Body

        # pose='rtmo' switches Body into its one-stage branch (rtmlib source,
        # rtmlib/tools/solution/body.py): only RTMO itself is instantiated,
        # no separate YOLOX detector -- this is the whole point of testing
        # RTMO here (see 04_modeles_optimisation.md 1.1: it should remove
        # ViTPose's dependency on an external detector). 'balanced' (rtmo-m)
        # matches RTMPoseBackend's mode choice for a fair comparison.
        self._body = Body(pose="rtmo", mode=mode, backend="onnxruntime", device=device)

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        t0 = time.perf_counter()
        keypoints, scores = self._body(image_bgr)
        dt = time.perf_counter() - t0

        people: List[PersonPose] = []
        for kp_person, sc_person in zip(keypoints, scores):
            # Same body7 dataset lineage as RTMPoseBackend's preset, already
            # verified there to emit COCO-17 order directly.
            person = PersonPose(
                keypoints=np.asarray(kp_person, dtype=np.float64)[:NUM_COCO_KEYPOINTS],
                scores=np.asarray(sc_person, dtype=np.float64)[:NUM_COCO_KEYPOINTS],
            )
            people.append(person)
        return FramePrediction(people=people, inference_seconds=dt)


class DETRPoseBackend(PoseBackend):
    """Runs DETRPose via subprocess in a separate conda env (`detrpose-env`),
    not because of a Python version mismatch (DETRPose supports 3.11, same as
    psifx-env) but because its requirements.txt pins numpy==1.23.5 with
    unpinned torch/torchvision -- installing that into psifx-env risks
    breaking mediapipe/rtmlib/opencv's numpy expectations. Same "external
    process, bridged by a file" pattern as Sapiens2Backend.

    DETRPose returns one score per person instance, not per keypoint (see
    detrpose_infer.py) -- every keypoint of a detected person is reported
    with score 1.0, matching PersonPose's documented convention for backends
    without per-keypoint confidence. No bounding box is returned either;
    left as None (metrics.py falls back to a keypoint-extent estimate).

    Only the M-size checkpoint is used here (matches RTMPose/RTMO's
    'balanced' choice) -- unlike Sapiens2, a size sweep wasn't requested for
    DETRPose, so S/L aren't downloaded.
    """

    name = "detrpose"

    def __init__(
        self,
        detrpose_root: str = "/home/psifx/detrpose",
        checkpoint_path: str = "/home/psifx/detrpose/detrpose_hgnetv2_m.pth",
        config_rel: str = "configs/detrpose/detrpose_hgnetv2_m.py",
        python_bin: str = "/home/psifx/conda/envs/detrpose-env/bin/python",
        device: str = "cuda",
    ):
        self._detrpose_root = detrpose_root
        self._checkpoint_path = checkpoint_path
        self._config_rel = config_rel
        self._python_bin = python_bin
        self._device = device
        self._script = str(Path(__file__).resolve().parent / "detrpose_infer.py")

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        return self.predict_all([image_bgr])[0]

    def predict_all(self, images_bgr: List[np.ndarray]) -> List[FramePrediction]:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            images_dir = Path(tmp) / "images"
            images_dir.mkdir()
            names = [f"{i:06d}.jpg" for i in range(len(images_bgr))]
            for name, img in zip(names, images_bgr):
                cv2_imwrite(str(images_dir / name), img)

            output_json = Path(tmp) / "results.json"
            cmd = [
                self._python_bin,
                self._script,
                "--detrpose-root", self._detrpose_root,
                "--config", self._config_rel,
                "--checkpoint", self._checkpoint_path,
                "--images-dir", str(images_dir),
                "--output", str(output_json),
                "--device", self._device,
            ]
            import os as _os

            # Same reasoning as Sapiens2Backend: don't leak psifx-env's
            # cuDNN/cuBLAS LD_LIBRARY_PATH into detrpose-env's own torch build.
            env = dict(_os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            env.pop("LD_LIBRARY_PATH", None)
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"detrpose_infer.py failed (exit {result.returncode}):\n{result.stderr[-4000:]}"
                )

            import json

            payload = json.loads(output_json.read_text())

        by_name = {r["image_name"]: r for r in payload["frames"]}
        out: List[FramePrediction] = []
        for name in names:
            record = by_name.get(name)
            if record is None or not record["instances"]:
                out.append(FramePrediction(people=[], inference_seconds=0.0))
                continue
            people = []
            for inst in record["instances"]:
                kp = np.asarray(inst["keypoints"], dtype=np.float64)[:NUM_COCO_KEYPOINTS]
                sc = np.asarray(inst["keypoint_scores"], dtype=np.float64)[:NUM_COCO_KEYPOINTS]
                people.append(PersonPose(keypoints=kp, scores=sc))
            out.append(FramePrediction(people=people, inference_seconds=record["inference_seconds"]))
        return out


class ViTPoseBackend(PoseBackend):
    name = "vitpose"

    def __init__(self, model_id: str = "usyd-community/vitpose-base-simple", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device)
        self._model.eval()
        self._torch = torch

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        # Standalone call (no external boxes): use the full frame as a
        # single person box. Only meant for a quick sanity check -- real
        # runs should go through predict_with_boxes (see module docstring).
        h, w = image_bgr.shape[:2]
        return self.predict_with_boxes(image_bgr, np.array([[0, 0, w, h]], dtype=np.float64))

    def predict_with_boxes(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> FramePrediction:
        import cv2

        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return FramePrediction(people=[], inference_seconds=0.0)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]

        t0 = time.perf_counter()
        inputs = self._processor(image_rgb, boxes=[boxes_xywh], return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
        dt = time.perf_counter() - t0

        people: List[PersonPose] = []
        for person_result in results:
            kp = np.asarray(person_result["keypoints"], dtype=np.float64)[:NUM_COCO_KEYPOINTS, :2]
            sc = np.asarray(person_result["scores"], dtype=np.float64)[:NUM_COCO_KEYPOINTS]
            if kp.shape[0] < NUM_COCO_KEYPOINTS:
                pad = NUM_COCO_KEYPOINTS - kp.shape[0]
                kp = np.vstack([kp, np.full((pad, 2), np.nan)])
                sc = np.concatenate([sc, np.full((pad,), np.nan)])
            people.append(PersonPose(keypoints=kp, scores=sc))
        return FramePrediction(people=people, inference_seconds=dt)


class Sapiens2Backend(PoseBackend):
    """Runs Sapiens2 (308-keypoint pose model) via subprocess in a separate
    Python 3.12 conda env (`sapiens2-env`), since sapiens2/pyproject.toml
    requires Python >=3.12 -- incompatible with psifx-env (3.11), where the
    rest of this benchmark (and psifx itself) runs. Same "external process,
    bridged by a file" pattern psifx already uses for OpenFace.

    Only the first 17 of Sapiens2's 308 keypoints are used here: verified in
    sapiens/pose/configs/_base_/keypoints308.py that indices 0-16 are the
    standard COCO-17 set in COCO_KEYPOINT_NAMES order (COCO-WholeBody
    convention), so no remapping is needed, just truncation.

    Runs on a *list* of images in one subprocess call (predict_all), not one
    subprocess per image -- interpreter/model-loading startup cost would
    otherwise dominate the timing and make the FPS number meaningless.
    predict() (single image) is provided for interface completeness but
    pays that startup cost every call -- avoid it on the hot path.
    """

    name = "sapiens2"

    def __init__(
        self,
        sapiens_root: str = "/home/psifx/sapiens2",
        checkpoint_root: str = "/home/psifx/sapiens2_host",
        model_size: str = "0.4b",
        python_bin: str = "/home/psifx/conda/envs/sapiens2-env/bin/python",
        device: str = "cuda",
    ):
        self._sapiens_root = sapiens_root
        self._checkpoint_root = checkpoint_root
        self._model_size = model_size
        self._python_bin = python_bin
        self._device = device
        self._script = str(Path(__file__).resolve().parent / "sapiens2_infer.py")

    def predict(self, image_bgr: np.ndarray) -> FramePrediction:
        return self.predict_all([image_bgr])[0]

    def predict_all(self, images_bgr: List[np.ndarray]) -> List[FramePrediction]:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            images_dir = Path(tmp) / "images"
            images_dir.mkdir()
            names = [f"{i:06d}.jpg" for i in range(len(images_bgr))]
            for name, img in zip(names, images_bgr):
                cv2_imwrite(str(images_dir / name), img)

            output_json = Path(tmp) / "results.json"
            dataset = "shutterstock_goliath_3po"
            model_name = f"sapiens2_{self._model_size}"
            config_rel = (
                f"configs/keypoints308/{dataset}/"
                f"{model_name}_keypoints308_{dataset}-1024x768.py"
            )
            cmd = [
                self._python_bin,
                self._script,
                "--sapiens-root", self._sapiens_root,
                "--det-checkpoint", f"{self._checkpoint_root}/detector/detr-resnet-101-dc5",
                "--config", config_rel,
                "--checkpoint", f"{self._checkpoint_root}/pose/{model_name}_pose.safetensors",
                "--images-dir", str(images_dir),
                "--output", str(output_json),
                "--device", self._device,
            ]
            import os as _os

            # Strip LD_LIBRARY_PATH rather than inherit it: run_accuracy.py /
            # run_video_benchmark.py export it (pointing at psifx-env's
            # cuDNN 9.10/CUDA12 libs, needed for onnxruntime's CUDA provider
            # in THIS process -- see README.md) before launching Python, so
            # it leaks into every subprocess by default. sapiens2-env has
            # its own torch build against CUDA 13 with its own bundled
            # cuDNN 9.24; picking up the inherited CUDA-12 libs instead
            # produces a version-mismatched cuDNN handle (confirmed:
            # "CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED"). Let sapiens2-env's
            # own binaries resolve via their own RPATH/site-packages.
            env = dict(_os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            env.pop("LD_LIBRARY_PATH", None)
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"sapiens2_infer.py failed (exit {result.returncode}):\n{result.stderr[-4000:]}"
                )

            import json

            payload = json.loads(output_json.read_text())

        by_name = {r["image_name"]: r for r in payload["frames"]}
        out: List[FramePrediction] = []
        for name in names:
            record = by_name.get(name)
            if record is None or not record["instances"]:
                out.append(FramePrediction(people=[], inference_seconds=0.0))
                continue
            people = []
            for inst in record["instances"]:
                kp = np.asarray(inst["keypoints"], dtype=np.float64)[:NUM_COCO_KEYPOINTS]
                sc = np.asarray(inst["keypoint_scores"], dtype=np.float64)[:NUM_COCO_KEYPOINTS]
                people.append(PersonPose(keypoints=kp, scores=sc))
            out.append(FramePrediction(people=people, inference_seconds=record["inference_seconds"]))
        return out


def cv2_imwrite(path: str, image_bgr: np.ndarray) -> None:
    import cv2

    cv2.imwrite(path, image_bgr)


def build_backend(name: str, device: str = "cuda") -> PoseBackend:
    if name == "mediapipe":
        return MediaPipeBackend()
    if name == "rtmpose":
        return RTMPoseBackend(device=device)
    if name == "vitpose":
        return ViTPoseBackend(device=device)
    if name == "rtmo":
        return RTMOBackend(device=device)
    if name == "detrpose":
        return DETRPoseBackend(device=device)
    if name == "sapiens2":
        return Sapiens2Backend(device=device)
    if name.startswith("sapiens2-"):
        # e.g. "sapiens2-0.8b" -> model_size="0.8b" (checkpoint must already
        # be downloaded under {checkpoint_root}/pose/sapiens2_{size}_pose.safetensors,
        # see README.md).
        model_size = name.removeprefix("sapiens2-")
        return Sapiens2Backend(device=device, model_size=model_size)
    raise ValueError(f"Unknown backend: {name}")
