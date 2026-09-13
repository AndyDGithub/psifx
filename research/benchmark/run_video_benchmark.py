"""Video pass: run every backend on example/data/Video.mp4, measure FPS,
GPU memory and the temporal-stability jerk proxy (metrics.normalized_jerk_series).

GPU memory is read via `nvidia-smi` (framework-agnostic: MediaPipe and
RTMPose don't run on PyTorch, so torch.cuda memory counters would silently
under-report for them) rather than torch.cuda.max_memory_allocated().

ViTPose has no built-in detector: on this video it reuses the person boxes
RTMPose's internal detector found on the same frame (see backends.py
docstring) -- so `--backends` must include rtmpose whenever it includes
vitpose, and rtmpose is run first.

Usage: python run_video_benchmark.py [--backends mediapipe,rtmpose,vitpose]
                                      [--max-frames 300] [--device cuda]
Writes: results/video_summary.csv, results/video_tracks/<backend>.npz
"""

from __future__ import annotations

# Import order matters here: onnxruntime-gpu (used by RTMPose) and torch
# (used by ViTPose) each bundle their own copy of the CUDA runtime. Whichever
# loads its libcudart/libcublasLt into the process first "wins" that soname
# slot; the other one then reuses it instead of loading its own copy. torch's
# build expects a newer cudart symbol (cudaGetDriverEntryPointByVersion) than
# onnxruntime-gpu==1.20.2 bundles, so if onnxruntime loads CUDA first, torch's
# later `import torch` crashes with an undefined-symbol ImportError. Importing
# torch first avoids this (onnxruntime is the more version-tolerant of the
# two and works fine reusing torch's already-loaded runtime).
import torch  # noqa: F401  (import order fix, see comment above -- not used directly here)

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backends import RTMPoseBackend, ViTPoseBackend, build_backend
from common import COCO_KEYPOINT_NAMES, NUM_COCO_KEYPOINTS
from metrics import normalized_jerk_series, person_scale

HERE = Path(__file__).resolve().parent
VIDEO_PATH = HERE.parent.parent / "example" / "data" / "Video.mp4"
RESULTS_DIR = HERE / "results"
TRACKS_DIR = RESULTS_DIR / "video_tracks"


def gpu_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return float(out.decode().splitlines()[0].strip())
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="mediapipe,rtmpose,vitpose")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not VIDEO_PATH.exists():
        raise SystemExit(f"{VIDEO_PATH} not found")

    backend_names = args.backends.split(",")
    if "vitpose" in backend_names and "rtmpose" not in backend_names:
        raise SystemExit("vitpose needs rtmpose's detections on video -- include both")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames at {fps:.1f} fps")

    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    rtmpose_boxes_per_frame: list[np.ndarray | None] = [None] * len(frames)
    summary_rows = []

    for backend_name in backend_names:
        print(f"=== {backend_name} ===")
        backend = build_backend(backend_name, device=args.device)
        mem_before = gpu_memory_mb()

        # (frame, person_slot=0, keypoint, xy) -- single-track (first
        # detected person per frame): sufficient for a stability/speed proxy,
        # not a full multi-person re-identification benchmark.
        track = np.full((len(frames), NUM_COCO_KEYPOINTS, 2), np.nan)
        inference_seconds = []

        if hasattr(backend, "predict_all"):
            # Subprocess-based backends (Sapiens2, DETRPose) -- one batched
            # subprocess call for the whole clip instead of per-frame, same
            # reasoning as in run_accuracy.py.
            preds = backend.predict_all(frames)
            for i, pred in enumerate(preds):
                inference_seconds.append(pred.inference_seconds)
                if pred.people:
                    track[i] = pred.people[0].keypoints
        else:
            for i, frame in enumerate(frames):
                if isinstance(backend, ViTPoseBackend):
                    boxes = rtmpose_boxes_per_frame[i]
                    if boxes is None or len(boxes) == 0:
                        continue
                    pred = backend.predict_with_boxes(frame, boxes[:1])
                else:
                    pred = backend.predict(frame)
                    if isinstance(backend, RTMPoseBackend):
                        rtmpose_boxes_per_frame[i] = backend.last_boxes()

                inference_seconds.append(pred.inference_seconds)
                if pred.people:
                    track[i] = pred.people[0].keypoints

        mem_after = gpu_memory_mb()
        backend.close()

        np.savez(TRACKS_DIR / f"{backend_name}.npz", track=track, fps=fps)

        scales = [person_scale_at(track, t) for t in range(len(frames))]
        scale = float(np.nanmedian(scales)) if scales else float("nan")
        jerks = [
            normalized_jerk_series(track[:, k, :], fps=fps, scale=scale)
            for k in range(NUM_COCO_KEYPOINTS)
        ]
        detection_rate = float(np.mean(~np.isnan(track[:, COCO_KEYPOINT_NAMES.index("nose"), 0])))

        summary_rows.append(
            dict(
                backend=backend_name,
                fps_achieved=1.0 / np.mean(inference_seconds) if inference_seconds else float("nan"),
                mean_inference_ms=1000 * float(np.mean(inference_seconds)) if inference_seconds else float("nan"),
                gpu_mem_delta_mb=mem_after - mem_before if not (np.isnan(mem_before) or np.isnan(mem_after)) else float("nan"),
                mean_normalized_jerk=float(np.nanmean(jerks)),
                detection_rate=detection_rate,
                n_frames=len(frames),
            )
        )

    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "video_summary.csv", index=False)
    print(df)


def person_scale_at(track: np.ndarray, t: int):
    from common import PersonPose

    kp = track[t]
    sc = np.ones(NUM_COCO_KEYPOINTS)
    return person_scale(PersonPose(keypoints=kp, scores=sc))


if __name__ == "__main__":
    main()
