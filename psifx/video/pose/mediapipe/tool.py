"""MediaPipe pose estimation tool."""

from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import urllib.request
from tqdm import tqdm

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode

from psifx.video.pose.tool import PoseEstimationTool
from psifx.video.pose.mediapipe import skeleton
from psifx.video.pose.mediapipe.face_fallback import FaceCropFallback
from psifx.io import tar, video
from psifx.utils.constants import (
    MEDIAPIPE_FACE_PATH,
    MEDIAPIPE_FACE_SHA256,
    MEDIAPIPE_HOLISTIC_PATH,
    MEDIAPIPE_HOLISTIC_SHA256,
)

# Checksums of the default bundles; a custom URL is trusted as is.
KNOWN_SHA256 = {MEDIAPIPE_HOLISTIC_PATH: MEDIAPIPE_HOLISTIC_SHA256, MEDIAPIPE_FACE_PATH: MEDIAPIPE_FACE_SHA256}


def bundle_path(source: str = MEDIAPIPE_HOLISTIC_PATH) -> Path:
    """
    Local path of a MediaPipe ``.task`` bundle, downloaded to ``~/.cache/psifx/mediapipe`` on first use.

    :param source: URL or local path of the bundle.
    :return: Path to the bundle.
    """
    if not source.startswith(("http://", "https://")):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {path}")
        return path

    path = Path.home() / ".cache" / "psifx" / "mediapipe" / source.rsplit("/", 1)[-1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        print(f"Downloading the MediaPipe model to {path}")
        urllib.request.urlretrieve(source, partial)
        if source in KNOWN_SHA256:
            digest = hashlib.sha256(partial.read_bytes()).hexdigest()
            if digest != KNOWN_SHA256[source]:
                partial.unlink()
                raise RuntimeError(f"Unexpected checksum for {source}: {digest}")
        partial.rename(path)
    return path


class MediaPipePoseEstimationTool(PoseEstimationTool):
    """
    MediaPipe pose estimation tool.

    :param model_path: URL or local path of the Holistic Landmarker ``.task`` bundle.
    :param face_fallback: Whether to look for the face on a pose-guided crop when the Holistic Landmarker misses it.
        It finds more faces (AVA clip: 74 % of the frames instead of 43 %, legacy API 80 %), but these extra faces, often small or turned,
        give a noisier mouth signal, so it is off by default.
    :param face_model_path: URL or local path of the Face Landmarker ``.task`` bundle used by the fallback.
    :param min_detection_confidence: Minimum score of the body and face detections that start a track.
    :param min_presence_confidence: Minimum score of the body, face and hand landmarks to keep them.
    :param model_complexity: Ignored, the MediaPipe Tasks API has a single Holistic model (kept for compatibility).
    :param smooth: Ignored, the video mode of the Tasks API always tracks and smooths (kept for compatibility).
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        model_path: str = MEDIAPIPE_HOLISTIC_PATH,
        face_fallback: bool = False,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        face_model_path: str = MEDIAPIPE_FACE_PATH,
        model_complexity: Optional[int] = None,
        smooth: Optional[bool] = None,
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        if self.device != "cpu":
            print("Only CPU support is currently available for MediaPipe pose estimation tool.")
        if model_complexity is not None or smooth is not None:
            print("model_complexity and smooth are ignored since the move to the MediaPipe Tasks API.")
        self.model_path = model_path
        self.face_fallback = face_fallback
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.face_model_path = face_model_path

    def process_part(
        self,
        landmarks,
        size: Tuple[int, int],
        n_points: int,
    ) -> List[float]:
        """
        Processes MediaPipe output into a simple flattened list of coordinates.

        :param landmarks: MediaPipe landmarks, empty when the part was not detected.
        :param size: Image resolution.
        :param n_points: Expected number of points.
        :return: Processed keypoint coordinates.
        """
        h, w = size
        if landmarks:
            # Only pose landmarks carry a visibility; face and hands get 0 as with the legacy API.
            landmarks = [[p.x, p.y, p.visibility or 0.0] for p in landmarks]
            points = np.array(landmarks, dtype=np.float32)
            points[:, 0] *= w - 1
            points[:, 1] *= h - 1
        else:
            points = np.zeros((n_points, 3), dtype=np.float32)
        return points.flatten().tolist()

    def process_world(
        self,
        landmarks,
        n_points: int,
    ) -> List[float]:
        """
        Processes MediaPipe world landmarks into a flattened list of 3D coordinates.

        :param landmarks: MediaPipe world landmarks, in meters with the origin between the hips.
        :param n_points: Expected number of points.
        :return: Flattened ``[x, y, z, visibility]`` per point, zeros if nobody was detected.
        """
        if landmarks:
            points = np.array([[p.x, p.y, p.z, p.visibility or 0.0] for p in landmarks], dtype=np.float32)
        else:
            points = np.zeros((n_points, 4), dtype=np.float32)
        return points.flatten().tolist()

    def process_pose(
        self,
        results,
        size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Process all the parts estimated by MediaPipe, e.g. body, face, hands.

        :param results: MediaPipe output.
        :param size: Image resolution.
        :return: Processed keypoints for every part.
        """
        return {
            "pose_keypoints_2d": self.process_part(
                landmarks=results.pose_landmarks,
                size=size,
                n_points=skeleton.N_POSE_LANDMARKS,
            ),
            "face_keypoints_2d": self.process_part(
                landmarks=results.face_landmarks,
                size=size,
                n_points=skeleton.N_FACE_LANDMARKS,
            ),
            "hand_left_keypoints_2d": self.process_part(
                landmarks=results.left_hand_landmarks,
                size=size,
                n_points=skeleton.N_LEFT_HAND_LANDMARKS,
            ),
            "hand_right_keypoints_2d": self.process_part(
                landmarks=results.right_hand_landmarks,
                size=size,
                n_points=skeleton.N_RIGHT_HAND_LANDMARKS,
            ),
            "pose_keypoints_3d": self.process_world(
                landmarks=results.pose_world_landmarks,
                n_points=skeleton.N_POSE_LANDMARKS,
            ),
        }

    def inference(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
    ):
        """
        Runs MediaPipe pose estimation model on a video.

        :param video_path: Path to the video file.
        :param poses_path: Path to the pose archive.
        :return:
        """
        video_path = Path(video_path)
        poses_path = Path(poses_path)

        if self.verbose:
            print(f"video   =   {video_path}")
            print(f"poses   =   {poses_path}")

        tar.TarWriter.check(path=poses_path, overwrite=self.overwrite)

        poses = {
            "edges": {
                "pose_keypoints_2d": skeleton.POSE_EDGES,
                "face_keypoints_2d": skeleton.FACE_EDGES,
                "hand_left_keypoints_2d": skeleton.LEFT_HAND_EDGES,
                "hand_right_keypoints_2d": skeleton.RIGHT_HAND_EDGES,
                "pose_keypoints_3d": skeleton.POSE_EDGES,
            }
        }

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(bundle_path(self.model_path))),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_face_detection_confidence=self.min_detection_confidence,
            min_pose_landmarks_confidence=self.min_presence_confidence,
            min_face_landmarks_confidence=self.min_presence_confidence,
            min_hand_landmarks_confidence=self.min_presence_confidence,
        )
        # A new landmarker per video: the video mode keeps a tracking state across frames.
        face_model = FaceCropFallback(bundle_path(self.face_model_path)) if self.face_fallback else None
        recovered = 0
        with (
            HolisticLandmarker.create_from_options(options) as model,
            video.VideoReader(path=video_path) as video_reader,
        ):
            frame_ms = 1000 / float(Fraction(video_reader.frame_rate))
            timestamp = -1
            for i, image in enumerate(
                tqdm(
                    video_reader,
                    desc="Processing",
                    disable=not self.verbose,
                )
            ):
                h, w, _ = image.shape
                # The video mode needs strictly increasing timestamps in milliseconds.
                timestamp = max(round(i * frame_ms), timestamp + 1)
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
                results = model.detect_for_video(frame, timestamp)
                if face_model is not None and results.pose_landmarks and not results.face_landmarks:
                    face = face_model.detect(frame.numpy_view(), results.pose_landmarks, timestamp)
                    if face is not None:
                        results.face_landmarks = face
                        recovered += 1
                poses[f"{i: 015d}"] = self.process_pose(
                    results=results,
                    size=(h, w),
                )

        if face_model is not None:
            face_model.close()
            if self.verbose:
                print(f"Face found on the pose-guided crop in {recovered} frames.")

        poses = {
            f"{k}.json": json.dumps(v)
            for k, v in tqdm(
                poses.items(),
                desc="Encoding",
                disable=not self.verbose,
            )
        }
        tar.TarWriter.write(
            dictionary=poses,
            path=poses_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )
