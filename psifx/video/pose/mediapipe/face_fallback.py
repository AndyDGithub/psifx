"""Face landmarks from a crop placed by the body pose, for frames where the Holistic Landmarker misses the face.

The legacy MediaPipe Holistic placed the face region from the body pose and ran the face landmark
model there directly. The Tasks Holistic Landmarker runs a face detector first, which misses many
medium-size faces (AVA clip: face in 43 % of the frames instead of 80 %). Here the head
region comes from the pose points 0-10 (nose, eyes, ears, mouth), is resized to a fixed square and
given to the Face Landmarker in video mode, which tracks the face from one frame to the next.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Side of the square crop given to the Face Landmarker, in pixels.
CROP_SIZE = 256
# Crop side over the spread of the head points: the face then fills about a third of the crop,
# which suits the face detector (measured: x1.6 finds 32 % of the faces, x3 57 %, x4 61 %).
HEAD_SCALE = 3.0
HEAD_POINTS = slice(0, 11)


class FaceCropFallback:
    """
    Face Landmarker on a pose-guided head crop.

    :param model_path: Local path of the Face Landmarker ``.task`` bundle.
    :param detection_confidence: Minimum face detection score; low, since the crop already holds a head.
    :param presence_confidence: Minimum face presence and tracking score.
    """

    def __init__(self, model_path: Path, detection_confidence: float = 0.1, presence_confidence: float = 0.5):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=presence_confidence,
            min_tracking_confidence=presence_confidence,
        )
        self.model = FaceLandmarker.create_from_options(options)

    @staticmethod
    def head_box(pose_landmarks: Sequence, width: int, height: int) -> tuple:
        """Center (x, y) and side of the square head box, in pixels."""
        points = np.array([[p.x * width, p.y * height] for p in pose_landmarks[HEAD_POINTS]])
        side = max(np.ptp(points[:, 0]), np.ptp(points[:, 1])) * HEAD_SCALE
        return points.mean(axis=0), max(side, 16.0)

    def detect(self, image: np.ndarray, pose_landmarks: Sequence, timestamp: int) -> Optional[List]:
        """
        Face landmarks of the person, in normalized image coordinates like the Holistic output.

        :param image: RGB frame.
        :param pose_landmarks: Holistic pose landmarks of the frame (normalized coordinates).
        :param timestamp: Frame timestamp in milliseconds, strictly increasing.
        :return: 478 landmarks with ``x``/``y``/``z`` attributes, or None if no face was found.
        """
        height, width = image.shape[:2]
        center, side = self.head_box(pose_landmarks, width, height)
        scale = CROP_SIZE / side
        # Box -> crop: scale around the box center, the parts outside the frame are black.
        affine = np.array([[scale, 0, CROP_SIZE / 2 - scale * center[0]],
                           [0, scale, CROP_SIZE / 2 - scale * center[1]]])
        crop = cv2.warpAffine(image, affine, (CROP_SIZE, CROP_SIZE))
        result = self.model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), timestamp)
        if not result.face_landmarks:
            return None
        # Crop -> frame, back to normalized coordinates.
        for p in result.face_landmarks[0]:
            p.x = ((p.x * CROP_SIZE - CROP_SIZE / 2) / scale + center[0]) / width
            p.y = ((p.y * CROP_SIZE - CROP_SIZE / 2) / scale + center[1]) / height
        return result.face_landmarks[0]

    def close(self):
        self.model.close()
