"""Shared runtime constants."""

import os


# Allow overriding the default SAM3 model path without editing source files.
SAM3_PATH = os.environ.get("SAM3_PATH", "facebook/sam3")

# Image encoder that re-identifies people across shot changes during SAM3 tracking.
SAM3_REID_PATH = os.environ.get("SAM3_REID_PATH", "facebook/dinov2-small")

# MediaPipe Holistic Landmarker bundle (Tasks API), downloaded on first use. A local path also works.
MEDIAPIPE_HOLISTIC_PATH = os.environ.get(
    "MEDIAPIPE_HOLISTIC_PATH",
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/"
    "holistic_landmarker.task",
)
MEDIAPIPE_HOLISTIC_SHA256 = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"

# MediaPipe Face Landmarker bundle, run on a crop placed from the body pose when the Holistic
# Landmarker misses the face (its face detector misses many medium-size faces).
MEDIAPIPE_FACE_PATH = os.environ.get(
    "MEDIAPIPE_FACE_PATH",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
)
MEDIAPIPE_FACE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
