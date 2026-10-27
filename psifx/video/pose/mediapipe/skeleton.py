"""Mediapipe's skeleton graph.'"""

from mediapipe.tasks.python.vision import (
    FaceLandmarksConnections,
    HandLandmarksConnections,
    PoseLandmarksConnections,
)


def _edges(connections) -> tuple:
    return tuple((c.start, c.end) for c in connections)


N_POSE_LANDMARKS = 33
# 468 mesh points plus 10 iris points.
N_FACE_LANDMARKS = 478
N_LEFT_HAND_LANDMARKS = 21
N_RIGHT_HAND_LANDMARKS = 21

POSE_EDGES = _edges(PoseLandmarksConnections.POSE_LANDMARKS)
FACE_EDGES = _edges(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
LEFT_HAND_EDGES = _edges(HandLandmarksConnections.HAND_CONNECTIONS)
RIGHT_HAND_EDGES = _edges(HandLandmarksConnections.HAND_CONNECTIONS)
