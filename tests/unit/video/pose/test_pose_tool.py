"""Unit tests for the pose estimation base tool and the MediaPipe output formatting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from psifx.io.tar import TarWriter
from psifx.video.pose.tool import PoseEstimationTool


def write_poses(path: Path, frames: dict, edges=None) -> Path:
    files = {f"{k}.json": json.dumps(v) for k, v in frames.items()}
    if edges is not None:
        files["edges.json"] = json.dumps(edges)
    TarWriter.write(files, path, overwrite=True, verbose=False)
    return path


@pytest.fixture
def tool() -> PoseEstimationTool:
    return PoseEstimationTool(device="cpu", overwrite=True, verbose=False)


@pytest.mark.unit
def test_inference_is_abstract(tool, tmp_path):
    with pytest.raises(NotImplementedError):
        tool.inference(tmp_path / "v.mp4", tmp_path / "p.tar.gz")


@pytest.mark.unit
def test_decode_roundtrip(tool, tmp_path):
    frames = {f"{i:015d}": {"pose_keypoints_2d": [i, i + 1, 0.5]} for i in range(3)}
    edges = {"pose_keypoints_2d": [[0, 1], [1, 2]]}
    poses, decoded_edges = tool.decode(write_poses(tmp_path / "p.tar.gz", frames, edges))

    assert sorted(poses) == [0, 1, 2]
    assert poses[2]["pose_keypoints_2d"] == [2, 3, 0.5]
    # Edges come back as tuples so they can be used as hashable skeleton definitions.
    assert decoded_edges == {"pose_keypoints_2d": ((0, 1), (1, 2))}


@pytest.mark.unit
def test_decode_without_edges_falls_back_to_empty(tool, tmp_path):
    frames = {f"{0:015d}": {"pose_keypoints_2d": [1, 2, 1]}}
    _, edges = tool.decode(write_poses(tmp_path / "p.tar.gz", frames))
    assert edges == {"pose_keypoints_2d": ()}


@pytest.mark.unit
def test_decode_rejects_non_tar(tool, tmp_path):
    bad = tmp_path / "poses.json"
    bad.write_text("{}")
    with pytest.raises(NameError):
        tool.decode(bad)


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_visualization_rejects_bad_threshold(tool, tmp_path, threshold):
    with pytest.raises(AssertionError):
        tool.visualization(tmp_path / "v.mp4", tmp_path / "p.tar.gz", tmp_path / "out.mp4", threshold)


# --- MediaPipe output formatting (no model needed) ---------------------------

@pytest.fixture
def mediapipe_tool():
    pytest.importorskip("mediapipe", reason="Mediapipe not installed")
    from psifx.video.pose.mediapipe.tool import MediaPipePoseEstimationTool

    return MediaPipePoseEstimationTool(device="cpu", overwrite=True, verbose=False)


def fake_landmarks(points):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, visibility=v) for x, y, v in points])


@pytest.mark.unit
def test_mediapipe_process_part_scales_to_pixels(mediapipe_tool):
    out = mediapipe_tool.process_part(fake_landmarks([(0.0, 0.0, 0.9), (1.0, 0.5, 0.2)]), size=(101, 201), n_points=2)
    np.testing.assert_allclose(out, [0, 0, 0.9, 200, 50, 0.2], rtol=1e-6)


@pytest.mark.unit
def test_mediapipe_process_part_missing_is_zeros(mediapipe_tool):
    assert mediapipe_tool.process_part(None, size=(10, 10), n_points=4) == [0.0] * 12


@pytest.mark.unit
def test_mediapipe_process_pose_has_all_parts(mediapipe_tool):
    from psifx.video.pose.mediapipe import skeleton

    results = SimpleNamespace(pose_landmarks=None, face_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None)
    pose = mediapipe_tool.process_pose(results, size=(10, 10))
    assert set(pose) == {
        "pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "pose_keypoints_3d"
    }
    assert len(pose["pose_keypoints_2d"]) == skeleton.N_POSE_LANDMARKS * 3
    assert pose["pose_keypoints_3d"] == [0.0] * (skeleton.N_POSE_LANDMARKS * 4)


@pytest.mark.unit
def test_mediapipe_process_world_keeps_meters(mediapipe_tool):
    world = SimpleNamespace(landmark=[SimpleNamespace(x=0.1, y=-0.5, z=0.2, visibility=0.9)])
    np.testing.assert_allclose(mediapipe_tool.process_world(world, n_points=1), [0.1, -0.5, 0.2, 0.9], rtol=1e-6)
