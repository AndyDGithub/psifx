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
    # The Tasks API returns plain lists of landmarks; face and hand landmarks have no visibility.
    return [SimpleNamespace(x=x, y=y, z=0.0, visibility=v) for x, y, v in points]


@pytest.mark.unit
def test_mediapipe_process_part_scales_to_pixels(mediapipe_tool):
    out = mediapipe_tool.process_part(fake_landmarks([(0.0, 0.0, 0.9), (1.0, 0.5, 0.2)]), size=(101, 201), n_points=2)
    np.testing.assert_allclose(out, [0, 0, 0.9, 200, 50, 0.2], rtol=1e-6)


@pytest.mark.unit
def test_mediapipe_process_part_missing_is_zeros(mediapipe_tool):
    assert mediapipe_tool.process_part([], size=(10, 10), n_points=4) == [0.0] * 12


@pytest.mark.unit
def test_mediapipe_process_part_without_visibility_is_zero_confidence(mediapipe_tool):
    out = mediapipe_tool.process_part(fake_landmarks([(0.5, 0.5, None)]), size=(11, 11), n_points=1)
    assert out == [5.0, 5.0, 0.0]


@pytest.mark.unit
def test_mediapipe_process_world_keeps_meters(mediapipe_tool):
    landmarks = [SimpleNamespace(x=0.1, y=-0.4, z=0.05, visibility=0.8)]
    np.testing.assert_allclose(mediapipe_tool.process_world(landmarks, n_points=1), [0.1, -0.4, 0.05, 0.8], rtol=1e-6)
    assert mediapipe_tool.process_world([], n_points=2) == [0.0] * 8


@pytest.mark.unit
def test_mediapipe_process_pose_has_all_parts(mediapipe_tool):
    from psifx.video.pose.mediapipe import skeleton

    results = SimpleNamespace(pose_landmarks=[], face_landmarks=[], left_hand_landmarks=[], right_hand_landmarks=[],
                              pose_world_landmarks=[])
    pose = mediapipe_tool.process_pose(results, size=(10, 10))
    assert set(pose) == {"pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d",
                         "pose_keypoints_3d"}
    assert len(pose["pose_keypoints_2d"]) == skeleton.N_POSE_LANDMARKS * 3
    assert len(pose["face_keypoints_2d"]) == skeleton.N_FACE_LANDMARKS * 3
    assert len(pose["pose_keypoints_3d"]) == skeleton.N_POSE_LANDMARKS * 4


@pytest.mark.unit
def test_holistic_model_path_rejects_missing_local_file(tmp_path):
    pytest.importorskip("mediapipe", reason="Mediapipe not installed")
    from psifx.video.pose.mediapipe.tool import bundle_path

    with pytest.raises(FileNotFoundError):
        bundle_path(str(tmp_path / "missing.task"))
    bundle = tmp_path / "model.task"
    bundle.write_bytes(b"x")
    assert bundle_path(str(bundle)) == bundle


@pytest.mark.unit
def test_holistic_model_download_checks_the_checksum(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe", reason="Mediapipe not installed")
    from psifx.video.pose.mediapipe import tool

    monkeypatch.setattr(tool.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(tool.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"corrupted"))
    with pytest.raises(RuntimeError, match="checksum"):
        tool.bundle_path()
    assert not any((tmp_path / ".cache" / "psifx" / "mediapipe").iterdir())


@pytest.mark.unit
def test_face_fallback_head_box_maps_back_to_the_frame():
    pytest.importorskip("mediapipe", reason="Mediapipe not installed")
    from psifx.video.pose.mediapipe.face_fallback import CROP_SIZE, FaceCropFallback

    # Head points spread over 20 x 10 px around (100, 50) in a 200 x 100 frame.
    pose = [SimpleNamespace(x=(90 + 20 * (k % 2)) / 200, y=(45 + 10 * (k % 2)) / 100) for k in range(33)]
    center, side = FaceCropFallback.head_box(pose, 200, 100)
    np.testing.assert_allclose(center, [100, 50], atol=1.0)
    assert side == pytest.approx(60.0)

    # A landmark at the crop center lands back on the head center.
    fallback = FaceCropFallback.__new__(FaceCropFallback)
    point = SimpleNamespace(x=0.5, y=0.5)
    fallback.model = SimpleNamespace(detect_for_video=lambda image, t: SimpleNamespace(face_landmarks=[[point]]))
    out = fallback.detect(np.zeros((100, 200, 3), np.uint8), pose, 0)
    assert out[0].x == pytest.approx(center[0] / 200) and out[0].y == pytest.approx(center[1] / 100)
    assert CROP_SIZE == 256
