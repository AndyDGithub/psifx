"""Unit tests for the COCO-17 pose tool base (psifx/video/pose/coco17)."""

import json
from pathlib import Path

import numpy as np
import pytest

from psifx.io.video import VideoWriter
from psifx.video.pose.coco17 import skeleton
from psifx.video.pose.coco17.command import collect_mask_files
from psifx.video.pose.coco17.tool import Coco17PoseEstimationTool


def person(offset: float, bbox_size: float):
    keypoints = np.arange(34, dtype=np.float32).reshape(17, 2) + offset
    return keypoints, np.full(17, 0.5, dtype=np.float32), np.array([0, 0, bbox_size, bbox_size])


class FakeTool(Coco17PoseEstimationTool):
    """Detects one person on even frames, nobody on odd frames."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def predict(self, image):
        assert image.ndim == 3 and image.shape[2] == 3
        self.calls += 1
        return [person(0, 10)] if (self.calls - 1) % 2 == 0 else []


@pytest.fixture
def small_video(tmp_path) -> Path:
    path = tmp_path / "video.mp4"
    with VideoWriter(path, input_dict={"-r": "5"}, output_dict={"-pix_fmt": "yuv420p"}) as writer:
        for _ in range(4):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    return path


@pytest.mark.unit
def test_skeleton_edges_are_valid_indices():
    assert skeleton.N_POSE_LANDMARKS == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in skeleton.POSE_EDGES)


@pytest.mark.unit
def test_select_person_keeps_largest_bbox():
    flat = Coco17PoseEstimationTool.select_person([person(0, 10), person(100, 50), person(200, 20)])
    points = np.asarray(flat).reshape(17, 3)
    assert points[0, 0] == pytest.approx(100)  # offset of the largest person
    assert np.all(points[:, 2] == pytest.approx(0.5))


@pytest.mark.unit
def test_select_person_nobody_is_zeros():
    assert Coco17PoseEstimationTool.select_person([]) == [0.0] * 51


@pytest.mark.unit
def test_select_person_rejects_wrong_shape():
    bad = (np.zeros((33, 2)), np.zeros(33), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="17"):
        Coco17PoseEstimationTool.select_person([bad])


@pytest.mark.unit
def test_predict_is_abstract():
    with pytest.raises(NotImplementedError):
        Coco17PoseEstimationTool(verbose=False).predict(np.zeros((8, 8, 3), np.uint8))


@pytest.mark.unit
def test_inference_writes_one_entry_per_frame(small_video, tmp_path):
    tool = FakeTool(overwrite=True, verbose=False)
    poses_path = tmp_path / "poses.tar.gz"
    tool.inference(small_video, poses_path)

    poses, edges = tool.decode(poses_path)
    assert sorted(poses) == [0, 1, 2, 3]
    assert edges["pose_keypoints_2d"] == skeleton.POSE_EDGES
    assert np.any(poses[0]["pose_keypoints_2d"])
    assert not np.any(poses[1]["pose_keypoints_2d"])  # nobody -> zeros
    assert len(poses[2]["pose_keypoints_2d"]) == 51


@pytest.mark.unit
def test_inference_refuses_to_overwrite_before_running(small_video, tmp_path):
    poses_path = tmp_path / "poses.tar.gz"
    poses_path.write_bytes(b"")
    tool = FakeTool(overwrite=False, verbose=False)
    with pytest.raises(FileExistsError):
        tool.inference(small_video, poses_path)
    assert tool.calls == 0  # failed before any model call


@pytest.mark.unit
def test_collect_mask_files(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "b.mp4").touch()
    (masks / "a.mp4").touch()
    single = tmp_path / "c.mp4"
    single.touch()
    assert [p.name for p in collect_mask_files([masks, single])] == ["a.mp4", "b.mp4", "c.mp4"]

    (masks / "notes.txt").touch()
    with pytest.raises(ValueError, match="non-.mp4"):
        collect_mask_files([masks])
    with pytest.raises(FileNotFoundError):
        collect_mask_files([tmp_path / "missing.mp4"])


@pytest.mark.unit
def test_bbox_from_keypoints():
    from psifx.video.pose.coco17.tool import bbox_from_keypoints

    kp = np.array([[3, 7], [1, 9], [5, 2]] + [[3, 5]] * 14, dtype=np.float32)
    np.testing.assert_allclose(bbox_from_keypoints(kp), [1, 2, 5, 9])
