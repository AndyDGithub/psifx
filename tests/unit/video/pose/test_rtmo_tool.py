"""Unit tests for the RTMO tool, with rtmlib replaced by a fake (no download, no model)."""

import sys
import types

import numpy as np
import pytest


class FakeBody:
    """Mimics rtmlib.Body with pose='rtmo': calling it returns (keypoints, scores), no boxes."""

    def __init__(self, pose, mode, backend, device):
        self.init_args = (pose, mode, backend, device)
        self.seen = []
        self.keypoints = np.stack([
            np.random.default_rng(0).uniform(0, 10, (17, 2)),   # small person
            np.random.default_rng(1).uniform(0, 100, (17, 2)),  # large person
        ])

    def __call__(self, image):
        self.seen.append(image.copy())
        return self.keypoints, np.full((len(self.keypoints), 17), 0.6)


@pytest.fixture
def tool(monkeypatch):
    fake_rtmlib = types.ModuleType("rtmlib")
    fake_rtmlib.Body = FakeBody
    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    from psifx.video.pose.rtmo.tool import RTMOPoseEstimationTool

    return RTMOPoseEstimationTool(device="cpu", verbose=False)


@pytest.mark.unit
def test_loads_one_stage_model(tool):
    assert tool.model.init_args[0] == "rtmo"


@pytest.mark.unit
def test_predict_converts_rgb_to_bgr(tool):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255
    tool.predict(image)
    assert tool.model.seen[0][0, 0].tolist() == [0, 0, 255]


@pytest.mark.unit
def test_predict_builds_boxes_from_keypoints(tool):
    people = tool.predict(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(people) == 2
    kp, sc, bbox = people[1]
    np.testing.assert_allclose(bbox, [*kp.min(axis=0), *kp.max(axis=0)], rtol=1e-6)


@pytest.mark.unit
def test_largest_person_selected(tool):
    flat = np.asarray(tool.select_person(tool.predict(np.zeros((4, 4, 3), dtype=np.uint8)))).reshape(17, 3)
    np.testing.assert_allclose(flat[:, :2], tool.model.keypoints[1], rtol=1e-6)


@pytest.mark.unit
def test_no_person(tool):
    tool.model.keypoints = np.empty((0, 17, 2))
    assert tool.predict(np.zeros((4, 4, 3), dtype=np.uint8)) == []


@pytest.mark.unit
def test_cuda_falls_back_to_cpu_without_provider(tool):
    from psifx.video.pose.rtmo.tool import RTMOPoseEstimationTool

    assert RTMOPoseEstimationTool(device="cuda", verbose=False).model.init_args[3] == "cpu"
