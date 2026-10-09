"""Unit tests for the ViTPose tool, with detector and pose model replaced by fakes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from psifx.video.pose.vitpose.tool import PERSON_LABEL, ViTPosePoseEstimationTool


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeDetectorProcessor:
    def __init__(self, boxes, labels):
        self.boxes, self.labels = boxes, labels

    def __call__(self, images, return_tensors):
        return FakeInputs()

    def post_process_object_detection(self, outputs, target_sizes, threshold):
        return [{"boxes": torch.tensor(self.boxes, dtype=torch.float32), "labels": torch.tensor(self.labels)}]


class FakePoseProcessor:
    def __init__(self):
        self.received_boxes = None

    def __call__(self, image, boxes, return_tensors):
        self.received_boxes = boxes[0].copy()
        return FakeInputs()

    def post_process_pose_estimation(self, outputs, boxes):
        return [[
            {"keypoints": torch.full((17, 2), float(i)), "scores": torch.full((17,), 0.9)}
            for i in range(len(boxes[0]))
        ]]


def make_tool(boxes, labels) -> ViTPosePoseEstimationTool:
    tool = ViTPosePoseEstimationTool.__new__(ViTPosePoseEstimationTool)
    tool.device = "cpu"
    tool.detection_threshold = 0.3
    tool.detector_processor = FakeDetectorProcessor(boxes, labels)
    tool.detector = lambda **kwargs: None
    tool.pose_processor = FakePoseProcessor()
    tool.pose_model = lambda **kwargs: None
    return tool


IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.mark.unit
def test_detect_keeps_only_people():
    tool = make_tool([[0, 0, 10, 10], [5, 5, 50, 60]], [PERSON_LABEL + 1, PERSON_LABEL])
    np.testing.assert_allclose(tool.detect(IMAGE), [[5, 5, 50, 60]])


@pytest.mark.unit
def test_predict_sends_xywh_boxes_and_returns_xyxy():
    tool = make_tool([[10, 20, 50, 80]], [PERSON_LABEL])
    people = tool.predict(IMAGE)
    np.testing.assert_allclose(tool.pose_processor.received_boxes, [[10, 20, 40, 60]])
    assert len(people) == 1
    keypoints, scores, bbox = people[0]
    assert keypoints.shape == (17, 2) and scores.shape == (17,)
    np.testing.assert_allclose(bbox, [10, 20, 50, 80])


@pytest.mark.unit
def test_predict_without_people_skips_pose_model():
    tool = make_tool(np.empty((0, 4)), [])
    assert tool.predict(IMAGE) == []
    assert tool.pose_processor.received_boxes is None


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="detection_threshold"):
        ViTPosePoseEstimationTool(detection_threshold=threshold, verbose=False)
