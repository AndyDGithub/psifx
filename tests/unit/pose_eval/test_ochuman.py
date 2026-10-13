"""Unit tests for the OCHuman evaluation helpers, on a tiny synthetic COCO-format split."""

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pycocotools")

from tests.pose_eval.ochuman import SPLITS, evaluate_tool, find_split, predictions_to_coco  # noqa: E402

RNG = np.random.default_rng(0)


def person(offset):
    kp = RNG.uniform(20, 80, size=(17, 2)) + offset
    x1, y1 = kp.min(axis=0)
    x2, y2 = kp.max(axis=0)
    return kp, [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


@pytest.fixture
def split(tmp_path):
    """Two images with two annotated people each, in OCHuman's file layout."""
    (tmp_path / "images").mkdir()
    images, annotations, truth = [], [], {}
    for image_id in (1, 2):
        Image.fromarray(np.zeros((200, 200, 3), np.uint8)).save(tmp_path / "images" / f"{image_id}.jpg")
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": 200, "height": 200})
        truth[image_id] = []
        for k, offset in enumerate((0, 90)):
            kp, bbox = person(offset)
            truth[image_id].append(kp)
            annotations.append({
                "id": image_id * 10 + k, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "bbox": bbox, "area": bbox[2] * bbox[3], "num_keypoints": 17,
                "keypoints": np.concatenate([kp, np.full((17, 1), 2)], axis=1).flatten().tolist(),
            })
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
    (tmp_path / SPLITS["test"]).write_text(json.dumps(coco))
    return tmp_path, truth


@pytest.mark.unit
def test_find_split(split, monkeypatch):
    root, _ = split
    monkeypatch.setenv("OCHUMAN_ROOT", str(root))
    images, annotations = find_split("test")
    assert images == root / "images" and annotations.name == SPLITS["test"]
    with pytest.raises(FileNotFoundError):
        find_split("val")  # file not there
    with pytest.raises(ValueError):
        find_split("train")


@pytest.mark.unit
def test_find_split_without_root(monkeypatch):
    monkeypatch.delenv("OCHUMAN_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="OCHUMAN_ROOT"):
        find_split("test")


@pytest.mark.unit
def test_predictions_to_coco():
    kp = np.arange(34, dtype=float).reshape(17, 2)
    result = predictions_to_coco(7, [(kp, np.full(17, 0.5), None)])[0]
    assert result["image_id"] == 7 and result["category_id"] == 1
    assert result["score"] == pytest.approx(0.5)
    assert len(result["keypoints"]) == 51 and result["keypoints"][2] == 0.5


def make_predict(split_fixture, transform):
    root, truth = split_fixture
    calls = iter(sorted(truth))

    def predict(image):
        image_id = next(calls)
        return [(transform(kp), np.ones(17), None) for kp in truth[image_id]]

    return root, predict


@pytest.mark.unit
def test_perfect_predictions_give_ap_one(split):
    root, predict = make_predict(split, lambda kp: kp)
    scores = evaluate_tool(predict, root / "images", root / SPLITS["test"])
    assert scores["AP"] == pytest.approx(1.0) and scores["n_images"] == 2


@pytest.mark.unit
def test_wrong_predictions_give_low_ap(split):
    root, predict = make_predict(split, lambda kp: kp + 40)
    assert evaluate_tool(predict, root / "images", root / SPLITS["test"])["AP"] < 0.2


@pytest.mark.unit
def test_no_prediction_gives_zero(split):
    root, _ = split
    scores = evaluate_tool(lambda image: [], root / "images", root / SPLITS["test"], limit=1)
    assert scores["AP"] == 0.0 and scores["n_images"] == 1
