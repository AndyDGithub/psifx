"""Unit tests for the psifx archive -> COCO evaluation helpers (research/benchmark/psifx_eval.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from common import COCO_KEYPOINT_NAMES, MEDIAPIPE_TO_COCO17, NUM_COCO_KEYPOINTS
from psifx_eval import ADAPTERS, GroundTruth, evaluate_archive, letterbox, load_coco_samples, read_archive
from psifx.io.tar import TarWriter

COCO_DIR = Path(__file__).resolve().parents[2] / "integration" / "data" / "CocoPose"


def write_archive(path: Path, frames: list) -> Path:
    files = {"edges.json": json.dumps({"pose_keypoints_2d": []})}
    files.update({f"{i:015d}.json": json.dumps(frame) for i, frame in enumerate(frames)})
    TarWriter.write(files, path, overwrite=True, verbose=False)
    return path


def mediapipe_frame_from_coco(coco_xy: np.ndarray) -> dict:
    """Build a psifx MediaPipe frame (33 landmarks) whose COCO-17 subset is ``coco_xy``."""
    points = np.zeros((33, 3))
    points[:, 2] = 1.0
    for name, mp_idx in MEDIAPIPE_TO_COCO17.items():
        points[mp_idx, :2] = coco_xy[COCO_KEYPOINT_NAMES.index(name)]
    return {"pose_keypoints_2d": points.flatten().tolist()}


@pytest.fixture
def gt() -> GroundTruth:
    kp = np.random.default_rng(0).uniform(100, 300, size=(NUM_COCO_KEYPOINTS, 2))
    return GroundTruth(keypoints=kp, visible=np.ones(NUM_COCO_KEYPOINTS, bool), area=40_000.0, bbox_diagonal=280.0)


@pytest.mark.unit
@pytest.mark.parametrize("h, w", [(500, 276), (300, 400), (640, 640)])
def test_letterbox_maps_corners(h, w):
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas, scale, ox, oy = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3)
    # Image corners land inside the white region, padding stays black.
    x2, y2 = (w - 1) * scale + ox, (h - 1) * scale + oy
    assert canvas[int(oy), int(ox)].min() == 255
    assert canvas[int(y2), int(x2)].min() == 255
    assert max(h, w) * scale == pytest.approx(640)
    if ox > 0:
        assert canvas[:, 0].max() == 0


@pytest.mark.unit
def test_mediapipe_adapter_maps_landmarks(gt):
    coco = ADAPTERS["mediapipe"](mediapipe_frame_from_coco(gt.keypoints))
    np.testing.assert_allclose(coco[:, :2], gt.keypoints)


@pytest.mark.unit
def test_mediapipe_adapter_empty_frame_is_nan():
    coco = ADAPTERS["mediapipe"]({"pose_keypoints_2d": [0.0] * 33 * 3})
    assert np.all(np.isnan(coco))


@pytest.mark.unit
def test_evaluate_archive_perfect_and_missed(tmp_path, gt):
    """One perfect frame + one undetected frame -> OKS 0.5, detection 50%."""
    frames = [mediapipe_frame_from_coco(gt.keypoints), {"pose_keypoints_2d": [0.0] * 99}]
    archive = write_archive(tmp_path / "poses.tar.gz", frames)
    scores = evaluate_archive(archive, [gt, gt], "mediapipe")
    assert scores["oks"] == pytest.approx(0.5)
    assert scores["pck"] == pytest.approx(0.5)
    assert scores["detection_rate"] == pytest.approx(0.5)


@pytest.mark.unit
def test_read_archive_orders_frames_numerically(tmp_path):
    frames = [{"pose_keypoints_2d": [float(i)]} for i in range(12)]
    archive = write_archive(tmp_path / "poses.tar.gz", frames)
    assert [f["pose_keypoints_2d"][0] for f in read_archive(archive)] == list(range(12))


@pytest.mark.unit
def test_evaluate_archive_rejects_frame_count_mismatch(tmp_path, gt):
    archive = write_archive(tmp_path / "poses.tar.gz", [mediapipe_frame_from_coco(gt.keypoints)])
    with pytest.raises(ValueError, match="frames"):
        evaluate_archive(archive, [gt, gt], "mediapipe")


@pytest.mark.unit
def test_evaluate_archive_rejects_unknown_tool(tmp_path, gt):
    archive = write_archive(tmp_path / "poses.tar.gz", [mediapipe_frame_from_coco(gt.keypoints)])
    with pytest.raises(KeyError, match="adapter"):
        evaluate_archive(archive, [gt], "does-not-exist")


@pytest.mark.unit
def test_load_coco_test_subset():
    samples = load_coco_samples(COCO_DIR / "annotations.json", COCO_DIR / "images")
    assert len(samples) == 8
    assert all(s["image_path"].is_file() for s in samples)


@pytest.mark.unit
def test_load_coco_rejects_multi_person_images(tmp_path):
    ann = {"image_id": 1, "keypoints": [0] * 51, "bbox": [0, 0, 1, 1], "area": 1}
    (tmp_path / "a.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "x.jpg"}], "annotations": [ann, ann]}))
    with pytest.raises(ValueError, match="exactly 1"):
        load_coco_samples(tmp_path / "a.json", tmp_path)
