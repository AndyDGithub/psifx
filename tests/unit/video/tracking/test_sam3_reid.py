"""Unit tests for SAM3 shot detection, re-identification and post-processing stand-ins."""

import numpy as np
from PIL import Image
import pytest
import torch

from psifx.video.tracking.sam3.postprocessing import TorchCvUtils
from psifx.video.tracking.sam3.reid import AppearanceGallery, ShotDetector, sample_frames

# Importing the tool stubs out the video and SAM3 dependencies when they are missing.
from tests.unit.video.tracking.test_sam3_tool import make_tool


def solid(color, size=(64, 48)) -> np.ndarray:
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def make_gallery(threshold: float = 0.9) -> AppearanceGallery:
    """Gallery whose embedding is the normalized mean color of the crops (no model)."""
    gallery = AppearanceGallery.__new__(AppearanceGallery)
    gallery.threshold = threshold
    gallery.max_samples = 32
    gallery.samples = {}

    def embed(crops):
        if not crops:
            return None
        color = np.mean([np.asarray(crop, dtype=float).reshape(-1, 3).mean(0) for crop in crops], axis=0)
        return color / np.linalg.norm(color)

    gallery.embed = embed
    return gallery


@pytest.mark.unit
def test_shot_detector_flags_cut_not_motion():
    detector = ShotDetector(threshold=0.15)
    frames = []
    # A bright square sliding on a dark background: continuous motion.
    for i in range(20):
        frame = solid((20, 20, 20), size=(320, 180))
        frame[60:120, 10 + 8 * i:70 + 8 * i] = 230
        frames.append(frame)
    frames.append(solid((200, 150, 30), size=(320, 180)))  # cut to another scene
    cuts = [i for i, frame in enumerate(frames) if detector.is_cut(frame)]
    assert cuts == [20]


@pytest.mark.unit
def test_sample_frames_spreads_and_keeps_short_lists():
    assert sample_frames([3, 1, 2], 4) == [1, 2, 3]
    assert sample_frames(range(10), 4) == [0, 3, 6, 9]


@pytest.mark.unit
def test_gallery_crop_grays_out_background():
    image = Image.fromarray(solid((255, 0, 0), size=(20, 20)))
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 4:14] = True
    mask[2, 4] = False
    crop = np.asarray(AppearanceGallery.crop(image, mask))
    assert crop.shape == (10, 10, 3)
    assert crop[0, 0].tolist() == [127, 127, 127]
    assert crop[5, 5].tolist() == [255, 0, 0]


def chunk_with(objects):
    """Chunk of 2 frames; ``objects`` maps an object id to (color, frames). Each object fills its own band."""
    frames = [solid((0, 0, 0)) for _ in range(2)]
    outputs = {i: {"object_ids": [], "masks": []} for i in range(2)}
    for k, (obj_id, (color, present)) in enumerate(objects.items()):
        for i in present:
            frames[i][:, 16 * k:16 * k + 16] = color
            mask = np.zeros((48, 64), dtype=bool)
            mask[:, 16 * k:16 * k + 16] = True
            outputs[i]["object_ids"].append(obj_id)
            outputs[i]["masks"].append(mask)
    return [Image.fromarray(f) for f in frames], outputs


@pytest.mark.unit
def test_reidentification_links_new_tracks_to_known_people():
    tool = make_tool()
    gallery = make_gallery()
    red, blue = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    gallery.add(0, red)
    gallery.add(1, blue)

    chunk, outputs = chunk_with({5: ((0, 0, 250), [0, 1]), 6: ((250, 0, 0), [0, 1])})
    mapping, next_id = tool._map_chunk_object_ids(
        chunk_outputs=outputs, prev_last_global_masks={}, iou_threshold=0.3, next_global_id=2,
        chunk=chunk, gallery=gallery,
    )
    assert mapping == {5: 1, 6: 0}
    assert next_id == 2


@pytest.mark.unit
def test_reidentification_never_merges_people_seen_together():
    tool = make_tool()
    gallery = make_gallery()
    gallery.add(0, np.array([1.0, 0.0, 0.0]))

    # Two red objects in the same frames: only one can be person 0.
    chunk, outputs = chunk_with({5: ((250, 0, 0), [0, 1]), 6: ((240, 0, 0), [0, 1])})
    mapping, next_id = tool._map_chunk_object_ids(
        chunk_outputs=outputs, prev_last_global_masks={}, iou_threshold=0.3, next_global_id=1,
        chunk=chunk, gallery=gallery,
    )
    assert mapping == {5: 0, 6: 1}
    assert next_id == 2


@pytest.mark.unit
def test_reidentification_with_cap_takes_the_most_similar_person():
    tool = make_tool()
    gallery = make_gallery(threshold=0.99)
    gallery.add(0, np.array([1.0, 0.0, 0.0]))
    gallery.add(1, np.array([0.0, 0.0, 1.0]))

    # Purple-ish is below the threshold for both, but no new id is allowed.
    chunk, outputs = chunk_with({5: ((200, 0, 120), [0, 1])})
    mapping, next_id = tool._map_chunk_object_ids(
        chunk_outputs=outputs, prev_last_global_masks={}, iou_threshold=0.3, next_global_id=2,
        max_num_objects=2, chunk=chunk, gallery=gallery,
    )
    assert mapping == {5: 0}
    assert next_id == 2


@pytest.mark.unit
def test_reidentification_creates_new_id_for_unknown_person():
    tool = make_tool()
    gallery = make_gallery()
    gallery.add(0, np.array([1.0, 0.0, 0.0]))

    chunk, outputs = chunk_with({5: ((0, 250, 0), [1])})
    mapping, next_id = tool._map_chunk_object_ids(
        chunk_outputs=outputs, prev_last_global_masks={}, iou_threshold=0.3, next_global_id=1,
        chunk=chunk, gallery=gallery,
    )
    assert mapping == {5: 1}
    assert next_id == 2
    assert set(gallery.samples) == {0, 1}


@pytest.mark.unit
def test_torch_nms_drops_overlapping_lower_scores():
    ious = torch.tensor([[1.0, 0.9, 0.1], [0.9, 1.0, 0.0], [0.1, 0.0, 1.0]])
    scores = torch.tensor([0.8, 0.9, 0.7])
    kept = TorchCvUtils.generic_nms(ious, scores, 0.5, use_iou_matrix=True)
    assert kept.tolist() == [1, 2]


@pytest.mark.unit
def test_torch_connected_components_counts_component_sizes():
    mask = torch.zeros((1, 1, 6, 6), dtype=torch.uint8)
    mask[0, 0, 0:2, 0:2] = 1  # 4 pixels
    mask[0, 0, 4, 4] = 1  # 1 pixel
    mask[0, 0, 5, 5] = 1  # touches diagonally: same component
    labels, counts = TorchCvUtils.cc_2d(mask, get_counts=True)
    assert counts[0, 0, 0, 0] == 4
    assert counts[0, 0, 4, 4] == 2 and counts[0, 0, 5, 5] == 2
    assert counts[0, 0, 3, 3] == 0 and labels[0, 0, 3, 3] == 0
    assert labels[0, 0, 4, 4] == labels[0, 0, 5, 5] != labels[0, 0, 0, 0]
