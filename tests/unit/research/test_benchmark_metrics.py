"""Unit tests for the pose benchmark metrics (research/benchmark/metrics.py)."""

import numpy as np
import pytest

from common import COCO_KEYPOINT_NAMES, NUM_COCO_KEYPOINTS, PersonPose, empty_person_pose
from metrics import (
    match_prediction_to_gt,
    normalized_jerk_series,
    object_keypoint_similarity,
    pck,
    person_scale,
)


def make_pose(keypoints: np.ndarray) -> PersonPose:
    return PersonPose(keypoints=keypoints.astype(np.float64), scores=np.ones(NUM_COCO_KEYPOINTS))


@pytest.fixture
def gt_keypoints() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(100, 300, size=(NUM_COCO_KEYPOINTS, 2))


@pytest.fixture
def all_visible() -> np.ndarray:
    return np.ones(NUM_COCO_KEYPOINTS, dtype=bool)


# --- OKS ---------------------------------------------------------------------

@pytest.mark.unit
def test_oks_perfect_prediction_is_one(gt_keypoints, all_visible):
    assert object_keypoint_similarity(make_pose(gt_keypoints), gt_keypoints, all_visible, 10_000) == pytest.approx(1.0)


@pytest.mark.unit
def test_oks_far_prediction_is_near_zero(gt_keypoints, all_visible):
    pred = make_pose(gt_keypoints + 1_000)
    assert object_keypoint_similarity(pred, gt_keypoints, all_visible, 10_000) < 1e-6


@pytest.mark.unit
def test_oks_decreases_with_error(gt_keypoints, all_visible):
    small = object_keypoint_similarity(make_pose(gt_keypoints + 2), gt_keypoints, all_visible, 10_000)
    large = object_keypoint_similarity(make_pose(gt_keypoints + 10), gt_keypoints, all_visible, 10_000)
    assert 1.0 > small > large > 0.0


@pytest.mark.unit
def test_oks_is_scale_invariant(gt_keypoints, all_visible):
    """Doubling the image (coords x2, area x4) must not change OKS."""
    base = object_keypoint_similarity(make_pose(gt_keypoints + 5), gt_keypoints, all_visible, 10_000)
    scaled = object_keypoint_similarity(make_pose((gt_keypoints + 5) * 2), gt_keypoints * 2, all_visible, 40_000)
    assert scaled == pytest.approx(base)


@pytest.mark.unit
def test_oks_undefined_without_visible_gt(gt_keypoints):
    none_visible = np.zeros(NUM_COCO_KEYPOINTS, dtype=bool)
    assert object_keypoint_similarity(make_pose(gt_keypoints), gt_keypoints, none_visible, 10_000) is None


@pytest.mark.unit
def test_oks_ignores_invisible_gt_keypoints(gt_keypoints):
    visible = np.zeros(NUM_COCO_KEYPOINTS, dtype=bool)
    visible[:5] = True
    pred = gt_keypoints.copy()
    pred[5:] += 1_000  # wrong only where GT is not labeled
    assert object_keypoint_similarity(make_pose(pred), gt_keypoints, visible, 10_000) == pytest.approx(1.0)


@pytest.mark.unit
def test_oks_all_nan_prediction_is_zero(gt_keypoints, all_visible):
    assert object_keypoint_similarity(empty_person_pose(), gt_keypoints, all_visible, 10_000) == 0.0


# --- PCK ---------------------------------------------------------------------

@pytest.mark.unit
def test_pck_threshold_boundary(gt_keypoints, all_visible):
    """Error exactly at threshold * ref_length counts as correct, just above does not."""
    ref = 100.0
    gt = np.round(gt_keypoints)  # integer coords: the 20 px offset stays exact in float
    at = gt.copy()
    at[:, 0] += 0.2 * ref
    above = gt.copy()
    above[:, 0] += 0.2 * ref + 1e-6
    assert pck(make_pose(at), gt, all_visible, ref) == pytest.approx(1.0)
    assert pck(make_pose(above), gt, all_visible, ref) == pytest.approx(0.0)


@pytest.mark.unit
def test_pck_partial(gt_keypoints, all_visible):
    pred = gt_keypoints.copy()
    pred[:4] += 1_000
    assert pck(make_pose(pred), gt_keypoints, all_visible, 100.0) == pytest.approx(13 / 17)


@pytest.mark.unit
@pytest.mark.parametrize("ref_length", [0.0, -1.0])
def test_pck_undefined_for_invalid_reference(gt_keypoints, all_visible, ref_length):
    assert pck(make_pose(gt_keypoints), gt_keypoints, all_visible, ref_length) is None


# --- matching ----------------------------------------------------------------

@pytest.mark.unit
def test_match_picks_closest_person(gt_keypoints):
    near, far = make_pose(gt_keypoints + 3), make_pose(gt_keypoints + 500)
    assert match_prediction_to_gt([far, near], gt_keypoints) is near


@pytest.mark.unit
def test_match_without_predictions_is_none(gt_keypoints):
    assert match_prediction_to_gt([], gt_keypoints) is None
    assert match_prediction_to_gt([empty_person_pose()], gt_keypoints) is None


# --- temporal stability ------------------------------------------------------

@pytest.mark.unit
def test_jerk_is_zero_for_constant_velocity():
    t = np.arange(20, dtype=np.float64)
    track = np.stack([3 * t, 5 * t], axis=1)
    assert normalized_jerk_series(track, fps=25, scale=1.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_jerk_scales_inversely_with_person_scale():
    track = np.random.default_rng(1).normal(size=(30, 2))
    j1 = normalized_jerk_series(track, fps=25, scale=1.0)
    j2 = normalized_jerk_series(track, fps=25, scale=2.0)
    assert j2 == pytest.approx(j1 / 2)


@pytest.mark.unit
def test_jerk_needs_three_valid_frames():
    track = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan]])
    assert np.isnan(normalized_jerk_series(track, fps=25, scale=1.0))


@pytest.mark.unit
def test_person_scale_shoulder_to_hip(gt_keypoints):
    ls, rh = COCO_KEYPOINT_NAMES.index("left_shoulder"), COCO_KEYPOINT_NAMES.index("right_hip")
    kp = gt_keypoints.copy()
    kp[ls], kp[rh] = [0, 0], [3, 4]
    assert person_scale(make_pose(kp)) == pytest.approx(5.0)
    kp[rh] = np.nan
    assert np.isnan(person_scale(make_pose(kp)))


@pytest.mark.unit
def test_person_pose_rejects_wrong_shape():
    with pytest.raises(AssertionError):
        PersonPose(keypoints=np.zeros((33, 2)), scores=np.zeros(33))
