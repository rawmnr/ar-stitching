import math

import numpy as np

from stitching.trusted.eval.metrics import geometry_metrics, signal_metrics
from stitching.trusted.scan.transforms import apply_integer_shift, rotation_matrix_deg


def test_geometry_metrics_for_identical_masks_are_perfect() -> None:
    mask = np.ones((4, 4), dtype=bool)

    metrics = geometry_metrics(mask, mask)

    assert metrics["footprint_iou"] == 1.0
    assert metrics["valid_pixel_recall"] == 1.0
    assert metrics["valid_pixel_precision"] == 1.0
    assert metrics["largest_component_ratio"] == 1.0
    assert metrics["hole_ratio"] == 0.0


def test_rotation_matrix_is_orthonormal_at_ninety_degrees() -> None:
    rotation = rotation_matrix_deg(90.0)
    identity = rotation.T @ rotation

    assert np.allclose(identity, np.eye(2), atol=1e-12)


def test_integer_shift_preserves_sum_when_content_remains_in_frame() -> None:
    values = np.zeros((5, 5), dtype=float)
    values[2, 2] = 1.0

    shifted = apply_integer_shift(values, (1, 0))

    assert shifted.sum() == values.sum()
    assert shifted[2, 3] == 1.0


def test_integer_shift_keeps_mask_and_values_aligned() -> None:
    values = np.zeros((5, 5), dtype=float)
    values[1, 1] = 2.0
    mask = values > 0.0

    shifted_values = apply_integer_shift(values, (2, 1))
    shifted_mask = apply_integer_shift(mask.astype(np.uint8), (2, 1)).astype(bool)

    assert shifted_mask[2, 3]
    assert shifted_values[2, 3] == 2.0
    assert np.all(shifted_values[~shifted_mask] == 0.0)


def test_integer_shift_reduces_coverage_near_borders() -> None:
    mask = np.ones((5, 5), dtype=bool)

    shifted_mask = apply_integer_shift(mask.astype(np.uint8), (3, 0)).astype(bool)

    assert shifted_mask.sum() < mask.sum()


def test_integer_shift_larger_than_image_size_returns_empty() -> None:
    values = np.ones((4, 4), dtype=float)

    shifted = apply_integer_shift(values, (10, 0))

    assert np.count_nonzero(shifted) == 0


def test_hole_ratio_detects_donut_mask() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False

    metrics = geometry_metrics(mask, mask)

    assert math.isclose(metrics["hole_ratio"], 1.0 / 24.0)


def test_largest_component_ratio_detects_fragmentation() -> None:
    reference = np.ones((5, 5), dtype=bool)
    fragmented = np.zeros((5, 5), dtype=bool)
    fragmented[0, 0] = True
    fragmented[0, 1] = True
    fragmented[4, 4] = True

    metrics = geometry_metrics(reference, fragmented)

    assert math.isclose(metrics["largest_component_ratio"], 2.0 / 3.0)


def test_empty_union_is_not_perfect_geometry() -> None:
    empty = np.zeros((3, 3), dtype=bool)

    metrics = geometry_metrics(empty, empty)

    assert metrics["footprint_iou"] == 0.0
    assert metrics["valid_pixel_recall"] == 0.0
    assert metrics["valid_pixel_precision"] == 0.0


def test_empty_valid_intersection_is_not_perfect_signal() -> None:
    reference = np.ones((3, 3), dtype=float)
    candidate = np.ones((3, 3), dtype=float)
    empty_intersection = np.zeros((3, 3), dtype=bool)

    metrics = signal_metrics(reference, candidate, empty_intersection)

    assert math.isinf(metrics["rms_on_valid_intersection"])
    assert math.isinf(metrics["mae_on_valid_intersection"])
    assert metrics["hf_retention"] == 0.0


def test_hf_retention_on_near_flat_truth_is_not_reported_as_perfect() -> None:
    reference = np.full((4, 4), 1.0, dtype=float)
    candidate = np.full((4, 4), 1.0, dtype=float)
    valid = np.ones((4, 4), dtype=bool)

    metrics = signal_metrics(reference, candidate, valid)

    assert metrics["hf_retention"] == 0.0
