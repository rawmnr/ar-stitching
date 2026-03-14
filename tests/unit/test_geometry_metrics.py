import numpy as np

from stitching.trusted.eval.metrics import geometry_metrics
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
