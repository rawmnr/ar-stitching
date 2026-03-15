import math

import numpy as np
from scipy import ndimage

from stitching.contracts import ScenarioConfig
from stitching.contracts import ReconstructionSurface, SubApertureObservation, SurfaceTruth
from stitching.trusted.eval.metrics import build_eval_report, geometry_metrics, signal_acceptance_threshold, signal_metrics
from stitching.trusted.noise.models import outlier_magnitude_scale
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

    assert math.isfinite(metrics["rms_on_valid_intersection"])
    assert math.isfinite(metrics["mae_on_valid_intersection"])
    assert metrics["rms_on_valid_intersection"] >= 1.0
    assert metrics["mae_on_valid_intersection"] >= 1.0
    assert metrics["hf_retention"] == 0.0


def test_hf_retention_on_near_flat_truth_is_not_reported_as_perfect() -> None:
    reference = np.full((4, 4), 1.0, dtype=float)
    candidate = np.full((4, 4), 1.0, dtype=float)
    valid = np.ones((4, 4), dtype=bool)

    metrics = signal_metrics(reference, candidate, valid)

    assert metrics["hf_retention"] == 0.0


def test_hf_retention_detects_blur_on_non_flat_truth() -> None:
    reference = np.zeros((7, 7), dtype=float)
    reference[3, 3] = 1.0
    candidate = np.array(reference, copy=True)
    candidate[3, 3] = 0.25
    candidate[2, 3] = 0.1875
    candidate[4, 3] = 0.1875
    candidate[3, 2] = 0.1875
    candidate[3, 4] = 0.1875
    valid = np.ones((7, 7), dtype=bool)

    metrics = signal_metrics(reference, candidate, valid)

    assert metrics["hf_retention"] < 1.0


def test_hf_retention_detects_high_frequency_noise_injection() -> None:
    yy, xx = np.indices((7, 7))
    reference = xx.astype(float) + yy.astype(float)
    candidate = reference.copy()
    candidate[1::2, 1::2] += 0.5
    valid = np.ones((7, 7), dtype=bool)

    metrics = signal_metrics(reference, candidate, valid)

    assert metrics["hf_retention"] < 1.0


def test_hf_retention_drops_under_gaussian_blur() -> None:
    reference = np.zeros((9, 9), dtype=float)
    reference[4, 4] = 1.0
    blurred = ndimage.gaussian_filter(reference, sigma=1.0)
    valid = np.ones((9, 9), dtype=bool)

    exact_metrics = signal_metrics(reference, reference, valid)
    blurred_metrics = signal_metrics(reference, blurred, valid)

    assert exact_metrics["hf_retention"] == 1.0
    assert blurred_metrics["hf_retention"] < exact_metrics["hf_retention"]


def test_signal_acceptance_threshold_is_exact_for_clean_scenarios() -> None:
    config = ScenarioConfig(
        scenario_id="clean",
        description="clean",
        grid_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
    )

    assert signal_acceptance_threshold(config) == 1e-12


def test_signal_acceptance_threshold_scales_with_noise_and_outliers() -> None:
    config = ScenarioConfig(
        scenario_id="corrupt",
        description="corrupt",
        grid_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        gaussian_noise_std=0.02,
        outlier_fraction=0.05,
        retrace_error=0.01,
    )
    reference = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=float,
    )
    valid_mask = np.ones_like(reference, dtype=bool)

    expected = 3.0 * config.gaussian_noise_std + config.outlier_fraction * outlier_magnitude_scale(reference, valid_mask) + abs(config.retrace_error)
    assert math.isclose(signal_acceptance_threshold(config, reference, valid_mask), expected, rel_tol=1e-9)


def test_eval_report_accepts_error_within_outlier_budget() -> None:
    truth_z = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=float,
    )
    truth = SurfaceTruth(z=truth_z, valid_mask=np.ones((3, 3), dtype=bool), pixel_size=1.0)
    config = ScenarioConfig(
        scenario_id="outlier_budget",
        description="outlier budget",
        grid_shape=(3, 3),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        outlier_fraction=0.1,
    )
    threshold = signal_acceptance_threshold(config, truth.z, truth.valid_mask)
    candidate = ReconstructionSurface(
        z=truth.z + (0.5 * threshold),
        valid_mask=np.ones((3, 3), dtype=bool),
        source_observation_ids=("obs",),
        observed_support_mask=np.ones((3, 3), dtype=bool),
    )

    observation = SubApertureObservation(
        observation_id="obs",
        z=np.array(truth.z, copy=True),
        valid_mask=np.array(truth.valid_mask, copy=True),
        tile_shape=(3, 3),
        center_xy=(1.0, 1.0),
        global_shape=(3, 3),
        rotation_deg=0.0,
    )

    report = build_eval_report(config, truth, candidate, observations=(observation,), runtime_sec=0.0)

    assert report.accepted is True
