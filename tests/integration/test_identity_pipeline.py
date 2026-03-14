from pathlib import Path

import numpy as np

import pytest

from stitching.editable.baseline import baseline_integer_unshift_mean, baseline_integer_unshift_median
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.harness.run_eval import run_baseline_eval, run_identity_eval, run_median_baseline_eval
from stitching.trusted.eval.metrics import _hole_ratio, build_eval_report, signal_acceptance_threshold
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_identity_scenario_is_accepted() -> None:
    report = run_baseline_eval(Path("scenarios/s00_identity.yaml"))

    assert report.accepted is True
    assert report.geometry_metrics["footprint_iou"] == 1.0
    assert report.signal_metrics["rms_on_valid_intersection"] == 0.0


def test_identity_simulator_returns_single_observation_matching_truth() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s00_identity.yaml"))
    truth, observations = simulate_identity_observations(config)

    assert len(observations) == 1
    assert observations[0].translation_xy == (0.0, 0.0)
    assert observations[0].rotation_deg == 0.0
    assert (observations[0].z == truth.z).all()
    assert (observations[0].valid_mask == truth.valid_mask).all()
    assert observations[0].tile_shape == truth.z.shape
    assert not np.allclose(truth.z, 0.0)


def test_baseline_returns_reconstruction_without_mutating_input_observation() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s00_identity.yaml"))
    _, observations = simulate_identity_observations(config)
    original_z = np.array(observations[0].z, copy=True)
    original_mask = np.array(observations[0].valid_mask, copy=True)

    reconstruction = baseline_integer_unshift_mean(observations)

    assert reconstruction.source_observation_ids == (observations[0].observation_id,)
    assert (observations[0].z == original_z).all()
    assert (observations[0].valid_mask == original_mask).all()


def test_shift_only_baseline_is_rejected_when_full_size_tile_clips_at_borders() -> None:
    shifted_report = run_baseline_eval(Path("scenarios/s01_shift_only.yaml"))

    assert shifted_report.accepted is False
    assert shifted_report.signal_metrics["mae_on_valid_intersection"] == 0.0
    assert shifted_report.geometry_metrics["footprint_iou"] < 1.0


def test_clean_scenario_rejects_single_pixel_mask_shrinkage() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s00_identity.yaml"))
    truth, observations = simulate_identity_observations(config)
    reconstruction = baseline_integer_unshift_mean(observations)
    shrunk_mask = np.array(reconstruction.valid_mask, copy=True)
    first_valid = np.argwhere(shrunk_mask)[0]
    shrunk_mask[tuple(first_valid)] = False
    shrunk = ReconstructionSurface(
        z=np.where(shrunk_mask, reconstruction.z, 0.0),
        valid_mask=shrunk_mask,
        source_observation_ids=reconstruction.source_observation_ids,
        observed_support_mask=np.array(reconstruction.observed_support_mask, copy=True),
        metadata=dict(reconstruction.metadata),
    )

    report = build_eval_report(config, truth, shrunk, runtime_sec=0.0)

    assert report.accepted is False
    assert report.geometry_metrics["valid_pixel_recall"] < 1.0


def test_reference_bias_scenario_is_rejected_or_degraded() -> None:
    report = run_baseline_eval(Path("scenarios/s02_reference_bias.yaml"))

    assert report.accepted is False
    assert report.signal_metrics["mae_on_valid_intersection"] > 0.0


def test_noise_scenario_degrades_signal_metrics_against_identity() -> None:
    identity_report = run_baseline_eval(Path("scenarios/s00_identity.yaml"))
    noisy_report = run_baseline_eval(Path("scenarios/s03_noise.yaml"))
    noisy_config = ScenarioConfig.from_yaml(Path("scenarios/s03_noise.yaml"))

    assert noisy_report.signal_metrics["mae_on_valid_intersection"] <= signal_acceptance_threshold(noisy_config)
    assert noisy_report.signal_metrics["rms_on_valid_intersection"] > identity_report.signal_metrics["rms_on_valid_intersection"]
    assert noisy_report.signal_metrics["mae_on_valid_intersection"] > identity_report.signal_metrics["mae_on_valid_intersection"]


def test_retrace_scenario_degrades_signal_metrics_against_identity() -> None:
    identity_report = run_baseline_eval(Path("scenarios/s00_identity.yaml"))
    retrace_report = run_baseline_eval(Path("scenarios/s05_retrace.yaml"))

    assert retrace_report.signal_metrics["mae_on_valid_intersection"] > identity_report.signal_metrics["mae_on_valid_intersection"]
    assert retrace_report.signal_metrics["rms_on_valid_intersection"] > identity_report.signal_metrics["rms_on_valid_intersection"]


def test_run_identity_eval_remains_backward_compatible() -> None:
    report = run_identity_eval(Path("scenarios/s00_identity.yaml"))

    assert report.accepted is True


def test_multi_observation_baseline_improves_coverage_over_single_shift() -> None:
    multi_overlap_report = run_baseline_eval(Path("scenarios/s06_multi_overlap.yaml"))

    assert multi_overlap_report.accepted is True
    assert multi_overlap_report.geometry_metrics["footprint_iou"] == 1.0
    assert multi_overlap_report.geometry_metrics["valid_pixel_recall"] == 1.0


def test_multi_observation_baseline_uses_all_source_observations() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s06_multi_overlap.yaml"))
    _, observations = simulate_identity_observations(config)

    reconstruction = baseline_integer_unshift_mean(observations)

    assert reconstruction.source_observation_ids == tuple(observation.observation_id for observation in observations)
    assert reconstruction.metadata["num_observations_used"] == len(observations)


def test_local_tile_multi_observation_baseline_improves_coverage() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s07_local_tiles.yaml"))
    truth, observations = simulate_identity_observations(config)
    single_reconstruction = baseline_integer_unshift_mean((observations[0],))
    multi_reconstruction = baseline_integer_unshift_mean(observations)
    single_report = build_eval_report(config, truth, single_reconstruction, runtime_sec=0.0)
    multi_report = build_eval_report(config, truth, multi_reconstruction, runtime_sec=0.0)

    assert multi_report.geometry_metrics["valid_pixel_recall"] > single_report.geometry_metrics["valid_pixel_recall"]
    assert multi_report.geometry_metrics["footprint_iou"] > single_report.geometry_metrics["footprint_iou"]


def test_local_tile_observation_requires_global_placement() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s07_local_tiles.yaml"))
    _, observations = simulate_identity_observations(config)
    reconstruction = baseline_integer_unshift_mean(observations)

    assert observations[0].z.shape == config.effective_tile_shape
    assert reconstruction.z.shape == config.grid_shape


def test_outlier_scenario_median_baseline_beats_mean_baseline() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s04_outliers.yaml"))
    truth, observations = simulate_identity_observations(config)
    mean_reconstruction = baseline_integer_unshift_mean(observations)
    median_reconstruction = baseline_integer_unshift_median(observations)
    mean_report = build_eval_report(config, truth, mean_reconstruction, runtime_sec=0.0)
    median_report = build_eval_report(config, truth, median_reconstruction, runtime_sec=0.0)

    assert median_report.signal_metrics["mae_on_valid_intersection"] <= mean_report.signal_metrics["mae_on_valid_intersection"]
    assert median_report.signal_metrics["rms_on_valid_intersection"] <= mean_report.signal_metrics["rms_on_valid_intersection"]
    assert median_report.signal_metrics["mae_on_valid_intersection"] <= signal_acceptance_threshold(config)


def test_trusted_evaluation_requires_observed_support_mask() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s00_identity.yaml"))
    truth, _ = simulate_identity_observations(config)
    candidate = ReconstructionSurface(
        z=np.array(truth.z, copy=True),
        valid_mask=np.array(truth.valid_mask, copy=True),
        source_observation_ids=("obs",),
    )

    with pytest.raises(ValueError):
        build_eval_report(config, truth, candidate, runtime_sec=0.0)


def test_harness_can_select_median_baseline() -> None:
    report = run_median_baseline_eval(Path("scenarios/s04_outliers.yaml"))

    assert report.signal_metrics["mae_on_valid_intersection"] >= 0.0


def test_harness_rejects_unknown_baseline_name() -> None:
    with pytest.raises(ValueError):
        run_baseline_eval(Path("scenarios/s00_identity.yaml"), baseline_name="unknown")


def test_harness_uses_declared_scenario_baseline_by_default() -> None:
    default_report = run_baseline_eval(Path("scenarios/s08_outliers_median.yaml"))
    explicit_report = run_baseline_eval(Path("scenarios/s08_outliers_median.yaml"), baseline_name="median")

    assert default_report.signal_metrics == explicit_report.signal_metrics


def test_harness_allows_explicit_override_of_declared_baseline() -> None:
    median_report = run_baseline_eval(Path("scenarios/s08_outliers_median.yaml"))
    mean_report = run_baseline_eval(Path("scenarios/s08_outliers_median.yaml"), baseline_name="mean")

    assert median_report.signal_metrics["mae_on_valid_intersection"] <= mean_report.signal_metrics["mae_on_valid_intersection"]


def test_dc_piston_scenario_rejects_naive_baseline_that_ignores_nuisance_terms() -> None:
    report = run_baseline_eval(Path("scenarios/s10_dc_piston.yaml"), baseline_name="mean")

    assert report.accepted is False
    assert report.signal_metrics["mae_on_valid_intersection"] > 0.0


def test_subpixel_offsets_are_rejected_explicitly_not_rounded() -> None:
    config = ScenarioConfig(
        scenario_id="subpixel_rejected",
        description="subpixel rejected",
        grid_shape=(9, 9),
        tile_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((0.5, 0.5),),
        seed=0,
    )

    with pytest.raises(ValueError):
        simulate_identity_observations(config)


def test_large_grid_hole_ratio_smoke_does_not_hang_and_detects_hole() -> None:
    mask = np.ones((2048, 2048), dtype=bool)
    mask[1024, 1024] = False

    ratio = _hole_ratio(mask)

    assert ratio > 0.0


def test_overlapping_same_pixel_outliers_median_beats_mean() -> None:
    truth_z = np.zeros((3, 3), dtype=float)
    truth_mask = np.ones((3, 3), dtype=bool)
    observations: list[SubApertureObservation] = []
    for index, value in enumerate((10.0, 0.0, 0.0)):
        z = np.zeros((3, 3), dtype=float)
        z[1, 1] = value
        observations.append(
            SubApertureObservation(
                observation_id=f"obs-{index}",
                z=z,
                valid_mask=np.array(truth_mask, copy=True),
                tile_shape=(3, 3),
                center_xy=(1.0, 1.0),
                global_shape=(3, 3),
                rotation_deg=0.0,
            )
        )

    mean_reconstruction = baseline_integer_unshift_mean(tuple(observations))
    median_reconstruction = baseline_integer_unshift_median(tuple(observations))

    assert mean_reconstruction.z[1, 1] > median_reconstruction.z[1, 1]
    assert np.isclose(median_reconstruction.z[1, 1], truth_z[1, 1])


def test_baseline_mean_handles_nan_observation_without_crashing() -> None:
    observation = SubApertureObservation(
        observation_id="nan-obs",
        z=np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=float),
        valid_mask=np.array([[True, True], [True, True]], dtype=bool),
        tile_shape=(2, 2),
        center_xy=(0.5, 0.5),
        global_shape=(2, 2),
        rotation_deg=0.0,
    )

    reconstruction = baseline_integer_unshift_mean((observation,))

    assert reconstruction.z.shape == (2, 2)
    assert np.isnan(reconstruction.z[1, 0])
