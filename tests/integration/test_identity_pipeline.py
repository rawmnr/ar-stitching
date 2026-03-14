from pathlib import Path

import numpy as np

import pytest

from stitching.editable.baseline import baseline_integer_unshift_mean, baseline_integer_unshift_median
from stitching.contracts import ReconstructionSurface, ScenarioConfig
from stitching.harness.run_eval import run_baseline_eval, run_identity_eval
from stitching.trusted.eval.metrics import build_eval_report
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


def test_reference_bias_scenario_is_rejected_or_degraded() -> None:
    report = run_baseline_eval(Path("scenarios/s02_reference_bias.yaml"))

    assert report.accepted is False
    assert report.signal_metrics["mae_on_valid_intersection"] > 0.0


def test_noise_scenario_degrades_signal_metrics_against_identity() -> None:
    identity_report = run_baseline_eval(Path("scenarios/s00_identity.yaml"))
    noisy_report = run_baseline_eval(Path("scenarios/s03_noise.yaml"))

    assert noisy_report.accepted is False
    assert noisy_report.signal_metrics["rms_on_valid_intersection"] > identity_report.signal_metrics["rms_on_valid_intersection"]
    assert noisy_report.signal_metrics["mae_on_valid_intersection"] > identity_report.signal_metrics["mae_on_valid_intersection"]


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
