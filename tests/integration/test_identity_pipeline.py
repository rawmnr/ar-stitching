from pathlib import Path

import numpy as np

from stitching.editable.baseline import baseline_integer_unshift_mean
from stitching.contracts import ScenarioConfig
from stitching.harness.run_eval import run_baseline_eval, run_identity_eval
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
    assert observations[0].valid_mask.sum() < observations[0].valid_mask.size
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


def test_shift_only_baseline_remains_rejected_due_to_shared_canvas_clipping() -> None:
    shifted_report = run_baseline_eval(Path("scenarios/s01_shift_only.yaml"))
    identity_report = run_baseline_eval(Path("scenarios/s00_identity.yaml"))

    assert shifted_report.accepted is False
    assert shifted_report.signal_metrics["mae_on_valid_intersection"] == 0.0
    assert shifted_report.geometry_metrics["footprint_iou"] < identity_report.geometry_metrics["footprint_iou"]
    assert shifted_report.geometry_metrics["valid_pixel_recall"] < identity_report.geometry_metrics["valid_pixel_recall"]


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
    single_shift_report = run_baseline_eval(Path("scenarios/s01_shift_only.yaml"))
    multi_overlap_report = run_baseline_eval(Path("scenarios/s06_multi_overlap.yaml"))

    assert multi_overlap_report.accepted is True
    assert multi_overlap_report.geometry_metrics["footprint_iou"] > single_shift_report.geometry_metrics["footprint_iou"]
    assert multi_overlap_report.geometry_metrics["valid_pixel_recall"] > single_shift_report.geometry_metrics["valid_pixel_recall"]


def test_multi_observation_baseline_uses_all_source_observations() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s06_multi_overlap.yaml"))
    _, observations = simulate_identity_observations(config)

    reconstruction = baseline_integer_unshift_mean(observations)

    assert reconstruction.source_observation_ids == tuple(observation.observation_id for observation in observations)
    assert reconstruction.metadata["num_observations_used"] == len(observations)
