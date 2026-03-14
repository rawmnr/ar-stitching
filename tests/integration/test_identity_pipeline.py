from pathlib import Path

import numpy as np

from stitching.editable.baseline import baseline_identity
from stitching.contracts import ScenarioConfig
from stitching.harness.run_eval import run_identity_eval
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_identity_scenario_is_accepted() -> None:
    report = run_identity_eval(Path("scenarios/s00_identity.yaml"))

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

    reconstruction = baseline_identity(observations)

    assert reconstruction.source_observation_ids == (observations[0].observation_id,)
    assert (observations[0].z == original_z).all()
    assert (observations[0].valid_mask == original_mask).all()
