from __future__ import annotations

from stitching.contracts import ScenarioConfig
from stitching.trusted.eval.metrics import signal_metrics
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_increasing_gaussian_noise_does_not_improve_rms() -> None:
    base_config = ScenarioConfig(
        scenario_id="noise_base",
        description="noise base",
        grid_shape=(11, 11),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=5,
    )
    noisy_config = ScenarioConfig(
        scenario_id="noise_high",
        description="noise high",
        grid_shape=(11, 11),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        gaussian_noise_std=0.1,
        seed=5,
    )

    truth, clean_obs = simulate_identity_observations(base_config)
    _, noisy_obs = simulate_identity_observations(noisy_config)

    clean_metrics = signal_metrics(truth.z, clean_obs[0].z, truth.valid_mask & clean_obs[0].valid_mask)
    noisy_metrics = signal_metrics(truth.z, noisy_obs[0].z, truth.valid_mask & noisy_obs[0].valid_mask)

    assert noisy_metrics["rms_on_valid_intersection"] >= clean_metrics["rms_on_valid_intersection"]
    assert noisy_metrics["mae_on_valid_intersection"] >= clean_metrics["mae_on_valid_intersection"]


def test_zero_bias_noise_outliers_and_retrace_match_identity_output() -> None:
    identity_config = ScenarioConfig(
        scenario_id="identity",
        description="identity",
        grid_shape=(9, 9),
        pixel_size=1.0,
        scan_offsets=((1.0, 1.0),),
        seed=3,
    )
    explicit_zero_config = ScenarioConfig(
        scenario_id="zeros",
        description="zeros",
        grid_shape=(9, 9),
        pixel_size=1.0,
        scan_offsets=((1.0, 1.0),),
        reference_bias=0.0,
        gaussian_noise_std=0.0,
        outlier_fraction=0.0,
        retrace_error=0.0,
        seed=3,
    )

    truth_identity, identity_obs = simulate_identity_observations(identity_config)
    truth_zero, zero_obs = simulate_identity_observations(explicit_zero_config)

    assert (truth_identity.valid_mask == truth_zero.valid_mask).all()
    assert (identity_obs[0].valid_mask == zero_obs[0].valid_mask).all()
    assert (identity_obs[0].z == zero_obs[0].z).all()
