"""Unit tests for Digital Twin high-fidelity features."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig
from stitching.trusted.eval.mismatch import compute_mismatch_map
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_optical_psf_smooths_surface() -> None:
    config_no_psf = ScenarioConfig(
        scenario_id="no_psf",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={"mid_spatial_ripple_std": 0.5} # Add ripples to smooth
    )
    truth_no_psf, obs_no_psf = simulate_identity_observations(config_no_psf)
    
    config_psf = ScenarioConfig(**{**config_no_psf.__dict__, "scenario_id": "psf", "metadata": {"mid_spatial_ripple_std": 0.5, "optical_psf_sigma": 1.5}})
    truth_psf, obs_psf = simulate_identity_observations(config_psf)
    
    # PSF is an instrument effect, not a piece change.
    assert np.array_equal(truth_psf.z, truth_no_psf.z, equal_nan=True)
    # PSF should reduce the standard deviation of the ripples in the observations
    assert np.std(obs_psf[0].z) < np.std(obs_no_psf[0].z)
    assert np.isfinite(obs_psf[0].z[obs_psf[0].valid_mask]).all()


def test_surface_bending_drift_changes_observations_over_time() -> None:
    config = ScenarioConfig(
        scenario_id="bending",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0), (0.0, 0.0)), # Two observations at same spot
        seed=0,
        metadata={"surface_bending_drift": 0.2}
    )
    _, observations = simulate_identity_observations(config)
    
    # First obs has 0 drift, second has full drift
    assert not np.allclose(observations[0].z, observations[1].z)


def test_mid_spatial_ripples_are_added() -> None:
    config = ScenarioConfig(
        scenario_id="ripples",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={"mid_spatial_ripple_std": 0.1}
    )
    truth, observations = simulate_identity_observations(config)
    
    baseline_config = ScenarioConfig(
        scenario_id="ripples_baseline",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={"mid_spatial_ripple_std": 0.0},
    )
    baseline_truth, _ = simulate_identity_observations(baseline_config)

    # Polishing marks belong to the piece and must therefore appear in the truth surface.
    truth_delta = truth.z - baseline_truth.z
    valid = truth.valid_mask & baseline_truth.valid_mask
    assert np.any(~np.isclose(truth_delta[valid], 0.0))
    assert np.isclose(np.std(truth_delta[valid]), 0.1, atol=0.05)
    assert np.allclose(observations[0].z[valid], truth.z[valid], atol=1e-12)

    # Horizontal/vertical stripe families are separable, so mixed variation
    # should stay small compared with axial variation.
    mixed = truth_delta[1:, 1:] - truth_delta[:-1, 1:] - truth_delta[1:, :-1] + truth_delta[:-1, :-1]
    grad_x = truth_delta[:, 1:] - truth_delta[:, :-1]
    grad_y = truth_delta[1:, :] - truth_delta[:-1, :]
    axial_scale = max(float(np.std(grad_x[valid[:, 1:] & valid[:, :-1]])), float(np.std(grad_y[valid[1:, :] & valid[:-1, :]])))
    mixed_scale = float(np.std(mixed[valid[1:, 1:] & valid[:-1, 1:] & valid[1:, :-1] & valid[:-1, :-1]]))
    assert mixed_scale < 0.35 * axial_scale


def test_high_fidelity_subpixel_circular_observations_keep_finite_signal() -> None:
    config = ScenarioConfig(
        scenario_id="digital_twin_finite",
        description="test",
        grid_shape=(64, 64),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=((-10.5, -10.5), (10.5, -10.5), (-10.5, 10.5), (10.5, 10.5)),
        seed=16,
        metadata={
            "truth_basis": "zernike",
            "truth_coefficients": [0.0, 0.0, 0.0, 0.2, 0.3, -0.4, 0.1, -0.2],
            "detector_pupil": "circular",
            "mid_spatial_ripple_std": 0.02,
            "low_frequency_noise_std": 0.05,
            "realized_pose_drift_std": 0.01,
            "interpolation_order": 3,
        },
    )
    _, observations = simulate_identity_observations(config)
    std_map, count_z = compute_mismatch_map(observations)

    assert all(np.isfinite(obs.z[obs.valid_mask]).all() for obs in observations)
    assert np.count_nonzero(count_z > 1) > 0
    assert np.isfinite(std_map[count_z > 1]).all()
