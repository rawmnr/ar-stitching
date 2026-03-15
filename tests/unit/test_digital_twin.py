"""Unit tests for Digital Twin high-fidelity features."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig
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
    
    # PSF should reduce the standard deviation of the ripples in the observations
    assert np.std(obs_psf[0].z) < np.std(obs_no_psf[0].z)


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
    
    # Difference should show periodic ripples
    diff = observations[0].z - truth.z
    assert np.any(diff != 0.0)
    assert np.isclose(np.std(diff), 0.1, atol=0.05)
