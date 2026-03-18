"""Unit tests for field-dependent instrument reference bias."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_zernike_reference_bias_is_applied() -> None:
    # We use a simple flat truth (zero coefficients) to easily see the bias
    # Z4 = 1.0 (Defocus)
    bias_coeffs = [0.0, 0.0, 0.0, 1.0] 
    
    config = ScenarioConfig(
        scenario_id="bias_test",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={
            "truth_basis": "legendre",
            "truth_coefficients": [[0.0]], # Flat (2D array)
            "reference_bias_coefficients": bias_coeffs
        }
    )
    
    truth, observations = simulate_identity_observations(config)
    obs = observations[0]
    
    # Difference should be exactly the Zernike field
    diff = obs.z - truth.z
    
    # Check that it's not a scalar (unless only Z1 is provided)
    assert np.nanstd(diff) > 0.0
    
    # Check shape
    assert diff.shape == config.tile_shape
    
    # Verify values at center vs edge (defocus characteristic)
    # For defocus on unit circle, center is usually -1 or 1 relative to edge
    center_val = diff[16, 16]
    edge_val = diff[0, 16]
    assert not np.isclose(center_val, edge_val)

def test_scalar_reference_bias_still_works() -> None:
    config = ScenarioConfig(
        scenario_id="scalar_bias",
        description="test",
        grid_shape=(16, 16),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        reference_bias=5.0,
        seed=0
    )
    truth, observations = simulate_identity_observations(config)
    diff = observations[0].z - truth.z
    assert np.all(np.isclose(diff, 5.0))
