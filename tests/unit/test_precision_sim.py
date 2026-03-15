"""Unit tests for high-precision simulation features (bicubic, fine rotation, retrace)."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_bicubic_interpolation_is_smoother_than_bilinear() -> None:
    # Create a truth surface with some curvature
    config_base = ScenarioConfig(
        scenario_id="interp_test",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.5, 0.5),), # Force interpolation
        seed=0,
    )
    
    # Bilinear (order 1)
    config_bilinear = ScenarioConfig(**{**config_base.__dict__, "metadata": {"interpolation_order": 1}})
    _, obs_bilinear = simulate_identity_observations(config_bilinear)
    
    # Bicubic (order 3)
    config_bicubic = ScenarioConfig(**{**config_base.__dict__, "metadata": {"interpolation_order": 3}})
    _, obs_bicubic = simulate_identity_observations(config_bicubic)
    
    # They should differ
    assert not np.allclose(obs_bilinear[0].z, obs_bicubic[0].z)


def test_fine_rotation_is_supported() -> None:
    config = ScenarioConfig(
        scenario_id="fine_rot",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        rotation_deg=(12.5,), # Non-90 degree rotation
        seed=0,
    )
    # This should no longer raise ValueError
    truth, observations = simulate_identity_observations(config)
    assert observations[0].rotation_deg == 12.5
    # Since it's rotated, it shouldn't match the center of the truth exactly (pixel-wise)
    center_z_truth = truth.z[8:24, 8:24]
    assert not np.allclose(observations[0].z, center_z_truth)


def test_geometric_retrace_distorts_surface() -> None:
    config_no_retrace = ScenarioConfig(
        scenario_id="no_retrace",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
    )
    _, obs_clean = simulate_identity_observations(config_no_retrace)
    
    config_retrace = ScenarioConfig(
        scenario_id="geom_retrace",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={
            "geometric_retrace_error": 2.0, # Large shift for visibility
            "truth_coefficients": [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]] # Non-flat surface
        }
    )
    _, obs_distorted = simulate_identity_observations(config_retrace)
    
    # Geometric retrace should change the values by shifting the sampling locations
    assert not np.allclose(obs_clean[0].z, obs_distorted[0].z, atol=1e-8)
