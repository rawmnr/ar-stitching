"""Unit tests for realistic metrology conditions (pupils, LF noise)."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_circular_detector_pupil_is_applied() -> None:
    config = ScenarioConfig(
        scenario_id="circular_detector",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(16, 16),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={"detector_pupil": "circular", "detector_radius_fraction": 0.4}
    )
    _, observations = simulate_identity_observations(config)
    
    obs = observations[0]
    # Corners of a 16x16 tile should be invalid in a circular pupil
    assert obs.valid_mask[0, 0] == False
    assert obs.valid_mask[15, 15] == False
    assert obs.valid_mask[8, 8] == True # Center should be valid


def test_low_frequency_noise_is_not_zero() -> None:
    config = ScenarioConfig(
        scenario_id="lf_noise",
        description="test",
        grid_shape=(32, 32),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
        metadata={"low_frequency_noise_std": 0.1}
    )
    truth, observations = simulate_identity_observations(config)
    
    # Without LF noise, observation 0 should match truth exactly for integer offset
    # With LF noise, they should differ
    diff = observations[0].z - truth.z
    assert np.any(diff != 0.0)
    # The noise should be relatively smooth (low frequency)
    # Check that gradient is not as high as pixel-level white noise
    gy, gx = np.gradient(diff)
    assert np.max(np.abs(gx)) < 1.0 


def test_ignore_piston_in_eval() -> None:
    from stitching.trusted.eval.metrics import signal_metrics
    
    ref = np.ones((10, 10))
    cand = np.ones((10, 10)) + 5.0 # Pure piston error
    mask = np.ones((10, 10), dtype=bool)
    
    # Default: MAE/RMS should be 5.0
    sig_default = signal_metrics(ref, cand, mask, ignore_piston=False)
    assert np.isclose(sig_default["rms_on_valid_intersection"], 5.0)
    
    # Ignore piston: MAE/RMS should be 0.0
    sig_ignored = signal_metrics(ref, cand, mask, ignore_piston=True)
    assert np.isclose(sig_ignored["rms_on_valid_intersection"], 0.0)
