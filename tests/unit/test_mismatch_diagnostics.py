"""Unit tests for overlap mismatch diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import SubApertureObservation
from stitching.trusted.eval.mismatch import compute_mismatch_metrics


def test_mismatch_metrics_zero_for_identical_observations() -> None:
    z = np.ones((5, 5), dtype=float)
    mask = np.ones((5, 5), dtype=bool)
    obs1 = SubApertureObservation(
        observation_id="obs-1",
        z=z,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.0, 2.0),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )
    obs2 = SubApertureObservation(
        observation_id="obs-2",
        z=z,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.0, 2.0),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )

    metrics = compute_mismatch_metrics((obs1, obs2))

    assert metrics["mismatch_rms"] == 0.0
    assert metrics["mismatch_max"] == 0.0


def test_mismatch_metrics_detects_dc_difference() -> None:
    z1 = np.ones((5, 5), dtype=float)
    z2 = np.ones((5, 5), dtype=float) + 1.0
    mask = np.ones((5, 5), dtype=bool)
    obs1 = SubApertureObservation(
        observation_id="obs-1",
        z=z1,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.0, 2.0),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )
    obs2 = SubApertureObservation(
        observation_id="obs-2",
        z=z2,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.0, 2.0),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )

    metrics = compute_mismatch_metrics((obs1, obs2))

    # Variance of (1.0, 2.0) is 0.25. Std dev is 0.5.
    assert np.isclose(metrics["mismatch_rms"], 0.5)
    assert np.isclose(metrics["mismatch_mean"], 0.5)


def test_mismatch_metrics_handles_no_overlap() -> None:
    z = np.ones((5, 5), dtype=float)
    mask = np.ones((5, 5), dtype=bool)
    obs1 = SubApertureObservation(
        observation_id="obs-1",
        z=z,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.0, 2.0),
        global_shape=(10, 10),
        rotation_deg=0.0,
    )
    obs2 = SubApertureObservation(
        observation_id="obs-2",
        z=z,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(7.0, 7.0),
        global_shape=(10, 10),
        rotation_deg=0.0,
    )

    metrics = compute_mismatch_metrics((obs1, obs2))

    assert metrics["mismatch_rms"] == 0.0


def test_mismatch_metrics_with_subpixel_shifts_uses_binning() -> None:
    z = np.ones((5, 5), dtype=float)
    mask = np.ones((5, 5), dtype=bool)
    # obs1 at (2.0, 2.0), obs2 at (2.4, 2.4).
    # Both should bin to (2.0, 2.0) if using _round_to_compatible_center
    # For odd tile_shape 5x5, compatible centers are integers.
    obs1 = SubApertureObservation(
        observation_id="obs-1",
        z=z,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.0, 2.0),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )
    obs2 = SubApertureObservation(
        observation_id="obs-2",
        z=z + 1.0,
        valid_mask=mask,
        tile_shape=(5, 5),
        center_xy=(2.4, 2.4),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )

    metrics = compute_mismatch_metrics((obs1, obs2))

    assert metrics["mismatch_rms"] > 0.0
    assert np.isclose(metrics["mismatch_rms"], 0.5)
