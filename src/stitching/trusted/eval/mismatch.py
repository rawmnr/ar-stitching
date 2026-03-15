"""Trusted mismatch diagnostics for overlap regions."""

from __future__ import annotations

import numpy as np

from stitching.contracts import SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices


def _round_to_compatible_center(center: float, tile_extent: int) -> float:
    """Round a center to the nearest value compatible with integer array placement."""

    origin = center - (tile_extent - 1) / 2.0
    return float(round(origin) + (tile_extent - 1) / 2.0)


def compute_mismatch_map(
    observations: tuple[SubApertureObservation, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw per-pixel standard deviation map and overlap count.

    Mismatch is standard deviation across valid contributions.
    Pixels with count <= 1 are zero in the std_map.
    """

    if not observations:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=int)

    global_shape = observations[0].global_shape
    sum_z = np.zeros(global_shape, dtype=float)
    sum_z2 = np.zeros(global_shape, dtype=float)
    count_z = np.zeros(global_shape, dtype=int)

    for obs in observations:
        compat_center_x = _round_to_compatible_center(obs.center_xy[0], obs.tile_shape[1])
        compat_center_y = _round_to_compatible_center(obs.center_xy[1], obs.tile_shape[0])

        try:
            gy, gx, ly, lx = placement_slices(global_shape, obs.tile_shape, (compat_center_x, compat_center_y))
            z_view = obs.z[ly, lx]
            m_view = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]

            sum_z[gy, gx][m_view] += z_view[m_view]
            sum_z2[gy, gx][m_view] += z_view[m_view] ** 2
            count_z[gy, gx][m_view] += 1
        except (ValueError, IndexError):
            continue

    overlap_mask = count_z > 1
    std_map = np.zeros(global_shape, dtype=float)

    if np.any(overlap_mask):
        mean_z = sum_z[overlap_mask] / count_z[overlap_mask]
        mean_z2 = sum_z2[overlap_mask] / count_z[overlap_mask]
        variance = np.maximum(0.0, mean_z2 - mean_z**2)
        std_map[overlap_mask] = np.sqrt(variance)

    return std_map, count_z


def compute_mismatch_metrics(
    observations: tuple[SubApertureObservation, ...],
) -> dict[str, float]:
    """Compute statistics on the disagreement between overlapping observations."""

    std_map, count_z = compute_mismatch_map(observations)
    overlap_mask = count_z > 1

    if not np.any(overlap_mask):
        return {
            "mismatch_rms": 0.0,
            "mismatch_mean": 0.0,
            "mismatch_median": 0.0,
            "mismatch_max": 0.0,
            "mismatch_p95": 0.0,
        }

    valid_std = std_map[overlap_mask]
    return {
        "mismatch_rms": float(np.sqrt(np.mean(valid_std**2))),
        "mismatch_mean": float(np.mean(valid_std)),
        "mismatch_median": float(np.median(valid_std)),
        "mismatch_max": float(np.max(valid_std)),
        "mismatch_p95": float(np.percentile(valid_std, 95)),
    }
