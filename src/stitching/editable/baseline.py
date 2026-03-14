"""Editable baselines kept intentionally simple during repository foundation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stitching.contracts import ReconstructionSurface, SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices


def baseline_integer_unshift_mean(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Reconstruct a global surface by placing local tiles into the global frame."""

    observation_list = list(observations)
    if not observation_list:
        raise ValueError("At least one observation is required for reconstruction.")

    global_shape = observation_list[0].global_shape
    sum_z = np.zeros(global_shape, dtype=float)
    hit_count = np.zeros(global_shape, dtype=int)
    source_observation_ids: list[str] = []
    centers_xy: list[tuple[float, float]] = []

    for observation in observation_list:
        global_y, global_x, local_y, local_x = placement_slices(
            observation.global_shape,
            observation.tile_shape,
            observation.center_xy,
        )
        local_z = np.array(observation.z, copy=True)[local_y, local_x]
        local_mask = np.array(observation.valid_mask, copy=True)[local_y, local_x]

        sum_z_view = sum_z[global_y, global_x]
        hit_count_view = hit_count[global_y, global_x]
        sum_z_view[local_mask] += local_z[local_mask]
        hit_count_view[local_mask] += 1
        sum_z[global_y, global_x] = sum_z_view
        hit_count[global_y, global_x] = hit_count_view
        source_observation_ids.append(observation.observation_id)
        centers_xy.append(observation.center_xy)

    valid_mask = hit_count > 0
    z = np.zeros(global_shape, dtype=float)
    z[valid_mask] = sum_z[valid_mask] / hit_count[valid_mask]

    first = observation_list[0]
    return ReconstructionSurface(
        z=z,
        valid_mask=valid_mask,
        source_observation_ids=tuple(source_observation_ids),
        metadata={
            "baseline": "integer_tile_place_mean",
            "reconstruction_frame": "global_truth",
            "tile_centers_xy": centers_xy[0] if len(centers_xy) == 1 else tuple(centers_xy),
            "num_observations_used": len(observation_list),
            **dict(first.metadata),
        },
    )


def baseline_identity(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Backward-compatible alias for the current integer-unshift mean baseline."""

    return baseline_integer_unshift_mean(observations)
