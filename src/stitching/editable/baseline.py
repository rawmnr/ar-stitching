"""Editable baselines kept intentionally simple during repository foundation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stitching.contracts import ReconstructionSurface, SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices


def _validate_observation_list(observations: Iterable[SubApertureObservation]) -> list[SubApertureObservation]:
    """Materialize and validate a non-empty observation list."""

    observation_list = list(observations)
    if not observation_list:
        raise ValueError("At least one observation is required for reconstruction.")
    return observation_list


def _place_observations(
    observation_list: list[SubApertureObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[tuple[float, float]]]:
    """Place local tiles on the global grid and collect stacked samples."""

    global_shape = observation_list[0].global_shape
    max_observations = len(observation_list)
    stacked = np.full((max_observations, *global_shape), np.nan, dtype=float)
    observed_support_mask = np.zeros(global_shape, dtype=bool)
    source_observation_ids: list[str] = []
    centers_xy: list[tuple[float, float]] = []

    for index, observation in enumerate(observation_list):
        if observation.global_shape != global_shape:
            raise ValueError("All observations must share the same global_shape.")
        global_y, global_x, local_y, local_x = placement_slices(
            observation.global_shape,
            observation.tile_shape,
            observation.center_xy,
        )
        local_z = np.array(observation.z, copy=True)[local_y, local_x]
        local_mask = np.array(observation.valid_mask, copy=True)[local_y, local_x]
        stack_view = stacked[index, global_y, global_x]
        support_view = observed_support_mask[global_y, global_x]
        stack_view[local_mask] = local_z[local_mask]
        support_view[local_mask] = True
        source_observation_ids.append(observation.observation_id)
        centers_xy.append(observation.center_xy)

    return stacked, observed_support_mask, np.any(np.isfinite(stacked), axis=0), source_observation_ids, centers_xy


def _reconstruction_metadata(
    observation_list: list[SubApertureObservation],
    baseline_name: str,
    centers_xy: list[tuple[float, float]],
) -> dict[str, object]:
    """Build shared reconstruction metadata."""

    first = observation_list[0]
    return {
        "baseline": baseline_name,
        "reconstruction_frame": "global_truth",
        "tile_centers_xy": centers_xy[0] if len(centers_xy) == 1 else tuple(centers_xy),
        "num_observations_used": len(observation_list),
        **dict(first.metadata),
    }


def baseline_integer_unshift_mean(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Reconstruct a global surface by placing local tiles into the global frame."""

    observation_list = _validate_observation_list(observations)
    stacked, observed_support_mask, valid_mask, source_observation_ids, centers_xy = _place_observations(observation_list)
    finite_mask = np.isfinite(stacked)
    sum_z = np.nansum(stacked, axis=0)
    count = np.sum(finite_mask, axis=0)
    z = np.zeros(global_shape := observation_list[0].global_shape, dtype=float)
    z[valid_mask] = sum_z[valid_mask] / count[valid_mask]

    return ReconstructionSurface(
        z=z,
        valid_mask=valid_mask,
        source_observation_ids=tuple(source_observation_ids),
        observed_support_mask=observed_support_mask,
        metadata=_reconstruction_metadata(observation_list, "integer_tile_place_mean", centers_xy),
    )


def baseline_integer_unshift_median(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Reconstruct a global surface by placing local tiles and taking the median on overlaps."""

    observation_list = _validate_observation_list(observations)
    stacked, observed_support_mask, valid_mask, source_observation_ids, centers_xy = _place_observations(observation_list)
    z = np.zeros(global_shape := observation_list[0].global_shape, dtype=float)
    z[valid_mask] = np.nanmedian(stacked[:, valid_mask], axis=0)

    return ReconstructionSurface(
        z=z,
        valid_mask=valid_mask,
        source_observation_ids=tuple(source_observation_ids),
        observed_support_mask=observed_support_mask,
        metadata=_reconstruction_metadata(observation_list, "integer_tile_place_median", centers_xy),
    )


def baseline_identity(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Backward-compatible alias for the current integer-unshift mean baseline."""

    return baseline_integer_unshift_mean(observations)
