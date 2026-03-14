"""Editable baselines kept intentionally simple during repository foundation."""

from __future__ import annotations

from collections import defaultdict
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
) -> tuple[np.ndarray, list[str], list[tuple[float, float]]]:
    """Validate shared placement metadata and collect support bookkeeping."""

    global_shape = observation_list[0].global_shape
    observed_support_mask = np.zeros(global_shape, dtype=bool)
    source_observation_ids: list[str] = []
    centers_xy: list[tuple[float, float]] = []

    for observation in observation_list:
        if observation.global_shape != global_shape:
            raise ValueError("All observations must share the same global_shape.")
        global_y, global_x, local_y, local_x = placement_slices(
            observation.global_shape,
            observation.tile_shape,
            observation.center_xy,
        )
        local_mask = np.array(observation.valid_mask, copy=True)[local_y, local_x]
        support_view = observed_support_mask[global_y, global_x]
        support_view[local_mask] = True
        source_observation_ids.append(observation.observation_id)
        centers_xy.append(observation.center_xy)

    return observed_support_mask, source_observation_ids, centers_xy


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


def _iter_placed_pixels(
    observation_list: list[SubApertureObservation],
):
    """Yield per-observation placement views for valid local pixels."""

    for observation in observation_list:
        global_y, global_x, local_y, local_x = placement_slices(
            observation.global_shape,
            observation.tile_shape,
            observation.center_xy,
        )
        local_z = np.array(observation.z, copy=False)[local_y, local_x]
        local_mask = np.array(observation.valid_mask, copy=False)[local_y, local_x]
        yield global_y, global_x, local_z, local_mask


def baseline_integer_unshift_mean(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Reconstruct a global surface by placing local tiles into the global frame."""

    observation_list = _validate_observation_list(observations)
    global_shape = observation_list[0].global_shape
    observed_support_mask, source_observation_ids, centers_xy = _place_observations(observation_list)
    sum_z = np.zeros(global_shape, dtype=float)
    count = np.zeros(global_shape, dtype=int)

    for global_y, global_x, local_z, local_mask in _iter_placed_pixels(observation_list):
        sum_view = sum_z[global_y, global_x]
        count_view = count[global_y, global_x]
        sum_view[local_mask] += local_z[local_mask]
        count_view[local_mask] += 1

    valid_mask = count > 0
    z = np.zeros(global_shape, dtype=float)
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
    global_shape = observation_list[0].global_shape
    observed_support_mask, source_observation_ids, centers_xy = _place_observations(observation_list)
    samples_by_pixel: defaultdict[int, list[float]] = defaultdict(list)

    for global_y, global_x, local_z, local_mask in _iter_placed_pixels(observation_list):
        global_rows, global_cols = np.nonzero(local_mask)
        for local_row, local_col in zip(global_rows.tolist(), global_cols.tolist(), strict=False):
            global_row = global_y.start + local_row
            global_col = global_x.start + local_col
            flat_index = np.ravel_multi_index((global_row, global_col), global_shape)
            samples_by_pixel[flat_index].append(float(local_z[local_row, local_col]))

    valid_mask = observed_support_mask
    z = np.zeros(global_shape, dtype=float)
    for flat_index, samples in samples_by_pixel.items():
        row, col = np.unravel_index(flat_index, global_shape)
        z[row, col] = float(np.median(np.asarray(samples, dtype=float)))

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
