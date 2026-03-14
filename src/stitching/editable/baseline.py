"""Editable baselines kept intentionally simple during repository foundation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stitching.contracts import ReconstructionSurface, SubApertureObservation
from stitching.trusted.scan.transforms import apply_integer_shift


def _integer_inverse_shift(translation_xy: tuple[float, float]) -> tuple[int, int]:
    """Convert detector translation metadata into the inverse integer shift."""

    dx = int(round(translation_xy[0]))
    dy = int(round(translation_xy[1]))
    return -dx, -dy


def baseline_integer_unshift_mean(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Reconstruct a global surface by inverse-shifting and averaging observations."""

    observation_list = list(observations)
    if not observation_list:
        raise ValueError("At least one observation is required for reconstruction.")

    shape = observation_list[0].z.shape
    sum_z = np.zeros(shape, dtype=float)
    hit_count = np.zeros(shape, dtype=int)
    source_observation_ids: list[str] = []
    inverse_shifts: list[tuple[int, int]] = []

    for observation in observation_list:
        inverse_shift = _integer_inverse_shift(observation.translation_xy)
        shifted_z = apply_integer_shift(np.array(observation.z, copy=True), inverse_shift)
        shifted_mask = apply_integer_shift(np.array(observation.valid_mask, copy=True).astype(np.uint8), inverse_shift).astype(bool)

        sum_z[shifted_mask] += shifted_z[shifted_mask]
        hit_count[shifted_mask] += 1
        source_observation_ids.append(observation.observation_id)
        inverse_shifts.append(inverse_shift)

    valid_mask = hit_count > 0
    z = np.zeros(shape, dtype=float)
    z[valid_mask] = sum_z[valid_mask] / hit_count[valid_mask]

    first = observation_list[0]
    return ReconstructionSurface(
        z=z,
        valid_mask=valid_mask,
        source_observation_ids=tuple(source_observation_ids),
        metadata={
            "baseline": "integer_unshift_mean",
            "reconstruction_frame": "global_truth",
            "inverse_shift_xy": inverse_shifts[0] if len(inverse_shifts) == 1 else tuple(inverse_shifts),
            "num_observations_used": len(observation_list),
            **dict(first.metadata),
        },
    )


def baseline_identity(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Backward-compatible alias for the current integer-unshift mean baseline."""

    return baseline_integer_unshift_mean(observations)
