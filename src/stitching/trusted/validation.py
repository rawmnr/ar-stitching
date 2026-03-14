"""Validation helpers for trusted simulator and evaluator contracts."""

from __future__ import annotations

import numpy as np

from stitching.contracts import SubApertureObservation


def validate_observation_alignment(observation: SubApertureObservation) -> None:
    """Enforce shape and mask/value alignment for trusted observations."""

    if observation.z.shape != observation.valid_mask.shape:
        raise ValueError("Observation values and valid_mask must have identical shapes.")
    if observation.valid_mask.dtype != np.bool_:
        raise ValueError("Observation valid_mask must be boolean.")
    if not np.all(observation.z[~observation.valid_mask] == 0.0):
        raise ValueError("Observation values outside valid_mask must be zero.")
