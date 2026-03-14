"""Validation helpers for trusted simulator and evaluator contracts."""

from __future__ import annotations

import numpy as np

from stitching.contracts import ReconstructionSurface, SubApertureObservation


def validate_surface_alignment(z: np.ndarray, valid_mask: np.ndarray) -> None:
    """Enforce shared shape and zero-outside-mask invariants."""

    if z.shape != valid_mask.shape:
        raise ValueError("Values and valid_mask must have identical shapes.")
    if valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean.")
    if not np.all(z[~valid_mask] == 0.0):
        raise ValueError("Values outside valid_mask must be zero.")


def validate_observation_alignment(observation: SubApertureObservation) -> None:
    """Enforce shape and mask/value alignment for trusted observations."""

    validate_surface_alignment(observation.z, observation.valid_mask)


def validate_reconstruction_alignment(reconstruction: ReconstructionSurface) -> None:
    """Enforce shape and mask/value alignment for reconstruction candidates."""

    validate_surface_alignment(reconstruction.z, reconstruction.valid_mask)
