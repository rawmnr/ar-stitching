"""Validation helpers for trusted simulator and evaluator contracts."""

from __future__ import annotations

import numpy as np

from stitching.contracts import ReconstructionSurface, SubApertureObservation


ZERO_OUTSIDE_MASK_ATOL = 1e-12


def validate_surface_alignment(z: np.ndarray, valid_mask: np.ndarray) -> None:
    """Enforce shared shape and zero-outside-mask invariants."""

    if z.shape != valid_mask.shape:
        raise ValueError("Values and valid_mask must have identical shapes.")
    if valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean.")
    if not np.allclose(z[~valid_mask], 0.0, atol=ZERO_OUTSIDE_MASK_ATOL, rtol=0.0):
        raise ValueError("Values outside valid_mask must be zero.")


def validate_observation_alignment(observation: SubApertureObservation) -> None:
    """Enforce shape and mask/value alignment for trusted observations."""

    validate_surface_alignment(observation.z, observation.valid_mask)
    if observation.z.shape != observation.tile_shape:
        raise ValueError("Observation tile_shape must match observation array shape.")
    if observation.global_shape[0] < observation.tile_shape[0] or observation.global_shape[1] < observation.tile_shape[1]:
        raise ValueError("Observation tile_shape must fit inside global_shape.")


def validate_reconstruction_alignment(reconstruction: ReconstructionSurface) -> None:
    """Enforce shape and mask/value alignment for reconstruction candidates."""

    validate_surface_alignment(reconstruction.z, reconstruction.valid_mask)
    if reconstruction.observed_support_mask is not None:
        validate_surface_alignment(
            np.zeros_like(reconstruction.observed_support_mask, dtype=float),
            reconstruction.observed_support_mask,
        )
        if reconstruction.observed_support_mask.shape != reconstruction.valid_mask.shape:
            raise ValueError("Reconstruction observed_support_mask must match reconstruction shape.")
