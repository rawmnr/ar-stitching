"""Trusted instrument-reference bias placeholders."""

from __future__ import annotations

import numpy as np


def stationary_reference_bias(shape: tuple[int, int], bias: float) -> np.ndarray:
    """Create a detector-frame stationary bias field."""

    return np.full(shape, bias, dtype=float)


def generate_reference_bias_field(
    shape: tuple[int, int],
    coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a static field-dependent reference bias from Zernike coefficients."""
    if coefficients is None or len(coefficients) == 0:
        return np.zeros(shape, dtype=float)
    
    from stitching.trusted.bases.zernike import generate_zernike_surface
    # We use the internal backend to generate the Zernike surface on the tile shape
    bias_field = generate_zernike_surface(
        coefficients, 
        shape, 
        indexing="noll", 
        backend="internal"
    )
    return np.asarray(bias_field, dtype=float)


def reference_bias_for_observation(
    base_bias: float,
    observation_index: int,
    metadata: dict[str, object] | None = None,
) -> float:
    """Return the total scalar detector bias for one observation.

    The foundation model keeps detector-frame bias scalar, but allows a simple
    time-varying drift sequence on top of the stationary component.
    """

    metadata = metadata or {}
    total_bias = float(base_bias)
    if "reference_bias_values" in metadata:
        values = tuple(float(value) for value in metadata["reference_bias_values"])  # type: ignore[index]
        if observation_index >= len(values):
            raise ValueError("reference_bias_values must provide one bias per observation.")
        total_bias += values[observation_index]
    elif "reference_bias_drift_step" in metadata:
        total_bias += float(metadata["reference_bias_drift_step"]) * float(observation_index)
    elif "reference_bias_drift" in metadata:
        total_bias += float(metadata["reference_bias_drift"])
    return total_bias


def apply_reference_bias(z: np.ndarray, bias: float | np.ndarray) -> np.ndarray:
    """Add a detector-frame reference bias (scalar or field)."""

    return np.asarray(z, dtype=float) + bias
