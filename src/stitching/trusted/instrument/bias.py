"""Trusted instrument-reference bias placeholders."""

from __future__ import annotations

import numpy as np


def stationary_reference_bias(shape: tuple[int, int], bias: float) -> np.ndarray:
    """Create a detector-frame stationary bias field."""

    return np.full(shape, bias, dtype=float)


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


def apply_reference_bias(z: np.ndarray, bias: float) -> np.ndarray:
    """Add a scalar detector-frame reference bias."""

    return np.asarray(z, dtype=float) + bias
