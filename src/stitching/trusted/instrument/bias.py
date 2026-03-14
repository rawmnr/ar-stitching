"""Trusted instrument-reference bias placeholders."""

from __future__ import annotations

import numpy as np


def stationary_reference_bias(shape: tuple[int, int], bias: float) -> np.ndarray:
    """Create a detector-frame stationary bias field."""

    return np.full(shape, bias, dtype=float)


def apply_reference_bias(z: np.ndarray, bias: float) -> np.ndarray:
    """Add a simple scalar reference bias."""

    return np.asarray(z, dtype=float) + bias
