"""Trusted nuisance and noise hooks with deterministic seed control."""

from __future__ import annotations

import numpy as np


def apply_nuisance_terms(z: np.ndarray, nuisance_terms: dict[str, float] | None = None) -> np.ndarray:
    """Apply simple additive nuisance terms by summing scalar contributions."""

    result = np.asarray(z, dtype=float).copy()
    for value in (nuisance_terms or {}).values():
        result = result + float(value)
    return result


def add_gaussian_noise(z: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Inject zero-mean Gaussian noise."""

    if std == 0.0:
        return np.asarray(z, dtype=float).copy()
    rng = np.random.default_rng(seed)
    return np.asarray(z, dtype=float) + rng.normal(0.0, std, size=z.shape)


def add_outliers(z: np.ndarray, fraction: float, magnitude: float, seed: int) -> np.ndarray:
    """Inject sparse additive outliers at random pixels."""

    result = np.asarray(z, dtype=float).copy()
    if fraction <= 0.0:
        return result
    rng = np.random.default_rng(seed)
    count = int(round(result.size * fraction))
    if count == 0:
        return result
    flat_indices = rng.choice(result.size, size=count, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    result.flat[flat_indices] += magnitude * signs
    return result


def apply_retrace_error(z: np.ndarray, magnitude: float) -> np.ndarray:
    """Hook for retrace-like systematics. Currently a no-op plus optional scalar drift."""

    return np.asarray(z, dtype=float) + magnitude
