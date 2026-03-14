"""Trusted nuisance and noise hooks with deterministic seed control."""

from __future__ import annotations

import numpy as np


OUTLIER_SCALE_EPS = 1e-12


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


def outlier_magnitude_scale(z: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """Estimate a deterministic signal scale for relative outlier injection."""

    values = np.asarray(z, dtype=float)
    if valid_mask is not None:
        values = values[np.asarray(valid_mask, dtype=bool)]
    if values.size == 0:
        return 1.0
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if scale <= OUTLIER_SCALE_EPS:
        span = float(np.max(values) - np.min(values))
        scale = span / 2.0
    return max(1.0, scale)


def add_outliers(
    z: np.ndarray,
    fraction: float,
    magnitude: float,
    seed: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inject sparse additive outliers at random pixels."""

    result = np.asarray(z, dtype=float).copy()
    if fraction <= 0.0:
        return result
    rng = np.random.default_rng(seed)
    candidate_mask = np.ones(result.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    candidate_count = int(candidate_mask.sum())
    count = int(round(candidate_count * fraction))
    if count == 0:
        return result
    flat_indices = _sample_flat_indices(candidate_mask, count, rng)
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    result.flat[flat_indices] += outlier_magnitude_scale(result, candidate_mask) * float(magnitude) * signs
    return result


def apply_retrace_error(z: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply a simple surface-dependent retrace distortion."""

    result = np.asarray(z, dtype=float).copy()
    if magnitude == 0.0:
        return result
    centered = result - float(np.mean(result))
    return result + float(magnitude) * centered * np.abs(centered)


def _sample_flat_indices(candidate_mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample valid flat indices without always materializing all candidate indices."""

    flat_mask = candidate_mask.ravel()
    if np.all(flat_mask):
        return np.asarray(rng.choice(flat_mask.size, size=count, replace=False), dtype=np.intp)

    valid_count = int(flat_mask.sum())
    if valid_count == 0:
        return np.zeros(0, dtype=np.intp)

    # For sparse masks, sample valid ranks first, then map them through the mask.
    chosen_ranks = np.sort(np.asarray(rng.choice(valid_count, size=count, replace=False), dtype=np.int64))
    flat_indices = np.empty(count, dtype=np.intp)
    rank_index = 0
    seen_valid = 0
    for flat_index, is_valid in enumerate(flat_mask.tolist()):
        if not is_valid:
            continue
        while rank_index < count and chosen_ranks[rank_index] == seen_valid:
            flat_indices[rank_index] = flat_index
            rank_index += 1
        seen_valid += 1
        if rank_index == count:
            break
    return flat_indices
