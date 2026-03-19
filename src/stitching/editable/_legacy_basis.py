"""Shared helpers for legacy editable baselines."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import legval
from stitching.contracts import SubApertureObservation
from stitching.trusted.bases.zernike import generate_zernike_surface


def rounded_placement_slices(
    global_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[slice, slice, slice, slice]:
    """Return the same rounded placement slices used by the evaluator."""

    rows, cols = tile_shape
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])

    top = int(round(center_y - (rows - 1) / 2.0))
    left = int(round(center_x - (cols - 1) / 2.0))
    bottom = top + rows
    right = left + cols

    gy_s = max(0, top)
    gy_e = min(global_shape[0], bottom)
    gx_s = max(0, left)
    gx_e = min(global_shape[1], right)

    ly_s = max(0, -top)
    lx_s = max(0, -left)
    ly_e = ly_s + max(0, gy_e - gy_s)
    lx_e = lx_s + max(0, gx_e - gx_s)

    return slice(gy_s, gy_e), slice(gx_s, gx_e), slice(ly_s, ly_e), slice(lx_s, lx_e)


def observed_support_mask(
    observations: tuple[SubApertureObservation, ...],
    global_shape: tuple[int, int],
) -> np.ndarray:
    """Compute physical support from rounded tile placement and local valid masks."""

    support = np.zeros(global_shape, dtype=bool)
    for obs in observations:
        gy, gx, ly, lx = rounded_placement_slices(global_shape, obs.tile_shape, obs.center_xy)
        if gy.stop <= gy.start or gx.stop <= gx.start:
            continue
        local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
        support_view = support[gy, gx]
        support_view[local_mask] = True
    return support


def normalized_tile_coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return local coordinates normalized to [-1, 1]."""

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
    x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
    return y_norm, x_norm


def _legendre_power_pair(term: int) -> tuple[int, int]:
    """Match the legacy 2D Legendre enumeration used in the MATLAB code."""

    if term < 0:
        raise ValueError("Legendre term index must be non-negative.")

    i = 0
    degree = 0
    while True:
        for x_power in range(degree, -1, -1):
            y_power = degree - x_power
            if i == term:
                return x_power, y_power
            i += 1
        degree += 1


def _legendre_1d(power: int, coord: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(power + 1, dtype=float)
    coeffs[-1] = 1.0
    return legval(coord, coeffs)


def basis_term_stack(
    mode: str,
    terms: Iterable[int],
    shape: tuple[int, int],
    *,
    radius_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return basis samples stacked as (n_terms, rows, cols) and a mask."""

    mode_norm = mode.upper()
    term_list = [int(term) for term in terms]
    if not term_list:
        empty = np.zeros((0, shape[0], shape[1]), dtype=float)
        return empty, np.zeros(shape, dtype=bool)

    if mode_norm == "Z":
        stack = []
        base_mask = None
        for term in term_list:
            coeffs = np.zeros(term + 1, dtype=float)
            coeffs[term] = 1.0
            sampled = generate_zernike_surface(
                coeffs,
                shape,
                indexing="fringe",
                backend="internal",
                radius_fraction=radius_fraction,
                fill_value=np.nan,
            )
            if base_mask is None:
                base_mask = np.isfinite(sampled)
            stack.append(np.where(np.isfinite(sampled), sampled, 0.0))
        return np.stack(stack, axis=0), base_mask if base_mask is not None else np.zeros(shape, dtype=bool)

    if mode_norm != "L":
        raise ValueError(f"Unsupported legacy basis mode '{mode}'.")

    y_norm, x_norm = normalized_tile_coords(shape)
    stack = []
    for term in term_list:
        x_power, y_power = _legendre_power_pair(term)
        basis = _legendre_1d(x_power, x_norm) * _legendre_1d(y_power, y_norm)
        stack.append(basis)
    return np.stack(stack, axis=0), np.ones(shape, dtype=bool)


def fit_basis_coefficients(
    data: np.ndarray,
    basis_stack: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Least-squares fit of basis coefficients on masked pixels."""

    if basis_stack.size == 0 or not np.any(mask):
        return np.zeros((basis_stack.shape[0],), dtype=float)

    design = basis_stack[:, mask].T
    target = np.asarray(data, dtype=float)[mask]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coeffs


def evaluate_basis_surface(
    coeffs: np.ndarray,
    basis_stack: np.ndarray,
) -> np.ndarray:
    """Reconstruct a surface from a basis stack and coefficient vector."""

    if basis_stack.size == 0:
        return np.zeros((0, 0), dtype=float)
    return np.tensordot(np.asarray(coeffs, dtype=float), basis_stack, axes=(0, 0))


def remove_low_order_modes(
    surface: np.ndarray,
    mask: np.ndarray,
    mode: str,
    terms: Iterable[int],
    *,
    radius_fraction: float | None = None,
) -> np.ndarray:
    """Project out low-order basis modes from a detector-fixed map."""

    basis_stack, basis_mask = basis_term_stack(mode, terms, surface.shape, radius_fraction=radius_fraction)
    valid_mask = np.asarray(mask, dtype=bool) & basis_mask
    if not np.any(valid_mask):
        return np.zeros_like(surface, dtype=float)

    coeffs = fit_basis_coefficients(surface, basis_stack, valid_mask)
    cleaned = np.array(surface, copy=True, dtype=float)
    cleaned[valid_mask] = cleaned[valid_mask] - evaluate_basis_surface(coeffs, basis_stack)[valid_mask]
    cleaned[~mask] = np.nan
    return cleaned

