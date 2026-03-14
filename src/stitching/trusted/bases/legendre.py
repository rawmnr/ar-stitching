"""Legendre basis helpers for square detector tiles and low-order truth surfaces."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval


def _normalized_axis(length: int) -> np.ndarray:
    """Return coordinates mapped from array indices to [-1, 1]."""

    if length <= 1:
        return np.array([0.0], dtype=float)
    return np.linspace(-1.0, 1.0, num=length, dtype=float)


def evaluate_legendre_basis_1d(length: int, max_degree: int) -> np.ndarray:
    """Sample 1D Legendre basis functions up to `max_degree` on a normalized grid."""

    axis = _normalized_axis(length)
    basis = np.zeros((max_degree + 1, length), dtype=float)
    for degree in range(max_degree + 1):
        coeffs = np.zeros(degree + 1, dtype=float)
        coeffs[-1] = 1.0
        basis[degree] = legval(axis, coeffs)
    return basis


def sample_legendre_basis_2d(shape: tuple[int, int], max_degree_y: int, max_degree_x: int) -> np.ndarray:
    """Return a tensor-product Legendre basis sampled on a 2D grid."""

    basis_y = evaluate_legendre_basis_1d(shape[0], max_degree_y)
    basis_x = evaluate_legendre_basis_1d(shape[1], max_degree_x)
    basis = np.zeros((max_degree_y + 1, max_degree_x + 1, shape[0], shape[1]), dtype=float)

    for degree_y in range(max_degree_y + 1):
        for degree_x in range(max_degree_x + 1):
            basis[degree_y, degree_x] = np.outer(basis_y[degree_y], basis_x[degree_x])
    return basis


def generate_legendre_surface(coefficients: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Synthesize a 2D surface from tensor-product Legendre coefficients."""

    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.ndim != 2:
        raise ValueError("Legendre coefficients must be a 2D array.")

    basis = sample_legendre_basis_2d(shape, coeffs.shape[0] - 1, coeffs.shape[1] - 1)
    return np.tensordot(coeffs, basis, axes=((0, 1), (0, 1)))
