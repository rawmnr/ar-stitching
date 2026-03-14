"""Trusted basis functions used for deterministic surface generation."""

from .legendre import evaluate_legendre_basis_1d, generate_legendre_surface, sample_legendre_basis_2d
from .zernike import generate_zernike_surface

__all__ = [
    "evaluate_legendre_basis_1d",
    "sample_legendre_basis_2d",
    "generate_legendre_surface",
    "generate_zernike_surface",
]
