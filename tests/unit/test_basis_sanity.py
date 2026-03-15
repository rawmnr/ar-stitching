"""Unit tests for basis function sanity and orthogonality."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.trusted.bases.legendre import sample_legendre_basis_2d
from stitching.trusted.bases.zernike import generate_zernike_surface


def test_legendre_basis_approximate_orthogonality() -> None:
    shape = (32, 32)
    max_deg = 2
    basis = sample_legendre_basis_2d(shape, max_deg, max_deg)
    # basis shape is (max_deg+1, max_deg+1, 32, 32)
    
    flat_basis = basis.reshape(((max_deg + 1) ** 2, -1))
    
    # Check orthogonality: dot product of different basis functions should be small
    # normalized by the number of pixels
    gram_matrix = (flat_basis @ flat_basis.T) / (shape[0] * shape[1])
    
    # Off-diagonal elements should be relatively small
    # Legendre polynomials are orthogonal on [-1, 1], but discrete sampling introduces errors
    diag = np.diag(gram_matrix)
    off_diag = gram_matrix - np.diag(diag)
    
    assert np.all(diag > 0)
    assert np.all(np.abs(off_diag) < 0.1) # Loose bound for discrete grid


def test_zernike_basis_circular_masking() -> None:
    shape = (64, 64)
    # n=1, m=1 is x-tilt
    z = generate_zernike_surface(np.array([0, 1.0]), shape, indexing="ansi")
    
    # Outside unit disk should be exactly zero
    yy, xx = np.indices(shape, dtype=float)
    x = 2.0 * xx / (shape[1] - 1) - 1.0
    y = 2.0 * yy / (shape[0] - 1) - 1.0
    mask = x**2 + y**2 <= 1.0
    
    assert np.all(z[~mask] == 0.0)
    assert np.any(z[mask] != 0.0)


def test_zernike_internal_vs_optiland_parity() -> None:
    # If optiland is available, check they agree
    try:
        import optiland
    except ImportError:
        pytest.skip("optiland not installed")
        
    shape = (32, 32)
    coeffs = np.array([1.0, 0.5, -0.2])
    z_internal = generate_zernike_surface(coeffs, shape, indexing="noll", backend="internal")
    z_optiland = generate_zernike_surface(coeffs, shape, indexing="noll", backend="optiland")
    
    assert np.allclose(z_internal, z_optiland, atol=1e-10)
