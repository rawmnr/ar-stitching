"""Trusted surface generation placeholders."""

from __future__ import annotations

import numpy as np

from stitching.contracts import SurfaceTruth
from stitching.trusted.bases.legendre import generate_legendre_surface
from stitching.trusted.bases.zernike import generate_zernike_surface


def _default_identity_coefficients() -> np.ndarray:
    """Return deterministic Legendre coefficients for a structured trusted truth surface."""

    coefficients = np.zeros((4, 4), dtype=float)
    coefficients[0, 0] = 0.012
    coefficients[0, 1] = 0.04
    coefficients[1, 0] = -0.025
    coefficients[0, 2] = 0.01
    coefficients[2, 0] = 0.006
    coefficients[1, 1] = 0.008
    coefficients[0, 3] = -0.004
    coefficients[3, 0] = 0.003
    coefficients[2, 1] = -0.005
    coefficients[1, 2] = 0.004
    return coefficients


def surface_from_basis(
    shape: tuple[int, int],
    pixel_size: float,
    basis_name: str,
    coefficients: np.ndarray,
    units: str = "arb",
) -> SurfaceTruth:
    """Construct a trusted surface from a supported basis family."""

    if basis_name == "legendre":
        z = generate_legendre_surface(coefficients, shape)
        valid_mask = np.ones(shape, dtype=bool)
    elif basis_name == "zernike":
        z = generate_zernike_surface(coefficients, shape, indexing="noll", backend="internal")
        yy, xx = np.indices(shape, dtype=float)
        x = 2.0 * xx / max(shape[1] - 1, 1) - 1.0
        y = 2.0 * yy / max(shape[0] - 1, 1) - 1.0
        valid_mask = x**2 + y**2 <= 1.0
    else:
        raise ValueError(f"Unsupported basis '{basis_name}'.")

    return SurfaceTruth(
        z=np.asarray(z, dtype=float),
        valid_mask=valid_mask,
        pixel_size=pixel_size,
        units=units,
        metadata={"surface_model": basis_name},
    )


def generate_identity_surface(shape: tuple[int, int], pixel_size: float) -> SurfaceTruth:
    """Create a deterministic non-flat truth surface for trusted reconstruction tests."""

    truth = surface_from_basis(
        shape=shape,
        pixel_size=pixel_size,
        basis_name="legendre",
        coefficients=_default_identity_coefficients(),
    )
    return SurfaceTruth(
        z=truth.z,
        valid_mask=truth.valid_mask,
        pixel_size=truth.pixel_size,
        units=truth.units,
        metadata={"surface_model": "legendre_structured_low_order"},
    )
