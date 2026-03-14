"""Trusted surface generation placeholders."""

from __future__ import annotations

import numpy as np

from stitching.contracts import SurfaceTruth
from stitching.trusted.bases.legendre import generate_legendre_surface


def _default_identity_coefficients() -> np.ndarray:
    """Return low-order Legendre coefficients for the default truth surface."""

    coefficients = np.zeros((3, 3), dtype=float)
    coefficients[0, 0] = 0.012
    coefficients[0, 1] = 0.04
    coefficients[1, 0] = -0.025
    coefficients[0, 2] = 0.01
    coefficients[2, 0] = 0.006
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
    else:
        raise ValueError(f"Unsupported basis '{basis_name}'.")

    return SurfaceTruth(
        z=np.asarray(z, dtype=float),
        valid_mask=np.ones(shape, dtype=bool),
        pixel_size=pixel_size,
        units=units,
        metadata={"surface_model": basis_name},
    )


def generate_identity_surface(shape: tuple[int, int], pixel_size: float) -> SurfaceTruth:
    """Create a deterministic low-order truth surface for identity-phase tests."""

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
        metadata={"surface_model": "legendre_tilt_plus_weak_quadratic"},
    )
