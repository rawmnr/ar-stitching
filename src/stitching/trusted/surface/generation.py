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


def generate_identity_surface(config: ScenarioConfig) -> SurfaceTruth:
    """Create a deterministic non-flat truth surface for trusted reconstruction tests.

    By default, uses a structured Legendre surface. Can be overridden in config metadata.
    """

    metadata = config.metadata
    basis_name = str(metadata.get("truth_basis", "legendre"))
    
    if "truth_coefficients" in metadata:
        coeffs = np.asarray(metadata["truth_coefficients"], dtype=float)
    else:
        coeffs = _default_identity_coefficients()

    truth = surface_from_basis(
        shape=config.grid_shape,
        pixel_size=config.pixel_size,
        basis_name=basis_name,
        coefficients=coeffs,
    )
    
    # Optional mask override (e.g. for circular truth pupils)
    if metadata.get("truth_pupil") == "circular":
        from stitching.trusted.surface.footprint import circular_pupil_mask
        mask = circular_pupil_mask(config.grid_shape, radius_fraction=float(metadata.get("truth_radius", 0.45)))
        truth = SurfaceTruth(
            z=np.where(mask, truth.z, 0.0),
            valid_mask=mask,
            pixel_size=truth.pixel_size,
            units=truth.units,
            metadata=truth.metadata,
        )

    return SurfaceTruth(
        z=truth.z,
        valid_mask=truth.valid_mask,
        pixel_size=truth.pixel_size,
        units=truth.units,
        metadata={
            **truth.metadata,
            "surface_model": basis_name if "truth_basis" in metadata else "legendre_structured_low_order",
        },
    )
