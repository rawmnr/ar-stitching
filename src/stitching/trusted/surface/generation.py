"""Trusted surface generation placeholders."""

from __future__ import annotations

import numpy as np

from stitching.contracts import SurfaceTruth


def generate_identity_surface(shape: tuple[int, int], pixel_size: float) -> SurfaceTruth:
    """Create a deterministic low-order truth surface for identity-phase tests."""

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    x_norm = (xx - (cols - 1) / 2.0) / max(cols - 1, 1)
    y_norm = (yy - (rows - 1) / 2.0) / max(rows - 1, 1)

    tilt_x = 0.08 * x_norm
    tilt_y = -0.05 * y_norm
    quadratic = 0.03 * (x_norm**2 + 0.6 * y_norm**2)
    z = tilt_x + tilt_y + quadratic
    valid_mask = np.ones(shape, dtype=bool)
    return SurfaceTruth(
        z=z.astype(float),
        valid_mask=valid_mask,
        pixel_size=pixel_size,
        metadata={"surface_model": "tilt_plus_weak_quadratic"},
    )
