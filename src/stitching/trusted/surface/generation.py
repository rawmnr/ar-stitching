"""Trusted surface generation placeholders."""

from __future__ import annotations

import numpy as np

from stitching.contracts import SurfaceTruth


def generate_identity_surface(shape: tuple[int, int], pixel_size: float) -> SurfaceTruth:
    """Create a global truth surface for identity tests."""

    z = np.zeros(shape, dtype=float)
    valid_mask = np.ones(shape, dtype=bool)
    return SurfaceTruth(z=z, valid_mask=valid_mask, pixel_size=pixel_size)
