"""Trusted pupil-mask and footprint generation helpers."""

from __future__ import annotations

import numpy as np


def circular_pupil_mask(shape: tuple[int, int], radius_fraction: float = 0.45) -> np.ndarray:
    """Return a centered circular mask for simple footprint experiments."""

    rows, cols = shape
    yy, xx = np.ogrid[:rows, :cols]
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = min(rows, cols) * radius_fraction
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
