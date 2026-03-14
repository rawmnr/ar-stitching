"""Trusted scan-plan and rigid-transform placeholders."""

from __future__ import annotations

import math

import numpy as np


def rotation_matrix_deg(angle_deg: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for bookkeeping and tests."""

    angle_rad = math.radians(angle_deg)
    return np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=float,
    )


def apply_integer_shift(values: np.ndarray, shift_xy: tuple[int, int]) -> np.ndarray:
    """Shift an array with zero fill. This keeps geometry tests explicit."""

    dx, dy = shift_xy
    result = np.zeros_like(values)

    src_x_start = max(0, -dx)
    src_x_end = values.shape[1] - max(0, dx)
    src_y_start = max(0, -dy)
    src_y_end = values.shape[0] - max(0, dy)

    dst_x_start = max(0, dx)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_start = max(0, dy)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    if src_x_end <= src_x_start or src_y_end <= src_y_start:
        return result

    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = values[src_y_start:src_y_end, src_x_start:src_x_end]
    return result
