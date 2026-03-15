"""Trusted scan-plan and rigid-transform placeholders."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates


CENTER_ALIGNMENT_TOL = 1e-9


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


def extract_tile(
    global_surface: np.ndarray,
    global_mask: np.ndarray,
    tile_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a detector tile from a global surface with sub-pixel interpolation.

    If `center_xy` aligns perfectly with integer pixel placement, exact slicing
    is used to avoid interpolation artifacts. Otherwise, bilinear interpolation
    is used for both the surface values and the mask (thresholded at 0.5).
    """

    try:
        global_y, global_x, local_y, local_x = placement_slices(global_surface.shape, tile_shape, center_xy)
        z = np.zeros(tile_shape, dtype=float)
        valid_mask = np.zeros(tile_shape, dtype=bool)
        z[local_y, local_x] = global_surface[global_y, global_x]
        valid_mask[local_y, local_x] = global_mask[global_y, global_x]
        return z, valid_mask
    except ValueError:
        return _extract_tile_interpolated(global_surface, global_mask, tile_shape, center_xy)


def _extract_tile_interpolated(
    global_surface: np.ndarray,
    global_mask: np.ndarray,
    tile_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Perform bilinear interpolation for sub-pixel tile extraction."""

    rows, cols = tile_shape
    origin_x = center_xy[0] - (cols - 1) / 2.0
    origin_y = center_xy[1] - (rows - 1) / 2.0

    yy, xx = np.indices(tile_shape, dtype=float)
    coords = np.array([yy.ravel() + origin_y, xx.ravel() + origin_x])

    # Map surface values (order=1 for bilinear)
    z = map_coordinates(global_surface, coords, order=1, mode="constant", cval=0.0)
    z = z.reshape(tile_shape)

    # Map mask (order=1 and thresholding at 0.5 effectively keeps mask tight)
    mask_f = map_coordinates(global_mask.astype(float), coords, order=1, mode="constant", cval=0.0)
    valid_mask = mask_f.reshape(tile_shape) >= 0.5

    return z, valid_mask


def placement_slices(
    global_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[slice, slice, slice, slice]:
    """Return aligned global and local slices for integer placement with clipping.

    `center_xy` is interpreted as the geometric center of the tile in pixel-center coordinates.
    For even tile sizes this naturally yields half-integer centers.
    """

    center_x = float(center_xy[0])
    center_y = float(center_xy[1])
    tile_rows, tile_cols = tile_shape

    top = _aligned_integer_origin(center_y, tile_rows, axis_name="y")
    left = _aligned_integer_origin(center_x, tile_cols, axis_name="x")
    bottom = top + tile_rows
    right = left + tile_cols

    global_y_start = max(0, top)
    global_y_end = min(global_shape[0], bottom)
    global_x_start = max(0, left)
    global_x_end = min(global_shape[1], right)

    local_y_start = max(0, -top)
    local_x_start = max(0, -left)
    local_y_end = local_y_start + max(0, global_y_end - global_y_start)
    local_x_end = local_x_start + max(0, global_x_end - global_x_start)

    return (
        slice(global_y_start, global_y_end),
        slice(global_x_start, global_x_end),
        slice(local_y_start, local_y_end),
        slice(local_x_start, local_x_end),
    )


def _aligned_integer_origin(center: float, tile_extent: int, axis_name: str) -> int:
    """Convert a parity-compatible geometric center to an integer array origin."""

    origin = center - (tile_extent - 1) / 2.0
    rounded_origin = round(origin)
    if not math.isclose(origin, rounded_origin, abs_tol=CENTER_ALIGNMENT_TOL):
        raise ValueError(
            f"Tile center along {axis_name}={center} is incompatible with integer placement for extent {tile_extent}."
        )
    return int(rounded_origin)
