"""Shared helpers for legacy editable baselines."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage
from numpy.polynomial.legendre import legval
from stitching.contracts import SubApertureObservation
from stitching.trusted.bases.zernike import generate_zernike_surface


def rounded_placement_slices(
    global_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[slice, slice, slice, slice]:
    """Return the same rounded placement slices used by the evaluator."""

    rows, cols = tile_shape
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])

    top = int(round(center_y - (rows - 1) / 2.0))
    left = int(round(center_x - (cols - 1) / 2.0))
    bottom = top + rows
    right = left + cols

    gy_s = max(0, top)
    gy_e = min(global_shape[0], bottom)
    gx_s = max(0, left)
    gx_e = min(global_shape[1], right)

    ly_s = max(0, -top)
    lx_s = max(0, -left)
    ly_e = ly_s + max(0, gy_e - gy_s)
    lx_e = lx_s + max(0, gx_e - gx_s)

    return slice(gy_s, gy_e), slice(gx_s, gx_e), slice(ly_s, ly_e), slice(lx_s, lx_e)


def observed_support_mask(
    observations: tuple[SubApertureObservation, ...],
    global_shape: tuple[int, int],
) -> np.ndarray:
    """Compute physical support from rounded tile placement and local valid masks."""

    support = np.zeros(global_shape, dtype=bool)
    for obs in observations:
        gy, gx, ly, lx = rounded_placement_slices(global_shape, obs.tile_shape, obs.center_xy)
        if gy.stop <= gy.start or gx.stop <= gx.start:
            continue
        local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
        support_view = support[gy, gx]
        support_view[local_mask] = True
    return support


def align_tile_to_rounded_grid(
    values: np.ndarray,
    mask: np.ndarray,
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a detector tile onto the rounded placement grid with subpixel interpolation."""

    rows, cols = values.shape
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])
    top_exact = center_y - (rows - 1) / 2.0
    left_exact = center_x - (cols - 1) / 2.0
    top_rounded = round(top_exact)
    left_rounded = round(left_exact)
    shift_y = top_exact - top_rounded
    shift_x = left_exact - left_rounded

    filled_values = np.where(np.asarray(mask, dtype=bool), np.asarray(values, dtype=float), 0.0)
    shifted_values = ndimage.shift(filled_values, shift=(shift_y, shift_x), order=1, mode="constant", cval=0.0)
    shifted_mask = ndimage.shift(
        np.asarray(mask, dtype=float),
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
    ) > 0.0
    return shifted_values, shifted_mask


def center_tile_in_canvas(
    values: np.ndarray,
    mask: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Embed a tile at the center of a larger canvas."""

    tile_rows, tile_cols = values.shape
    canvas_rows, canvas_cols = canvas_shape
    top = (canvas_rows - tile_rows) // 2
    left = (canvas_cols - tile_cols) // 2
    canvas_values = np.zeros(canvas_shape, dtype=float)
    canvas_mask = np.zeros(canvas_shape, dtype=bool)
    canvas_values[top : top + tile_rows, left : left + tile_cols] = np.asarray(values, dtype=float)
    canvas_mask[top : top + tile_rows, left : left + tile_cols] = np.asarray(mask, dtype=bool)
    return canvas_values, canvas_mask


def resize_carte(carte: np.ndarray, resolution_finale: int) -> np.ndarray:
    """Match the legacy MATLAB `resizeCarte` helper for square arrays."""

    carte = np.asarray(carte)
    size = int(carte.shape[0]) if carte.ndim else 0
    if size == resolution_finale:
        return np.array(carte, copy=True)

    if size > resolution_finale:
        offset = (size - resolution_finale) // 2
        return np.array(
            carte[offset : offset + resolution_finale, offset : offset + resolution_finale],
            copy=True,
        )

    out = np.full((resolution_finale, resolution_finale), np.nan, dtype=float)
    offset = (resolution_finale - size) // 2
    out[offset : offset + size, offset : offset + size] = np.asarray(carte, dtype=float)
    return out


def shift_canvas(
    values: np.ndarray,
    mask: np.ndarray,
    shift_xy: tuple[float, float],
    *,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a canvas in global coordinates using interpolation."""

    shift_x = float(shift_xy[0])
    shift_y = float(shift_xy[1])
    filled_values = np.where(np.asarray(mask, dtype=bool), np.asarray(values, dtype=float), 0.0)
    shifted_values = ndimage.shift(filled_values, shift=(shift_y, shift_x), order=order, mode="constant", cval=0.0)
    shifted_mask = ndimage.shift(np.asarray(mask, dtype=float), shift=(shift_y, shift_x), order=1, mode="constant", cval=0.0) >= 0.5
    return shifted_values, shifted_mask


def place_tile_in_global_frame(
    values: np.ndarray,
    mask: np.ndarray,
    global_shape: tuple[int, int],
    center_xy: tuple[float, float],
    *,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a local tile in a global canvas using the exact tile center.

    The tile is first centered in a canvas of the target global shape and then
    shifted to its requested center. This keeps sub-pixel placement explicit and
    avoids the hard edges introduced by rounded crop placement.
    """

    canvas_values, canvas_mask = center_tile_in_canvas(values, mask, global_shape)
    global_center_x = (global_shape[1] - 1) / 2.0
    global_center_y = (global_shape[0] - 1) / 2.0
    shift_x = float(center_xy[0]) - global_center_x
    shift_y = float(center_xy[1]) - global_center_y
    return shift_canvas(canvas_values, canvas_mask, (shift_x, shift_y), order=order)


def overlap_support_mask(
    observations: tuple[SubApertureObservation, ...],
    global_shape: tuple[int, int],
) -> np.ndarray:
    """Return pixels covered by at least two valid observations.

    This mirrors the legacy NEOSS `calculMasquerecouvrement` path more closely
    than rounded crop placement by placing each tile in the global frame and
    then applying the same interpolation-based shift used during stitching.
    """

    counts = np.zeros(global_shape, dtype=int)
    for obs in observations:
        values = np.asarray(obs.z, dtype=float)
        mask = np.asarray(obs.valid_mask, dtype=bool)
        if not np.any(mask):
            continue
        _, shifted_mask = place_tile_in_global_frame(values, mask, global_shape, obs.center_xy, order=3)
        counts += shifted_mask.astype(int)
    return counts > 1


def project_global_mask_to_tile(
    global_mask: np.ndarray,
    global_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> np.ndarray:
    """Project a global support mask back into the tile frame.

    This mirrors the MATLAB `reinterpADO(-X,-Y,mask)` followed by `resizeCarte`.
    """

    global_center_x = (global_shape[1] - 1) / 2.0
    global_center_y = (global_shape[0] - 1) / 2.0
    shift_x = global_center_x - float(center_xy[0])
    shift_y = global_center_y - float(center_xy[1])
    shifted_mask = shift_canvas(
        np.asarray(global_mask, dtype=float),
        np.asarray(global_mask, dtype=bool),
        (shift_x, shift_y),
        order=3,
    )[1]
    tiled = resize_carte(shifted_mask.astype(float), tile_shape[0])
    return np.asarray(tiled, dtype=float) > 0.0


def normalized_tile_coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return local coordinates normalized to [-1, 1]."""

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
    x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
    return y_norm, x_norm


def _legendre_power_pair(term: int) -> tuple[int, int]:
    """Match the legacy 2D Legendre enumeration used in the MATLAB code."""

    if term < 0:
        raise ValueError("Legendre term index must be non-negative.")

    i = 0
    degree = 0
    while True:
        for x_power in range(degree, -1, -1):
            y_power = degree - x_power
            if i == term:
                return x_power, y_power
            i += 1
        degree += 1


def _legendre_1d(power: int, coord: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(power + 1, dtype=float)
    coeffs[-1] = 1.0
    return legval(coord, coeffs)


def _zernike_index_pairs(indexing: str, num_terms: int) -> list[tuple[int, int]]:
    """Return mode pairs for standard real-valued Zernike orderings."""

    indexing = indexing.lower()
    if indexing == "iso":
        return _iso_pairs(num_terms)

    pairs: list[tuple[int, int]] = []
    n = 0
    while len(pairs) < num_terms:
        if indexing == "ansi":
            ordered_m = list(range(-n, n + 1, 2))
        elif indexing == "noll":
            if n % 2 == 0:
                ordered_m = [0]
                for abs_m in range(2, n + 1, 2):
                    ordered_m.extend((-abs_m, abs_m))
            else:
                ordered_m = []
                for abs_m in range(1, n + 1, 2):
                    ordered_m.extend((-abs_m, abs_m))
        elif indexing == "fringe":
            ordered_m = [0] if n % 2 == 0 else []
            for abs_m in range(1 if n % 2 else 2, n + 1, 2):
                ordered_m.extend((abs_m, -abs_m))
        else:
            raise ValueError(f"Unsupported Zernike indexing '{indexing}'.")

        for m in ordered_m:
            pairs.append((n, m))
            if len(pairs) == num_terms:
                break
        n += 1
    return pairs


def sample_basis_term_stack_from_coords(
    mode: str,
    terms: Iterable[int],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    zernike_indexing: str = "fringe",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a basis stack on arbitrary coordinates.

    This is useful for NEOSS CS terms, where the legacy code evaluates the
    modes on translated/scaled grids rather than on the canonical tile grid.
    """

    mode_norm = mode.upper()
    zernike_indexing = zernike_indexing.lower()
    term_list = [int(term) for term in terms]
    if not term_list:
        empty = np.zeros((0,) + x_coords.shape, dtype=float)
        return empty, np.zeros(x_coords.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    if mode_norm == "Z":
        rho = np.sqrt(x_coords**2 + y_coords**2)
        theta = np.mod(np.arctan2(y_coords, x_coords), 2.0 * np.pi)
        stack = []
        base_mask = np.asarray(mask, dtype=bool) if mask is not None else (rho <= 1.0)
        for term in term_list:
            n, m = _zernike_index_pairs(zernike_indexing, term + 1)[term]
            radial = _zernike_radial(n, m, rho)
            if m == 0:
                sampled = radial
            elif m > 0:
                sampled = radial * np.cos(m * theta)
            else:
                sampled = radial * np.sin(abs(m) * theta)
            stack.append(np.where(base_mask, sampled, 0.0))
        return np.stack(stack, axis=0), base_mask

    if mode_norm != "L":
        raise ValueError(f"Unsupported legacy basis mode '{mode}'.")

    stack = []
    base_mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones(x_coords.shape, dtype=bool)
    for term in term_list:
        x_power, y_power = _legendre_power_pair(term)
        basis = _legendre_1d(x_power, x_coords) * _legendre_1d(y_power, y_coords)
        stack.append(np.where(base_mask, basis, 0.0))
    return np.stack(stack, axis=0), base_mask


def _normalized_pupil_grid(
    shape: tuple[int, int],
    radius_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized x/y coordinates and a unit-disk mask."""

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    if radius_fraction is None:
        x = 2.0 * xx / max(cols - 1, 1) - 1.0
        y = 2.0 * yy / max(rows - 1, 1) - 1.0
    else:
        cy = (rows - 1) / 2.0
        cx = (cols - 1) / 2.0
        r_pixel = min(rows, cols) * radius_fraction
        x = (xx - cx) / r_pixel
        y = (yy - cy) / r_pixel
    mask = x**2 + y**2 <= 1.0
    return x, y, mask


def _iso_pairs(num_terms: int) -> list[tuple[int, int]]:
    """Enumerate ISO/legacy Zernike (n, m) pairs."""

    pairs: list[tuple[int, int]] = []
    no = 0
    while len(pairs) < num_terms:
        for n in range(no // 2, no + 1):
            m = no - n
            pairs.append((n, m))
            if len(pairs) == num_terms:
                break
            if m != 0:
                pairs.append((n, -m))
                if len(pairs) == num_terms:
                    break
        no += 2
    return pairs


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Evaluate the radial Zernike polynomial."""

    abs_m = abs(m)
    if (n - abs_m) % 2 != 0 or abs_m > n:
        return np.zeros_like(rho, dtype=float)

    radial = np.zeros_like(rho, dtype=float)
    max_k = (n - abs_m) // 2
    for k in range(max_k + 1):
        coefficient = (
            (-1.0) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + abs_m) // 2 - k)
                * math.factorial((n - abs_m) // 2 - k)
            )
        )
        radial = radial + coefficient * rho ** (n - 2 * k)
    return radial


def _generate_iso_zernike_surface(
    term: int,
    shape: tuple[int, int],
    *,
    radius_fraction: float | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Generate the legacy ISO-ordered Zernike surface for one term."""

    x, y, mask = _normalized_pupil_grid(shape, radius_fraction=radius_fraction)
    rho = np.sqrt(x**2 + y**2)
    theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    n, m = _iso_pairs(term + 1)[term]
    radial = _zernike_radial(n, m, rho)
    if m == 0:
        surface = radial
    elif m > 0:
        surface = radial * np.cos(m * theta)
    else:
        surface = radial * np.sin(abs(m) * theta)
    return np.where(mask, surface, fill_value)


def basis_term_stack(
    mode: str,
    terms: Iterable[int],
    shape: tuple[int, int],
    *,
    radius_fraction: float | None = None,
    zernike_indexing: str = "fringe",
) -> tuple[np.ndarray, np.ndarray]:
    """Return basis samples stacked as (n_terms, rows, cols) and a mask."""

    mode_norm = mode.upper()
    zernike_indexing = zernike_indexing.lower()
    term_list = [int(term) for term in terms]
    if not term_list:
        empty = np.zeros((0, shape[0], shape[1]), dtype=float)
        return empty, np.zeros(shape, dtype=bool)

    if mode_norm == "Z":
        stack = []
        base_mask = None
        for term in term_list:
            if zernike_indexing == "iso":
                sampled = _generate_iso_zernike_surface(
                    term,
                    shape,
                    radius_fraction=radius_fraction,
                    fill_value=np.nan,
                )
            else:
                coeffs = np.zeros(term + 1, dtype=float)
                coeffs[term] = 1.0
                sampled = generate_zernike_surface(
                    coeffs,
                    shape,
                    indexing=zernike_indexing,
                    backend="internal",
                    radius_fraction=radius_fraction,
                    fill_value=np.nan,
                )
            if base_mask is None:
                base_mask = np.isfinite(sampled)
            stack.append(np.where(np.isfinite(sampled), sampled, 0.0))
        return np.stack(stack, axis=0), base_mask if base_mask is not None else np.zeros(shape, dtype=bool)

    if mode_norm != "L":
        raise ValueError(f"Unsupported legacy basis mode '{mode}'.")

    y_norm, x_norm = normalized_tile_coords(shape)
    stack = []
    for term in term_list:
        x_power, y_power = _legendre_power_pair(term)
        basis = _legendre_1d(x_power, x_norm) * _legendre_1d(y_power, y_norm)
        stack.append(basis)
    return np.stack(stack, axis=0), np.ones(shape, dtype=bool)


def fit_basis_coefficients(
    data: np.ndarray,
    basis_stack: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Least-squares fit of basis coefficients on masked pixels."""

    if basis_stack.size == 0 or not np.any(mask):
        return np.zeros((basis_stack.shape[0],), dtype=float)

    design = basis_stack[:, mask].T
    target = np.asarray(data, dtype=float)[mask]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coeffs


def evaluate_basis_surface(
    coeffs: np.ndarray,
    basis_stack: np.ndarray,
) -> np.ndarray:
    """Reconstruct a surface from a basis stack and coefficient vector."""

    if basis_stack.size == 0:
        return np.zeros((0, 0), dtype=float)
    return np.tensordot(np.asarray(coeffs, dtype=float), basis_stack, axes=(0, 0))


def remove_low_order_modes(
    surface: np.ndarray,
    mask: np.ndarray,
    mode: str,
    terms: Iterable[int],
    *,
    radius_fraction: float | None = None,
    zernike_indexing: str = "fringe",
) -> np.ndarray:
    """Project out low-order basis modes from a detector-fixed map."""

    basis_stack, basis_mask = basis_term_stack(
        mode,
        terms,
        surface.shape,
        radius_fraction=radius_fraction,
        zernike_indexing=zernike_indexing,
    )
    valid_mask = np.asarray(mask, dtype=bool) & basis_mask
    if not np.any(valid_mask):
        return np.zeros_like(surface, dtype=float)

    coeffs = fit_basis_coefficients(surface, basis_stack, valid_mask)
    cleaned = np.array(surface, copy=True, dtype=float)
    cleaned[valid_mask] = cleaned[valid_mask] - evaluate_basis_surface(coeffs, basis_stack)[valid_mask]
    cleaned[~mask] = np.nan
    return cleaned
