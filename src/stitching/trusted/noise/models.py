"""Trusted nuisance and noise hooks with deterministic seed control."""

from __future__ import annotations

import numpy as np
from stitching.trusted.bases.zernike import generate_zernike_surface


OUTLIER_SCALE_EPS = 1e-12


from scipy.ndimage import gaussian_filter, distance_transform_edt


def apply_optical_psf(z: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Apply a Gaussian blur to simulate optical PSF and pixel integration.

    Eliminates mathematical aliasing by smoothing high-frequencies before sampling.
    """

    if sigma_pixels <= 0.0:
        return z
    return gaussian_filter(z, sigma=sigma_pixels)


def apply_edge_degradation(
    z: np.ndarray,
    mask: np.ndarray,
    roll_off_width: float,
    noise_boost: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply signal attenuation and noise increase near the pupil edges.

    Simulates diffraction/vignetting/roll-off at the boundary.
    Returns (degraded_z, softened_mask_or_weights).
    """

    if roll_off_width <= 0.0:
        return z, mask

    # Compute distance to edge (inside the mask)
    dist = distance_transform_edt(mask)
    
    # Sigmoid-like attenuation (0 at edge, 1 at center)
    weights = np.clip(dist / roll_off_width, 0, 1)
    weights = 0.5 * (1 + np.cos(np.pi * (1 - weights))) # Smooth cosine bell

    result = z * weights
    
    # Optionally boost noise at the edges (lower SNR)
    if noise_boost > 0.0:
        rng = np.random.default_rng(seed)
        edge_noise = rng.normal(0.0, noise_boost, size=z.shape)
        result += edge_noise * (1 - weights)

    return result, mask


def apply_global_drift(
    z: np.ndarray,
    center_xy: tuple[float, float],
    global_shape: tuple[int, int],
    drift_coefficients: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply a global-frame perturbation field evaluated at the tile location.

    This represents slowly-varying spatial bias across the full scan area.
    """

    if not drift_coefficients:
        return z

    rows, cols = z.shape
    yy, xx = np.indices(z.shape, dtype=float)
    
    # Global normalized coordinates [-1, 1]
    gx = 2.0 * (xx + center_xy[0] - (cols - 1) / 2.0) / max(global_shape[1] - 1, 1) - 1.0
    gy = 2.0 * (yy + center_xy[1] - (rows - 1) / 2.0) / max(global_shape[0] - 1, 1) - 1.0

    result = np.asarray(z, dtype=float).copy()
    for name, value in drift_coefficients.items():
        if name == "drift_x":
            result += float(value) * gx
        elif name == "drift_y":
            result += float(value) * gy
        elif name == "drift_r2":
            result += float(value) * (gx**2 + gy**2)
    return result


def apply_nuisance_terms(z: np.ndarray, nuisance_terms: dict[str, float] | None = None) -> np.ndarray:
    """Apply additive nuisance terms including tip/tilt/focus and scalar DC.

    Terms are applied on a normalized detector grid [-1, 1].
    """

    result = np.asarray(z, dtype=float).copy()
    if not nuisance_terms:
        return result

    rows, cols = z.shape
    yy, xx = np.indices(z.shape, dtype=float)
    x = 2.0 * xx / max(cols - 1, 1) - 1.0
    y = 2.0 * yy / max(rows - 1, 1) - 1.0

    for name, value in nuisance_terms.items():
        if name == "subaperture_dc":
            result += float(value)
        elif name == "subaperture_tilt":  # x-gradient
            result += float(value) * x
        elif name == "subaperture_tip":  # y-gradient
            result += float(value) * y
        elif name == "subaperture_focus":  # simple power
            result += float(value) * (x**2 + y**2)
        elif name.startswith("subaperture_"):
            # Skip unknown subaperture-specific terms for now or treat as scalar if appropriate
            # but usually they are structured. For now, keep it simple.
            continue
        else:
            # Generic scalar terms
            result += float(value)
    return result


def add_gaussian_noise(z: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Inject zero-mean Gaussian noise."""

    if std == 0.0:
        return np.asarray(z, dtype=float).copy()
    rng = np.random.default_rng(seed)
    return np.asarray(z, dtype=float) + rng.normal(0.0, std, size=z.shape)


def outlier_magnitude_scale(z: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """Estimate a deterministic signal scale for relative outlier injection."""

    values = np.asarray(z, dtype=float)
    if valid_mask is not None:
        values = values[np.asarray(valid_mask, dtype=bool)]
    if values.size == 0:
        return 1.0
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if scale <= OUTLIER_SCALE_EPS:
        span = float(np.max(values) - np.min(values))
        scale = span / 2.0
    return max(1.0, scale)


def add_outliers(
    z: np.ndarray,
    fraction: float,
    magnitude: float,
    seed: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inject sparse additive outliers at random pixels."""

    result = np.asarray(z, dtype=float).copy()
    if fraction <= 0.0:
        return result
    rng = np.random.default_rng(seed)
    candidate_mask = np.ones(result.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    candidate_count = int(candidate_mask.sum())
    count = int(round(candidate_count * fraction))
    if count == 0:
        return result
    flat_indices = _sample_flat_indices(candidate_mask, count, rng)
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    result.flat[flat_indices] += outlier_magnitude_scale(result, candidate_mask) * float(magnitude) * signs
    return result


def add_mid_spatial_ripples(
    z: np.ndarray,
    magnitude: float,
    seed: int,
    num_ripples: int = 5,
) -> np.ndarray:
    """Add periodic mid-spatial frequency ripples (polishing marks).

    Sum of 2D sine waves with random orientations and frequencies.
    """

    if magnitude <= 0.0:
        return z

    rng = np.random.default_rng(seed)
    rows, cols = z.shape
    yy, xx = np.indices(z.shape, dtype=float)
    
    result = np.asarray(z, dtype=float).copy()
    
    for _ in range(num_ripples):
        # Random frequency (mid-range)
        freq = rng.uniform(0.05, 0.2)
        # Random angle
        angle = rng.uniform(0, np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        
        kx = freq * np.cos(angle)
        ky = freq * np.sin(angle)
        
        wave = np.sin(2 * np.pi * (kx * xx + ky * yy) + phase)
        # Each sine has std = 1/sqrt(2). Sum of N has std = sqrt(N/2).
        # We want total std = magnitude.
        result += (magnitude * np.sqrt(2.0 / num_ripples)) * wave
        
    return result


def add_low_frequency_noise(
    z: np.ndarray,
    magnitude: float,
    seed: int,
    indexing: str = "fringe",
    num_modes: int = 15,
) -> np.ndarray:
    """Add low-frequency noise based on Zernike modes with decreasing power.

    Simulates optical/mechanical instabilities (Z1 to Z15).
    Power follows a 1/n decay where n is the mode index.
    """

    if magnitude <= 0.0:
        return z

    rng = np.random.default_rng(seed)
    # Generate random coefficients for modes 1 to num_modes
    # Power decreases with mode index
    coeffs = rng.normal(0.0, 1.0, size=num_modes)
    powers = 1.0 / np.arange(1, num_modes + 1)
    effective_coeffs = coeffs * powers * magnitude

    # Generate the noise surface (circular support assumed for Zernike modes)
    noise_surface = generate_zernike_surface(
        effective_coeffs,
        z.shape,
        indexing=indexing,
        backend="internal",
    )
    
    return np.asarray(z, dtype=float) + noise_surface


def apply_retrace_error(z: np.ndarray, magnitude: float, slope_magnitude: float = 0.0) -> np.ndarray:
    """Apply surface-dependent retrace distortion (scalar and/or slope-dependent)."""

    result = np.asarray(z, dtype=float).copy()
    
    # 1. Simple scalar-based retrace
    if magnitude != 0.0:
        centered = result - float(np.mean(result))
        result += float(magnitude) * centered * np.abs(centered)
        
    # 2. Slope-based retrace (local gradient dependent)
    if slope_magnitude != 0.0:
        gy, gx = np.gradient(result)
        slope = np.sqrt(gx**2 + gy**2)
        # Model: error proportional to local slope and local surface value
        result += float(slope_magnitude) * slope * result
        
    return result


def _sample_flat_indices(candidate_mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample valid flat indices without always materializing all candidate indices."""

    flat_mask = candidate_mask.ravel()
    if np.all(flat_mask):
        return np.asarray(rng.choice(flat_mask.size, size=count, replace=False), dtype=np.intp)

    valid_count = int(flat_mask.sum())
    if valid_count == 0:
        return np.zeros(0, dtype=np.intp)

    # For sparse masks, sample valid ranks first, then map them through the mask.
    chosen_ranks = np.sort(np.asarray(rng.choice(valid_count, size=count, replace=False), dtype=np.int64))
    flat_indices = np.empty(count, dtype=np.intp)
    rank_index = 0
    seen_valid = 0
    for flat_index, is_valid in enumerate(flat_mask.tolist()):
        if not is_valid:
            continue
        while rank_index < count and chosen_ranks[rank_index] == seen_valid:
            flat_indices[rank_index] = flat_index
            rank_index += 1
        seen_valid += 1
        if rank_index == count:
            break
    return flat_indices
