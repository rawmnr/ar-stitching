"""Trusted instrument-reference bias placeholders."""

from __future__ import annotations

import numpy as np


def stationary_reference_bias(
    shape: tuple[int, int], 
    bias: float,
    radius_fraction: float | None = None,
) -> np.ndarray:
    """Create a detector-frame stationary bias field."""

    field = np.full(shape, bias, dtype=float)
    if radius_fraction is not None:
        from stitching.trusted.surface.footprint import circular_pupil_mask
        mask = circular_pupil_mask(shape, radius_fraction=radius_fraction)
        field = np.where(mask, field, np.nan)
    return field


def generate_reference_bias_field(
    shape: tuple[int, int],
    coefficients: np.ndarray | None = None,
    radius_fraction: float | None = None,
    hf_amplitude: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a static field-dependent reference bias with optional HF artifacts."""
    
    # 1. Base Zernike field (Low Frequency)
    if coefficients is not None and len(coefficients) > 0:
        from stitching.trusted.bases.zernike import generate_zernike_surface
        bias_field = generate_zernike_surface(
            coefficients, 
            shape, 
            indexing="noll", 
            backend="internal",
            radius_fraction=radius_fraction,
            fill_value=np.nan
        )
        bias_field = np.asarray(bias_field, dtype=float)
    else:
        bias_field = np.zeros(shape, dtype=float)

    # 2. High Frequency Artifacts
    if hf_amplitude > 0.0:
        hf_field = _generate_hf_artifacts(shape, hf_amplitude, seed=seed)
        bias_field += hf_field

    # 3. Apply mask if requested and not already applied by Zernike
    if radius_fraction is not None and np.all(np.isnan(bias_field) == False):
        from stitching.trusted.surface.footprint import circular_pupil_mask
        mask = circular_pupil_mask(shape, radius_fraction=radius_fraction)
        bias_field = np.where(mask, bias_field, np.nan)
        
    return bias_field


def _generate_hf_artifacts(
    shape: tuple[int, int], 
    amplitude: float, 
    seed: int | None = None
) -> np.ndarray:
    """Generate high-frequency patterns (grid, rings, and localized defects)."""
    
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape, dtype=float)
    
    # Normalize coordinates to [-1, 1]
    ny = 2.0 * (yy - (shape[0] - 1) / 2.0) / (shape[0] - 1)
    nx = 2.0 * (xx - (shape[1] - 1) / 2.0) / (shape[1] - 1)
    rr = np.sqrt(nx**2 + ny**2)
    
    hf = np.zeros(shape, dtype=float)
    
    # 1. Grid lines (Simulation of stitching/manufacturing artifacts)
    grid_freq = rng.uniform(20.0, 40.0)
    hf += 0.3 * np.sin(grid_freq * nx) * np.sin(grid_freq * ny)
    
    # 2. Concentric rings (Simulation of diamond turning or centering artifacts)
    ring_freq = rng.uniform(30.0, 60.0)
    hf += 0.4 * np.cos(ring_freq * rr)
    
    # 3. Localized defects (Random Gaussian blobs)
    num_blobs = rng.integers(5, 15)
    for _ in range(num_blobs):
        bx, by = rng.uniform(-0.8, 0.8, size=2)
        b_sigma = rng.uniform(0.02, 0.08)
        b_mag = rng.uniform(-1.0, 1.0)
        dist_sq = (nx - bx)**2 + (ny - by)**2
        hf += b_mag * np.exp(-dist_sq / (2 * b_sigma**2))
        
    # 4. Global "Orange Peel" / Roughness (Mid-HF)
    from stitching.trusted.noise.models import add_mid_spatial_ripples
    # We use a temporary seed offset to not collide with other generators
    hf = add_mid_spatial_ripples(hf, magnitude=0.2, seed=(seed + 1 if seed is not None else None))

    return hf * amplitude


def reference_bias_for_observation(
    base_bias: float,
    observation_index: int,
    metadata: dict[str, object] | None = None,
) -> float:
    """Return the total scalar detector bias for one observation.

    The foundation model keeps detector-frame bias scalar, but allows a simple
    time-varying drift sequence on top of the stationary component.
    """

    metadata = metadata or {}
    total_bias = float(base_bias)
    if "reference_bias_values" in metadata:
        values = tuple(float(value) for value in metadata["reference_bias_values"])  # type: ignore[index]
        if observation_index >= len(values):
            raise ValueError("reference_bias_values must provide one bias per observation.")
        total_bias += values[observation_index]
    elif "reference_bias_drift_step" in metadata:
        total_bias += float(metadata["reference_bias_drift_step"]) * float(observation_index)
    elif "reference_bias_drift" in metadata:
        total_bias += float(metadata["reference_bias_drift"])
    return total_bias


def apply_reference_bias(z: np.ndarray, bias: float | np.ndarray) -> np.ndarray:
    """Add a detector-frame reference bias (scalar or field)."""

    return np.asarray(z, dtype=float) + bias
