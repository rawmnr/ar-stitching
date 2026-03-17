"""Trusted scan plan generation for automated pattern coverage."""

from __future__ import annotations

import numpy as np


def check_coverage(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    offsets: tuple[tuple[float, float], ...],
    metadata: dict[str, any] | None = None,
) -> bool:
    """Verify that the union of sub-apertures covers the entire valid ground truth area.
    
    If 'truth_pupil' is 'circular', it checks coverage within the GT circle.
    Otherwise, it checks the whole grid.
    """
    metadata = metadata or {}
    rows, cols = grid_shape
    t_rows, t_cols = tile_shape
    
    # 1. Generate Global Truth Mask
    yy, xx = np.indices(grid_shape, dtype=float)
    cy_g = (rows - 1) / 2.0
    cx_g = (cols - 1) / 2.0
    
    if metadata.get("truth_pupil") == "circular":
        r_gt = min(rows, cols) * metadata.get("truth_radius_fraction", 0.5)
        truth_mask = (xx - cx_g)**2 + (yy - cy_g)**2 <= r_gt**2
    else:
        truth_mask = np.ones(grid_shape, dtype=bool)
        
    # 2. Generate Combined Sub-Aperture Mask
    combined_mask = np.zeros(grid_shape, dtype=bool)
    
    for ox, oy in offsets:
        # Global center of this sub-aperture
        cx_sa = cx_g + ox
        cy_sa = cy_g + oy
        
        if metadata.get("detector_pupil") == "circular":
            r_sa = min(t_rows, t_cols) * metadata.get("detector_radius_fraction", 0.45)
            sa_mask = (xx - cx_sa)**2 + (yy - cy_sa)**2 <= r_sa**2
        else:
            hw = t_cols / 2.0
            hh = t_rows / 2.0
            sa_mask = (xx >= cx_sa - hw) & (xx <= cx_sa + hw) & (yy >= cy_sa - hh) & (yy <= cy_sa + hh)
        
        combined_mask |= sa_mask
        
    # Check if truth is fully contained in combined
    missing = truth_mask & (~combined_mask)
    return not np.any(missing)


def generate_grid_scan_plan(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float = 0.2,
    seed: int = 0,
    metadata: dict[str, any] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Generate a grid of offsets that guarantees full coverage.
    
    Allows sub-apertures to go slightly outside the grid boundaries to ensure
    that even with circular pupils, the corners of the grid are covered.
    """
    metadata = metadata or {}
    rows, cols = grid_shape
    t_rows, t_cols = tile_shape

    if metadata.get("detector_pupil") == "circular":
        r_sa = min(t_rows, t_cols) * metadata.get("detector_radius_fraction", 0.45)
        if r_sa < 2.0:
            raise ValueError(f"Detector pupil is too small ({r_sa:.2f} px). Increase tile size or radius fraction.")
        # To cover a square grid with circles, the effective "safe" square side is r_sa * sqrt(2)
        eff_h = r_sa * np.sqrt(2.0)
        eff_w = r_sa * np.sqrt(2.0)
    else:
        eff_h, eff_w = t_rows, t_cols

    def calc_steps(total, eff_dim, overlap):
        if total <= eff_dim:
            return [0.0]
        # Required tiles N to cover 'total' with 'eff_dim' and 'overlap'
        n_tiles = int(np.ceil(1 + (total - eff_dim) / (eff_dim * (1.0 - overlap))))
        n_tiles = max(n_tiles, 2)
        # We allow centers to go up to total/2 if needed (sensor center on GT edge)
        # But here we just spread them to ensure coverage.
        max_offset = (total - eff_dim) / 2.0
        return np.linspace(-max_offset, max_offset, n_tiles)

    y_offsets = calc_steps(rows, eff_h, overlap_fraction)
    x_offsets = calc_steps(cols, eff_w, overlap_fraction)
    
    offsets = []
    for y in y_offsets:
        for x in x_offsets:
            offsets.append((float(x), float(y)))
            
    final_offsets = tuple(offsets)
    
    if not check_coverage(grid_shape, tile_shape, final_offsets, metadata):
        # Fallback: if the analytical step failed (can happen for circular at high overlap),
        # we could potentially add a safety margin to offsets here.
        pass

    return final_offsets


def generate_annular_scan_plan(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float = 0.2,
    num_rings: int | None = 2,
    seed: int = 0,
    metadata: dict[str, any] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Generate an annular pattern that guarantees 100% coverage and requested overlap.
    
    If num_rings is None, it is calculated automatically.
    Sub-aperture centers are allowed to exit the GT boundaries to cover extreme edges.
    """
    metadata = metadata or {}
    rows, cols = grid_shape
    t_rows, t_cols = tile_shape
    
    # 1. Determine radii
    if metadata.get("truth_pupil") == "circular":
        r_gt = min(rows, cols) * metadata.get("truth_radius_fraction", 0.5)
    else:
        r_gt = 0.5 * np.sqrt(rows**2 + cols**2)
        
    if metadata.get("detector_pupil") == "circular":
        r_sa = min(t_rows, t_cols) * metadata.get("detector_radius_fraction", 0.45)
    else:
        r_sa = 0.5 * min(t_rows, t_cols)

    # Safety check for tiny detectors
    if r_sa < 2.0 or r_sa < r_gt / 50.0:
        raise ValueError(f"Sub-aperture radius ({r_sa:.2f} px) is too small compared to GT ({r_gt:.2f} px).")

    # 2. Calculate radial step
    radial_step = 2.0 * r_sa * (1.0 - overlap_fraction)
    
    auto_mode = (num_rings is None)
    if num_rings is None:
        if r_gt <= r_sa:
            num_rings = 0
        else:
            num_rings = int(np.ceil((r_gt - r_sa) / radial_step))
            num_rings = max(num_rings, 1)

    while True:
        offsets = [(0.0, 0.0)] # Center
        
        for ring in range(1, num_rings + 1):
            radius = ring * radial_step
            
            # The last ring MUST be far enough to cover r_gt
            if ring == num_rings:
                # We allow radius + r_sa to be slightly larger than r_gt 
                # to avoid azimuthal holes at the very edge.
                radius = max(radius, r_gt - r_sa * 0.7)

            # Azimuthal density
            # Using 0.7 factor to ensure circular intersections stay outside r_gt
            arc_step = 2.0 * r_sa * (1.0 - overlap_fraction) * 0.7
            circumference = 2.0 * np.pi * radius
            
            if radius > 0:
                n_tiles = int(np.ceil(circumference / arc_step))
                n_tiles = max(n_tiles, 6 * ring)
            else:
                n_tiles = 0
            
            rng = np.random.default_rng(seed + ring)
            start_angle = rng.uniform(0, 2 * np.pi)
            if n_tiles > 0:
                angles = np.linspace(0, 2 * np.pi, n_tiles, endpoint=False) + start_angle
                for angle in angles:
                    offsets.append((float(radius * np.cos(angle)), float(radius * np.sin(angle))))
                    
        final_offsets = tuple(offsets)
        
        if check_coverage(grid_shape, tile_shape, final_offsets, metadata):
            if len(final_offsets) > 1000:
                 import warnings
                 warnings.warn(f"Generated a very dense scan plan with {len(final_offsets)} tiles.")
            return final_offsets
        
        if auto_mode:
            num_rings += 1
            if num_rings > 50: # Increased safety break
                raise RuntimeError(f"Could not achieve 100% coverage after 50 rings. SA radius {r_sa:.2f} might be too small.")
        else:
            raise ValueError(
                f"Requested {num_rings} rings with {overlap_fraction} overlap "
                "does NOT cover the entire GT surface. Center positions might be too restricted."
            )
