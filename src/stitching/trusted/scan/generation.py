"""Trusted scan plan generation for automated pattern coverage."""

from __future__ import annotations

import numpy as np


def generate_grid_scan_plan(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Generate a regular grid of sub-aperture offsets with specified overlap."""

    rows, cols = grid_shape
    t_rows, t_cols = tile_shape
    
    # Step size to achieve overlap
    dy = t_rows * (1.0 - overlap_fraction)
    dx = t_cols * (1.0 - overlap_fraction)
    
    # Center-relative offsets
    # Max reach is (grid_shape - tile_shape) / 2
    max_y = (rows - t_rows) / 2.0
    max_x = (cols - t_cols) / 2.0
    
    y_coords = np.arange(-max_y, max_y + 0.1, dy)
    x_coords = np.arange(-max_x, max_x + 0.1, dx)
    
    offsets = []
    for y in y_coords:
        for x in x_coords:
            offsets.append((float(x), float(y)))
            
    return tuple(offsets)


def generate_annular_scan_plan(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float = 0.2,
    num_rings: int = 2,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Generate an annular pattern of sub-aperture offsets."""

    rows, cols = grid_shape
    t_rows, t_cols = tile_shape
    
    max_radius = min(rows - t_rows, cols - t_cols) / 2.0
    if max_radius <= 0:
        return ((0.0, 0.0),)

    offsets = [(0.0, 0.0)] # Always include center
    
    for ring in range(1, num_rings + 1):
        radius = max_radius * (ring / num_rings)
        # Circumference approx 2*pi*radius
        # Number of tiles to maintain overlap along the ring
        arc_step = min(t_rows, t_cols) * (1.0 - overlap_fraction)
        num_tiles = max(4, int(np.ceil(2 * np.pi * radius / arc_step)))
        
        # Add a small random rotation per ring based on seed
        rng = np.random.default_rng(seed + ring)
        start_angle = rng.uniform(0, 2 * np.pi)
        
        angles = np.linspace(0, 2 * np.pi, num_tiles, endpoint=False) + start_angle
        for angle in angles:
            offsets.append((float(radius * np.cos(angle)), float(radius * np.sin(angle))))
            
    return tuple(offsets)
