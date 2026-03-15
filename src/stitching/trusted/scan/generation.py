"""Trusted scan plan generation for automated pattern coverage."""

from __future__ import annotations

import numpy as np


def generate_grid_scan_plan(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Generate a grid of offsets that guarantees full coverage of the grid_shape.

    Calculates the required number of tiles to ensure at least the requested overlap
    while reaching the exact edges of the global grid.
    """

    rows, cols = grid_shape
    t_rows, t_cols = tile_shape
    
    def calc_steps(total, tile, overlap):
        if total <= tile:
            return [0.0]
        # Required tiles N: tile + (N-1) * tile * (1 - overlap) >= total
        # N-1 >= (total - tile) / (tile * (1 - overlap))
        n_tiles = int(np.ceil(1 + (total - tile) / (tile * (1.0 - overlap))))
        n_tiles = max(n_tiles, 2)
        # Spread centers between -(total-tile)/2 and (total-tile)/2
        max_offset = (total - tile) / 2.0
        return np.linspace(-max_offset, max_offset, n_tiles)

    y_offsets = calc_steps(rows, t_rows, overlap_fraction)
    x_offsets = calc_steps(cols, t_cols, overlap_fraction)
    
    offsets = []
    for y in y_offsets:
        for x in x_offsets:
            offsets.append((float(x), float(y)))
            
    return tuple(offsets)


def generate_annular_scan_plan(
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float = 0.2,
    num_rings: int = 2,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    """Generate an annular pattern that reaches the grid boundaries."""

    rows, cols = grid_shape
    t_rows, t_cols = tile_shape
    
    # Max radius to touch the edges
    max_radius_y = (rows - t_rows) / 2.0
    max_radius_x = (cols - t_cols) / 2.0
    max_radius = min(max_radius_x, max_radius_y)
    
    if max_radius <= 0:
        return ((0.0, 0.0),)

    offsets = [(0.0, 0.0)] # Center
    
    for ring in range(1, num_rings + 1):
        # Scale radius linearly
        radius = max_radius * (ring / num_rings)
        
        # Step size to maintain overlap along the arc
        # We take the smaller dimension of the tile for safety
        arc_step = min(t_rows, t_cols) * (1.0 - overlap_fraction)
        
        # Circumference of the ring
        circumference = 2 * np.pi * radius
        num_tiles = int(np.ceil(circumference / arc_step))
        num_tiles = max(num_tiles, 4 * ring) # Ensure density increases with radius
        
        rng = np.random.default_rng(seed + ring)
        start_angle = rng.uniform(0, 2 * np.pi)
        
        angles = np.linspace(0, 2 * np.pi, num_tiles, endpoint=False) + start_angle
        for angle in angles:
            offsets.append((float(radius * np.cos(angle)), float(radius * np.sin(angle))))
            
    return tuple(offsets)
