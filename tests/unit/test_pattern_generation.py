"""Unit tests for automated scan plan generation."""

from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig


def test_grid_pattern_generation() -> None:
    config = ScenarioConfig(
        scenario_id="grid_test",
        description="test",
        grid_shape=(64, 64),
        tile_shape=(32, 32),
        pixel_size=1.0,
        scan_offsets=(), # Empty, should be generated
        seed=0,
        metadata={"pattern_type": "grid", "overlap_fraction": 0.5}
    )
    # Step size should be 32 * 0.5 = 16 pixels
    # Range is -16 to 16. So offsets at -16, 0, 16.
    # Total 3x3 = 9 tiles.
    
    # We need to trigger from_yaml logic or manually call generation
    from stitching.trusted.scan.generation import generate_grid_scan_plan
    offsets = generate_grid_scan_plan(config.grid_shape, config.effective_tile_shape, overlap_fraction=0.5)
    
    assert len(offsets) == 9
    assert (0.0, 0.0) in offsets
    assert (-16.0, -16.0) in offsets
    assert (16.0, 16.0) in offsets


def test_annular_pattern_generation() -> None:
    from stitching.trusted.scan.generation import generate_annular_scan_plan
    grid_shape = (100, 100)
    tile_shape = (20, 20)
    
    offsets = generate_annular_scan_plan(grid_shape, tile_shape, overlap_fraction=0.2, num_rings=1)
    
    # Ring 1 radius should be (100-20)/2 = 40.
    # Circumference 2*pi*40 = 251
    # Step 20 * 0.8 = 16
    # Tiles ~ 251/16 = 15.7 -> 16 tiles + 1 center = 17 tiles.
    assert len(offsets) >= 10
    assert (0.0, 0.0) in offsets


def test_scenario_config_auto_generates_from_payload() -> None:
    import yaml
    from pathlib import Path
    
    yaml_content = """
scenario_id: auto_gen
description: test
grid_shape: [64, 64]
tile_shape: [32, 32]
pixel_size: 1.0
pattern_type: grid
overlap_fraction: 0.5
seed: 0
"""
    tmp_path = Path("temp_scenario.yaml")
    tmp_path.write_text(yaml_content)
    
    try:
        config = ScenarioConfig.from_yaml(tmp_path)
        assert len(config.scan_offsets) == 9
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
