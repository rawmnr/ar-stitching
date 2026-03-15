# Metrics Contract

## Scope

This document defines the trusted evaluation contract for geometry, signal, and internal consistency (mismatch) metrics.

## Geometry Metrics

Geometry metrics validate the structural integrity of the reconstruction footprint:
- `footprint_iou`: Intersection over Union of valid masks.
- `valid_pixel_recall` / `precision`.
- `largest_component_ratio`: Detects fragmented reconstructions.
- `hole_ratio`: Detects internal voids in the valid area.

**Hard Gate**: `footprint_iou >= 0.99`. This is relaxed from 1.0 to support sub-pixel interpolation artifacts at mask boundaries.

## Signal Metrics (Truth-based)

Signal metrics are computed on the intersection of truth and candidate valid masks:
- `rms_on_valid_intersection`: Root Mean Square error.
- `mae_on_valid_intersection`: Mean Absolute Error.
- `hf_retention`: Laplacian-based proxy for high-frequency preservation.

**Piston Correction**: If `ignore_piston: true` is set in the scenario, the average height is removed from both reference and candidate before scoring.

## Mismatch Diagnostics (Consistency-based)

Mismatch metrics assess internal agreement in overlap regions without requiring ground truth:
- `mismatch_rms` / `mean` / `median` / `p95` / `max`.

Mismatch is defined as the **standard deviation** of all valid observation contributions at a global pixel.
- Observations are binned to the nearest integer global pixel to avoid re-interpolation noise.
- Pixels covered by only one observation have zero mismatch by definition.

## NaN Semantics in Metrics

- All metrics MUST ignore `NaN` values.
- Signal scoring is strictly masked by the `valid_mask` intersection.
- Propagation of a single `NaN` into a summary metric indicates a contract violation in the mask-handling logic.

## Known Current Limitations

- High-frequency retention is a proxy, not a formal MTF transfer function.
- Mismatch diagnostics do not currently account for local tilt correction before binning.
