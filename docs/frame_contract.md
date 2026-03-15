# Frame Contract

## Scope

This document defines the coordinate-frame, interpolation, and NaN semantics for observations and reconstructions.

## Objects and Frames

- `SurfaceTruth` lives on the global truth grid.
- `SubApertureObservation` is a local detector tile with its own `tile_shape`, `valid_mask`, and pose in the global frame.
- `ReconstructionSurface` lives in the global truth frame and is directly comparable to `SurfaceTruth`.

## Translation and Interpolation Semantics

- `center_xy`: Geometric center of the detector tile in global pixel-center coordinates.
- **Bicubic Interpolation**: By default, the simulator uses order 3 interpolation for high-fidelity surface sampling.
- **Fine Rotation**: Arbitrary float `rotation_deg` is supported and integrated into the coordinate sampling process.
- **Exact Path**: If `rotation_deg=0` and the center is integer-aligned, exact slicing is used to preserve signal integrity.

## `valid_mask` and NaN Semantics

- **Explicit Missing Data**: The repository uses `np.nan` to represent pixels outside the observed or valid support.
- **SurfaceTruth**: Areas outside the global pupil are `NaN`.
- **Observations**: Pixels outside the local tile or detector pupil are `NaN`.
- **Validation**: The trusted validator enforces that `z[~valid_mask]` is `NaN` (though `0.0` is tolerated for compatibility).
- **Averaging**: Stitching algorithms must use `nanmean` or mask-aware logic to avoid propagating `NaN` into the final reconstruction.

## Reconstruction Constraints

- `ReconstructionSurface.observed_support_mask`: The union of all physically observed pixels.
- A reconstruction **must not** claim valid data (non-NaN) outside this observed support.
- Mismatch diagnostics are only computed where at least two observations overlap.

## Current Limitations

- Higher-order interpolation (e.g., Lanczos) is not yet implemented.
- The global grid is assumed to be uniform and rectangular.
