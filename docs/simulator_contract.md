# Simulator Contract

## Scope

This document defines the current trusted simulator behavior for the repository foundation phase.

## Coordinate Conventions

- Arrays use `z[y, x]` indexing.
- `x` increases to the right and `y` increases downward.
- `center_xy` is the geometric center of a tile in global pixel-center coordinates.
- `translation_xy` is derived from `center_xy` relative to the geometric center of the global grid.
- The trusted simulator supports float `scan_offsets` with sub-pixel interpolation.
- Bilinear interpolation is used for non-integer tile extraction; exact slicing is used for integer alignment.
- Positive `dx` shifts content right. Positive `dy` shifts content down.
- Rotation is stored as metadata and physically applied using `np.rot90` (limited to 90-degree increments in this phase).

## Data Objects

- `SurfaceTruth` is the global reference field for evaluation.
- `SubApertureObservation` is a local detector tile with known pose in the global frame.
- `ReconstructionSurface` is intended to be a global-frame reconstruction aligned to the truth grid.
- `valid_mask` marks which pixels are physically observed and valid for metric computation.

## `valid_mask` Semantics

- `True` means the detector reports a valid sample at that pixel.
- `False` means the pixel is outside the local tile support or clipped by image borders.
- Invalid pixels must not contribute to geometry or signal metrics.
- The trusted simulator zeroes `z` outside `valid_mask` to keep mask/value alignment explicit.
- Reconstruction support is additionally constrained by the union of observed support when provided.

## Order of Simulation Operations

1. Create the global truth surface.
2. Resolve the commanded tile center in the global frame.
3. Apply `realized_pose_error` (optional metadata `realized_pose_error_std`).
4. Extract the local tile using `extract_tile` (bilinear interpolation for sub-pixel, exact for integer).
5. Apply `reference_bias`, `nuisance_terms` (including tip/tilt/focus), Gaussian noise, outliers, and retrace hook.
6. Zero all pixels outside `valid_mask`.
7. Apply discrete rotation (90-degree increments).

## Determinism

- All stochastic processes (noise, outliers, pose error) use explicit seeds derived from `ScenarioConfig.seed`.
- Zero bias, zero noise, zero outliers, and zero retrace must preserve the identity result exactly for integer offsets.
- No stochastic resampling is used in this phase except for explicitly requested pose error.

## Current Limitations

- Rotation is still restricted to 90-degree increments in this phase.
- Interpolation for sub-pixel offsets introduces minor smoothing (bilinear).
- Geometry metrics are relaxed (e.g. 0.99 IoU) to accommodate interpolation artifacts at mask boundaries.
- Thermal drift and spatial bias are modeled as additive nuisance terms.
- Reconstruction is still a simple place-and-average baseline.
