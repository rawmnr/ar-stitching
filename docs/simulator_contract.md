# Simulator Contract

## Scope

This document defines the current trusted simulator behavior for the repository simulation bench.

## Coordinate Conventions

- Arrays use `z[y, x]` indexing.
- `x` increases to the right and `y` increases downward.
- `center_xy` is the geometric center of a tile in global pixel-center coordinates.
- The trusted simulator supports float `scan_offsets` with sub-pixel interpolation.
- Bilinear interpolation is used for non-integer tile extraction; exact slicing is used for integer alignment.
- Rotation is stored as metadata and physically applied using `np.rot90` (90-degree increments).

## Data Objects

- `SurfaceTruth`: Global reference field for evaluation.
- `SubApertureObservation`: Local detector tile with known pose, metadata, and nuisance terms.
- `valid_mask`: Marks pixels physically observed and valid for metric computation.

## `valid_mask` and Pupil Semantics

- **Detector Pupil**: Observations can have square (full tile) or circular pupils (metadata `detector_pupil: circular`).
- **Global Masking**: `SurfaceTruth.valid_mask` defines the global field of interest.
- **Clipping**: Observations are clipped by both global and local pupil masks.
- Invalid pixels must not contribute to metrics; the simulator zeroes `z` outside `valid_mask`.

## Order of Simulation Operations

1. **Truth Generation**: Create the global truth surface (Legendre or Zernike).
2. **Pose Resolution**: Resolve commanded tile center and apply `realized_pose_error` (bias, drift, jitter).
3. **Tile Extraction**: Extract local tile via `extract_tile` (bilinear or exact).
4. **Instrument Masking**: Apply detector pupil mask (optional circular).
5. **Low-Frequency Noise**: Add Zernike-based (Fringe Z1-Z15) noise.
6. **Global Drift**: Apply spatial drift field (evaluated at global tile location).
7. **Nuisance terms**: Apply Tip, Tilt, Focus, and DC (piston).
8. **High-Frequency Noise**: Add Gaussian noise and outliers.
9. **Retrace**: Apply scalar and/or slope-dependent retrace distortion.
10. **Finalization**: Zero outside `valid_mask` and apply discrete rotation.

## Determinism

- All stochastic processes (noise, outliers, pose error) use explicit seeds derived from `ScenarioConfig.seed`.
- Zero-perturbation cases (no noise, no bias, integer offsets) preserve the identity result exactly.

## Current Limitations

- Rotation is restricted to 90-degree increments.
- Interpolation introduces minor bilinear smoothing for sub-pixel offsets.
- Higher-order interpolation (bicubic/splines) is not yet supported.
