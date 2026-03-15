# Simulator Contract

## Scope

This document defines the current trusted simulator behavior for the repository Digital Twin simulation bench.

## Coordinate Conventions

- Arrays use `z[y, x]` indexing.
- `center_xy` is the geometric center of a tile in global pixel-center coordinates.
- **High-Fidelity Sampling**: Supports float `scan_offsets` and arbitrary `rotation_deg`.
- **Interpolation**: Default is **Bicubic (order 3)** for high-fidelity surface sampling.
- **Geometric Retrace**: Coordinate-based sampling distortion (perturbation of the sampling grid).

## Data Objects

- `SurfaceTruth`: Global reference field for evaluation.
- `SubApertureObservation`: Local detector tile with known pose, metadata, and nuisance terms.
- `valid_mask`: Marks pixels physically observed and valid for metric computation.

## `valid_mask` and NaN Semantics

- **NaN-based storage**: Pixels outside the `valid_mask` are stored as `np.nan` in both `SurfaceTruth` and `SubApertureObservation`.
- **Detector Pupil**: Observations can have square or circular pupils.
- **Edge Roll-off**: Signal attenuation and noise boost at pupil boundaries (soft edges).
- **Clipping**: Observations are clipped by both global and local pupil masks.
- Invalid pixels must not contribute to metrics; the validator enforces that `z` is `NaN` (or `0.0` for compatibility) outside `valid_mask`.

## Automated Scan Plans

The simulator can automatically generate scan offsets if not provided in the YAML:
- **`grid`**: Regular rectangular tiling with guaranteed edge-to-edge coverage.
- **`annular`**: Concentric rings reaching the grid boundaries.
- **`overlap_fraction`**: Controls the minimum overlap between adjacent sub-apertures.

## Order of Simulation Operations

1. **Truth Generation**: Create the global truth surface.
2. **Optical PSF**: Pre-filter the truth with Gaussian smoothing.
3. **Mid-Spatial Ripples**: Add periodic perturbations (polishing marks) to the global surface.
4. **Pose Resolution**: Resolve commanded tile center and apply `realized_pose_error` (bias, drift, jitter).
5. **Multi-mode Drift**: Apply time-varying deformation (Zernike Z4-Z8) to the global surface for the current extraction.
6. **Tile Extraction**: Extract local tile via `extract_tile` with bicubic interpolation and fine rotation.
7. **Geometric Retrace**: Distort the sampling coordinates before extraction.
8. **Instrument Masking**: Apply detector pupil mask and **Edge Roll-off**.
9. **Low-Frequency Noise**: Add Zernike-based (Fringe Z1-Z15) noise.
10. **Global Drift**: Apply spatial drift field.
11. **Nuisance terms**: Apply Tip, Tilt, Focus, and DC (piston).
12. **High-Frequency Noise**: Add Gaussian noise and outliers.
13. **Retrace (Scalar/Slope)**: Apply additive retrace distortions.
14. **Finalization**: Set pixels outside `valid_mask` to `np.nan`.

## Determinism

- All stochastic processes use explicit seeds derived from `ScenarioConfig.seed`.
- Zero-perturbation cases (no noise, integer offsets, no PSF) preserve the identity result exactly.
