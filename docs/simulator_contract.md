# Simulator Contract

## Scope

This document defines the current trusted simulator behavior for the repository Digital Twin simulation bench.

## Coordinate Conventions

- Arrays use `z[y, x]` indexing.
- `center_xy` is the geometric center of a tile in global pixel-center coordinates.
- **High-Fidelity Sampling**: Supports float `scan_offsets` and arbitrary `rotation_deg`.
- **Interpolation**: Default is **Bicubic (order 3)** for high-fidelity surface sampling. Bilinear (order 1) is available.
- **Geometric Retrace**: Coordinate-based sampling distortion (perturbation of the sampling grid).

## Data Objects

- `SurfaceTruth`: Global reference field for evaluation.
- `SubApertureObservation`: Local detector tile with known pose, metadata, and nuisance terms.
- `valid_mask`: Marks pixels physically observed and valid for metric computation.

## `valid_mask` and Pupil Semantics

- **Detector Pupil**: Observations can have square or circular pupils.
- **Edge Roll-off**: Signal attenuation and noise boost at pupil boundaries (soft edges).
- **Clipping**: Observations are clipped by both global and local pupil masks.

## Order of Simulation Operations

1. **Truth Generation**: Create the global truth surface.
2. **Optical PSF**: Pre-filter the truth with Gaussian smoothing (simulates optical MTF and fill factor).
3. **Pose Resolution**: Apply `realized_pose_error` (bias, drift, jitter).
4. **Bending Drift**: Perturb the truth with time-varying deformation (e.g., focus drift).
5. **Tile Extraction**: Extract local tile via `extract_tile` with bicubic interpolation and fine rotation.
6. **Geometric Retrace**: Distort the sampling coordinates before extraction.
7. **Instrument Masking**: Apply detector pupil mask and **Edge Roll-off**.
8. **Mid-Spatial Ripples**: Add periodic perturbations (polishing marks).
9. **Low-Frequency Noise**: Add Zernike-based (Fringe Z1-Z15) noise.
10. **Global Drift**: Apply spatial drift field.
11. **Nuisance terms**: Apply Tip, Tilt, Focus, and DC (piston).
12. **High-Frequency Noise**: Add Gaussian noise and outliers.
13. **Retrace (Scalar/Slope)**: Apply additive retrace distortions.
14. **Finalization**: Zero outside `valid_mask`.

## Determinism

- All stochastic processes use explicit seeds derived from `ScenarioConfig.seed`.
- Zero-perturbation cases (no noise, integer offsets, no PSF) preserve the identity result exactly.

## Current Limitations

- Higher-order interpolation (order > 3) is not yet supported.
- Wavefront-based diffraction modeling is replaced by PSF smoothing and edge roll-off proxies.
