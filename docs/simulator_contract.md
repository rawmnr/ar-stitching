# Simulator Contract

## Scope

This document defines the current trusted simulator behavior for the repository foundation phase.

## Coordinate Conventions

- Arrays use `z[y, x]` indexing.
- `x` increases to the right and `y` increases downward.
- `center_xy` is the geometric center of a tile in global pixel-center coordinates.
- `translation_xy` is derived from `center_xy` relative to the geometric center of the global grid.
- The current scaffold rounds offsets to integer pixels before simulation.
- Positive `dx` shifts content right. Positive `dy` shifts content down.
- Rotation is stored as metadata only. No rotation resampling is applied yet.

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
2. Choose a local detector tile shape and tile center in the global frame.
3. Extract the corresponding global window into a local tile with integer clipping.
4. Apply reference bias, nuisance terms, Gaussian noise, outliers, and retrace hook on the local tile.
5. Zero all pixels outside `valid_mask`.

## Determinism

- Gaussian noise and outliers use explicit seeds derived from `ScenarioConfig.seed`.
- Zero bias, zero noise, zero outliers, and zero retrace must preserve the identity result exactly.
- No sub-pixel interpolation or stochastic resampling is used in this phase.

## Current Limitations

- The truth surface is a deterministic low-order test surface, not a realistic optical process model.
- Detector tiles are integer-placed local windows, not physically sampled sub-apertures.
- Translation is integer-only; no sub-pixel motion model exists yet.
- Rotation is not physically applied.
- Instrument effects are placeholder additive hooks, not calibrated physical models.
- Reconstruction is still a simple place-and-average baseline.
