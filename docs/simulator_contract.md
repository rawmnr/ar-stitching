# Simulator Contract

## Scope

This document defines the current trusted simulator behavior for the repository foundation phase.

## Coordinate Conventions

- Arrays use `z[y, x]` indexing.
- `x` increases to the right and `y` increases downward.
- `translation_xy` is expressed in detector-frame pixel units as `(dx, dy)`.
- The current scaffold rounds offsets to integer pixels before simulation.
- Positive `dx` shifts content right. Positive `dy` shifts content down.
- Rotation is stored as metadata only. No rotation resampling is applied yet.

## Data Objects

- `SurfaceTruth` is the global reference field for evaluation.
- `SubApertureObservation` is a detector-frame measurement after footprinting and integer translation.
- `ReconstructionSurface` is intended to be a global-frame reconstruction aligned to the truth grid.
- `valid_mask` marks which pixels are physically observed and valid for metric computation.

## `valid_mask` Semantics

- `True` means the detector reports a valid sample at that pixel.
- `False` means the pixel is outside the shifted pupil footprint or clipped by image borders.
- Invalid pixels must not contribute to geometry or signal metrics.
- The trusted simulator zeroes `z` outside `valid_mask` to keep mask/value alignment explicit.

## Order of Simulation Operations

1. Create the global truth surface.
2. Create a centered circular detector footprint.
3. Apply the footprint to form detector-frame values and detector-frame validity.
4. Apply the same integer translation to both values and `valid_mask`.
5. Apply reference bias, nuisance terms, Gaussian noise, outliers, and retrace hook.
6. Zero all pixels outside `valid_mask`.

## Determinism

- Gaussian noise and outliers use explicit seeds derived from `ScenarioConfig.seed`.
- Zero bias, zero noise, zero outliers, and zero retrace must preserve the identity result exactly.
- No sub-pixel interpolation or stochastic resampling is used in this phase.

## Current Limitations

- The truth surface is a deterministic low-order test surface, not a realistic optical process model.
- The footprint is circular but fixed and centered before translation.
- Translation is integer-only; no sub-pixel motion model exists yet.
- Rotation is not physically applied.
- Instrument effects are placeholder additive hooks, not calibrated physical models.
- Observation arrays and reconstruction arrays still share the same canvas shape in this scaffold.
- Shifted observations can lose border coverage by clipping on that shared canvas.
