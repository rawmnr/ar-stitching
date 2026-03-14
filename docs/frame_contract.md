# Frame Contract

## Scope

This document fixes the intended coordinate-frame contract for observations and reconstructions during repository foundation.

## Objects and Frames

- `SurfaceTruth` lives on the global truth grid.
- `SubApertureObservation` is a detector-frame measurement represented on the current shared array canvas.
- `ReconstructionSurface` is intended to live in the global truth frame and be directly comparable to `SurfaceTruth`.

## Translation Semantics

- `translation_xy` is detector motion in pixel units as `(dx, dy)`.
- The trusted simulator currently rounds offsets to integers.
- A positive shift means the detector observation is displaced right/down on the shared canvas.
- The current baseline reconstructs by applying the inverse integer shift back into the global truth frame.

## `valid_mask` Semantics

- Observation `valid_mask` marks where the detector measured a valid sample after footprinting and clipping.
- Reconstruction `valid_mask` marks where the reconstruction artifact claims valid support in the global frame.
- Values outside any `valid_mask` must be zero by trusted contract.

## Current Foundation Limitation

- The simulator still uses full-grid arrays for observations instead of smaller detector tiles.
- Because of that, detector-frame and global-frame arrays currently share shape even though they represent different semantics.
- The current baseline inverse-shifts and averages observations, but does not do any sub-pixel registration or advanced merge logic.
- A shifted single observation can still be clipped by the shared canvas, so inverse shifting alone does not guarantee full global-footprint recovery.
