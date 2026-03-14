# Frame Contract

## Scope

This document fixes the intended coordinate-frame contract for observations and reconstructions during repository foundation.

## Objects and Frames

- `SurfaceTruth` lives on the global truth grid.
- `SubApertureObservation` is a local detector tile with its own `tile_shape`, `valid_mask`, and pose in the global frame.
- `ReconstructionSurface` is intended to live in the global truth frame and be directly comparable to `SurfaceTruth`.

## Translation Semantics

- `translation_xy` is detector motion in pixel units as `(dx, dy)`.
- `center_xy` is the detector-tile center expressed in global pixel coordinates.
- The trusted simulator currently rounds offsets to integers.
- A positive shift means the detector center moves right/down in the global frame.
- The current baseline reconstructs by placing each local tile into the global truth frame and averaging overlaps.

## `valid_mask` Semantics

- Observation `valid_mask` marks where the detector measured a valid sample after footprinting and clipping.
- Reconstruction `valid_mask` marks where the reconstruction artifact claims valid support in the global frame.
- Values outside any `valid_mask` must be zero by trusted contract.

## Current Foundation Limitation

- The current simulator uses integer tile placement only.
- The current baseline places and averages tiles, but does not do any sub-pixel registration or advanced merge logic.
- Rotation is still metadata only.
