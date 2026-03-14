# Frame Contract

## Scope

This document fixes the intended coordinate-frame contract for observations and reconstructions during repository foundation.

## Objects and Frames

- `SurfaceTruth` lives on the global truth grid.
- `SubApertureObservation` is a local detector tile with its own `tile_shape`, `valid_mask`, and pose in the global frame.
- `ReconstructionSurface` is intended to live in the global truth frame and be directly comparable to `SurfaceTruth`.

## Translation Semantics

- `center_xy` is the geometric center of the detector tile expressed in global pixel-center coordinates.
- Odd tile sizes therefore use integer centers, while even tile sizes use half-integer centers.
- `translation_xy` is a derived quantity: `center_xy - global_geometric_center`.
- The trusted simulator currently rounds offsets to integers.
- A positive shift means the detector center moves right/down in the global frame.
- The current baseline reconstructs by placing each local tile into the global truth frame and averaging overlaps.

## `valid_mask` Semantics

- Observation `valid_mask` marks where the detector measured a valid sample after footprinting and clipping.
- Reconstruction `valid_mask` marks where the reconstruction artifact claims valid support in the global frame.
- `ReconstructionSurface.observed_support_mask`, when present, is the union of physically observed support used to constrain the reconstruction.
- A reconstruction is not allowed to claim `valid_mask=True` outside `observed_support_mask`.
- Values outside any `valid_mask` must be zero by trusted contract.

## Current Foundation Limitation

- The current simulator uses integer tile placement only.
- The current baseline places and averages tiles, but does not do any sub-pixel registration or advanced merge logic.
- Rotation is still metadata only.
