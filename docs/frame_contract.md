# Frame Contract

## Scope

This document defines the coordinate-frame and interpolation contract for observations and reconstructions.

## Objects and Frames

- `SurfaceTruth` lives on the global truth grid.
- `SubApertureObservation` is a local detector tile with its own `tile_shape`, `valid_mask`, and pose in the global frame.
- `ReconstructionSurface` lives in the global truth frame and is directly comparable to `SurfaceTruth`.

## Translation and Interpolation Semantics

- `center_xy` is the geometric center of the detector tile expressed in global pixel-center coordinates.
- **Sub-pixel Support**: The trusted simulator supports float `center_xy`.
- **Bilinear Interpolation**: When `center_xy` is non-integer (or non-half-integer for even tiles), bilinear interpolation is used for both surface values and the mask (thresholded at 0.5).
- **Exact Path**: If `center_xy` aligns perfectly with integer pixel placement, exact slicing is used to avoid interpolation artifacts.
- `translation_xy` is a derived quantity: `center_xy - global_geometric_center`.
- A positive shift means the detector center moves right/down in the global frame.

## Pupil and Mask Semantics

- **Detector Pupil**: Observations can have square (full tile) or circular pupils. The `valid_mask` reflects this instrument constraint.
- **Global Masking**: `SurfaceTruth.valid_mask` defines the global field of interest (e.g., a large circular mirror).
- **Clipping**: Observations are clipped both by the global truth mask and the detector's local pupil mask.

## `valid_mask` Semantics

- Observation `valid_mask` marks where the detector measured a valid sample after footprinting, interpolation, and clipping.
- Reconstruction `valid_mask` marks where the reconstruction artifact claims valid support in the global frame.
- `ReconstructionSurface.observed_support_mask` is the union of physically observed support.
- A reconstruction is not allowed to claim `valid_mask=True` outside `observed_support_mask`.
- Values outside any `valid_mask` must be zero by trusted contract.

## Current Limitations

- Rotation is still restricted to 90-degree increments in this phase.
- No advanced sub-pixel registration (e.g., shift-and-add) or spline interpolation is implemented yet.
