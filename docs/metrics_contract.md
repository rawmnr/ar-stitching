# Metrics Contract

## Scope

This document defines the current trusted evaluation contract for geometry and signal metrics.

## Geometry Metrics

The trusted evaluator computes:

- `footprint_iou`
- `valid_pixel_recall`
- `valid_pixel_precision`
- `largest_component_ratio`
- `hole_ratio`

These are computed from `truth.valid_mask` and `candidate.valid_mask`.

## Hard Gates

Geometry is a blocking acceptance layer. A candidate must first preserve the valid footprint.

Current acceptance logic in the scaffold is:

- `footprint_iou >= 1.0`
- `valid_pixel_recall >= 1.0`
- `valid_pixel_precision >= 1.0`
- `largest_component_ratio >= 0.999`
- `hole_ratio <= 1e-6`
- `mae_on_valid_intersection <= signal_acceptance_threshold(...)`

These gates are intentionally conservative for the current deterministic scaffold.

## Signal Metrics

Signal metrics are computed only on `truth.valid_mask & candidate.valid_mask`:

- `rms_on_valid_intersection`
- `mae_on_valid_intersection`
- `hf_retention`

Pixels outside the valid overlap are excluded from signal scoring by contract.

## Why RMS Alone Is Insufficient

- RMS can look good while the predicted footprint is too small.
- RMS can improve if invalid or difficult regions are silently dropped.
- RMS does not detect holes, fragmentation, or disconnected valid regions.
- RMS does not protect high-frequency content on its own.

For this repository, geometry and footprint integrity are first-order constraints, not secondary diagnostics.

## Known Current Limitations

- `hf_retention` is a lightweight Laplacian-agreement proxy, not a full optical MTF measure.
- Metrics assume exact pixel-grid alignment; no sub-pixel registration logic exists yet.
- The current truth surface is still low-order and deterministic, so richer optical failure modes are not yet stressed.
- The median baseline is available for robustness comparisons, but it remains experimental and is not treated as a scalable production-like reference.
