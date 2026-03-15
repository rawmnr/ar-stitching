# Metrics Contract

## Scope

This document defines the current trusted evaluation contract for geometry, signal, and mismatch metrics.

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

- `footprint_iou >= 0.99` (Relaxed from 1.0 to support sub-pixel interpolation artifacts)
- `valid_pixel_recall >= 0.99`
- `valid_pixel_precision >= 0.99`
- `largest_component_ratio >= 0.99`
- `hole_ratio <= 1e-4`
- `mae_on_valid_intersection <= signal_acceptance_threshold(...)`

These gates are intentionally strict to ensure geometry and footprint integrity are first-order constraints.

## Signal Metrics

Signal metrics are computed only on `truth.valid_mask & candidate.valid_mask`:

- `rms_on_valid_intersection`
- `mae_on_valid_intersection`
- `hf_retention` (Laplacian-agreement proxy for high-frequency retention)

Pixels outside the valid overlap are excluded from signal scoring by contract.

## Mismatch Diagnostics (Internal Consistency)

Mismatch metrics assess the internal consistency of the reconstruction in overlap regions without requiring ground truth:

- `mismatch_rms` (RMS disagreement between overlapping observations)
- `mismatch_mean`
- `mismatch_median`
- `mismatch_max`
- `mismatch_p95` (95th percentile of per-pixel mismatch)

Mismatch is defined as the standard deviation of valid observation contributions at each global pixel. Observations are binned to the nearest integer-compatible global pixel to avoid re-interpolation artifacts in this diagnostic.

## Why RMS Alone Is Insufficient

- RMS can look good while the predicted footprint is too small.
- RMS can improve if invalid or difficult regions are silently dropped.
- RMS does not detect holes, fragmentation, or disconnected valid regions.
- RMS does not protect high-frequency content on its own.
- Mismatch detects internal alignment errors even when ground truth is unavailable or bias is present.

## Known Current Limitations

- `hf_retention` is a lightweight Laplacian-agreement proxy, not a full optical MTF measure.
- Sub-pixel registration logic is now supported, which relaxes geometry thresholds slightly.
- The current truth surface is still low-order and deterministic, so richer optical failure modes are not yet stressed.
