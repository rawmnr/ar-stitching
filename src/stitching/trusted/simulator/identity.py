"""Trusted identity simulator for early repository validation."""

from __future__ import annotations

import math

import numpy as np

from stitching.contracts import ScenarioConfig, SubApertureObservation, SurfaceTruth
from stitching.trusted.instrument.bias import apply_reference_bias, reference_bias_for_observation
from stitching.trusted.noise.models import (
    add_gaussian_noise,
    add_low_frequency_noise,
    add_outliers,
    apply_global_drift,
    apply_nuisance_terms,
    apply_retrace_error,
)
from stitching.trusted.scan.transforms import extract_tile
from stitching.trusted.surface.generation import generate_identity_surface
from stitching.trusted.validation import validate_observation_alignment


def _safe_offset_metadata(offset_xy: tuple[float, float]) -> tuple[float, float] | tuple[int, int]:
    """Return integer offsets if exact, otherwise return floats."""

    rx, ry = round(float(offset_xy[0])), round(float(offset_xy[1]))
    if math.isclose(float(offset_xy[0]), rx) and math.isclose(float(offset_xy[1]), ry):
        return int(rx), int(ry)
    return float(offset_xy[0]), float(offset_xy[1])


def _tile_center(global_shape: tuple[int, int], offset_xy: tuple[float, float]) -> tuple[float, float]:
    """Return the detector-tile geometric center in global pixel-center coordinates."""

    center_x = (global_shape[1] - 1) / 2.0 + float(offset_xy[0])
    center_y = (global_shape[0] - 1) / 2.0 + float(offset_xy[1])
    return center_x, center_y


def _realized_pose_error(center_xy: tuple[float, float], config: ScenarioConfig, index: int) -> tuple[float, float]:
    """Apply deterministic realized pose error (independent, drift, and bias)."""

    metadata = config.metadata
    cx, cy = float(center_xy[0]), float(center_xy[1])

    # 1. Systematic Bias (Calibration error)
    if "realized_pose_bias_xy" in metadata:
        bias = metadata["realized_pose_bias_xy"]
        cx += float(bias[0])
        cy += float(bias[1])

    # 2. Correlated Drift (Slow mechanical/thermal drift)
    drift_std = float(metadata.get("realized_pose_drift_std", 0.0))
    if drift_std > 0.0:
        # Use a cumulative random walk based on scan index
        rng_drift = np.random.default_rng(config.seed + 20_000)
        # We generate all steps up to index and sum them for a stable walk
        steps = rng_drift.normal(0.0, drift_std, size=(index + 1, 2))
        drift_xy = np.sum(steps, axis=0)
        cx += drift_xy[0]
        cy += drift_xy[1]

    # 3. Independent Noise (Random jitter)
    noise_std = float(metadata.get("realized_pose_error_std", 0.0))
    if noise_std > 0.0:
        rng_noise = np.random.default_rng(config.seed + index + 10_000)
        cx += rng_noise.normal(0.0, noise_std)
        cy += rng_noise.normal(0.0, noise_std)

    return cx, cy


def _apply_discrete_rotation(values: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply exact quarter-turn rotations without interpolation."""

    normalized = float(angle_deg) % 360.0
    quarter_turns = normalized / 90.0
    rounded = int(round(quarter_turns))
    if not np.isclose(quarter_turns, rounded):
        raise ValueError("Trusted simulator only supports rotations in multiples of 90 degrees.")
    return np.rot90(values, k=rounded)


def _scenario_nuisance_terms(config: ScenarioConfig, index: int) -> dict[str, float]:
    """Return explicit nuisance terms for one observation.

    Nuisance is opt-in. Identity-like scenarios should stay clean unless a scenario
    explicitly requests a detector-fixed or per-observation term.
    """

    metadata = config.metadata
    terms: dict[str, float] = {}

    # DC / Piston
    if "subaperture_dc_values" in metadata:
        values = tuple(float(value) for value in metadata["subaperture_dc_values"])
        terms["subaperture_dc"] = values[index] if index < len(values) else 0.0
    elif "subaperture_dc_step" in metadata:
        terms["subaperture_dc"] = float(metadata["subaperture_dc_step"]) * float(index)
    elif "subaperture_dc" in metadata:
        terms["subaperture_dc"] = float(metadata["subaperture_dc"])
    else:
        terms["subaperture_dc"] = 0.0

    # Tip/Tilt/Focus
    for key in ("subaperture_tip", "subaperture_tilt", "subaperture_focus"):
        if key in metadata:
            terms[key] = float(metadata[key])

    return terms


def simulate_identity_observations(
    config: ScenarioConfig,
) -> tuple[SurfaceTruth, tuple[SubApertureObservation, ...]]:
    """Simulate identity observations without any stitching optimization."""

    truth = generate_identity_surface(config)
    truth = SurfaceTruth(
        z=np.array(truth.z, copy=True),
        valid_mask=np.array(truth.valid_mask, copy=True),
        pixel_size=truth.pixel_size,
        units=truth.units,
        metadata={"global_truth_shape": tuple(config.grid_shape), "surface_model": truth.metadata["surface_model"]},
    )
    observations: list[SubApertureObservation] = []
    tile_shape = config.effective_tile_shape

    for index, offset in enumerate(config.scan_offsets):
        rotation_deg = float(config.rotation_deg[min(index, len(config.rotation_deg) - 1)])
        center_xy = _tile_center(config.grid_shape, offset)
        realized_center_xy = _realized_pose_error(center_xy, config, index)

        z, valid_mask = extract_tile(truth.z, truth.valid_mask, tile_shape, realized_center_xy)

        # 1. Apply Detector Pupil Mask (Instrument constraint)
        if config.metadata.get("detector_pupil") == "circular":
            from stitching.trusted.surface.footprint import circular_pupil_mask
            radius_frac = float(config.metadata.get("detector_radius_fraction", 0.45))
            detector_mask = circular_pupil_mask(tile_shape, radius_fraction=radius_frac)
            valid_mask = valid_mask & detector_mask
            z = np.where(valid_mask, z, 0.0)

        # 2. Add Low-Frequency Noise (Z1-Z15 Fringe)
        lf_magnitude = float(config.metadata.get("low_frequency_noise_std", 0.0))
        if lf_magnitude > 0.0:
            z = add_low_frequency_noise(z, lf_magnitude, seed=config.seed + index + 30_000)

        # 3. Global Drift
        drift_coeffs = {k: float(v) for k, v in config.metadata.items() if k.startswith("drift_")}
        z = apply_global_drift(z, realized_center_xy, config.grid_shape, drift_coeffs)

        nuisance_terms = _scenario_nuisance_terms(config, index)
        effective_reference_bias = reference_bias_for_observation(config.reference_bias, index, config.metadata)

        z = apply_reference_bias(z, effective_reference_bias)
        z = apply_nuisance_terms(z, nuisance_terms)
        z = add_gaussian_noise(z, config.gaussian_noise_std, seed=config.seed + index)
        z = add_outliers(z, config.outlier_fraction, magnitude=1.0, seed=config.seed + index, valid_mask=valid_mask)
        
        slope_retrace = float(config.metadata.get("slope_retrace_error", 0.0))
        z = apply_retrace_error(z, config.retrace_error, slope_magnitude=slope_retrace)
        
        z = np.where(valid_mask, z, 0.0)
        z = _apply_discrete_rotation(z, rotation_deg)
        valid_mask = _apply_discrete_rotation(valid_mask, rotation_deg).astype(bool)

        observations.append(
            SubApertureObservation(
                observation_id=f"{config.scenario_id}-obs-{index:02d}",
                z=z,
                valid_mask=valid_mask,
                tile_shape=tile_shape,
                center_xy=realized_center_xy,
                global_shape=config.grid_shape,
                rotation_deg=rotation_deg,
                reference_bias=effective_reference_bias,
                nuisance_terms=nuisance_terms,
                metadata={
                    "simulator": "identity",
                    "global_truth_shape": tuple(config.grid_shape),
                    "commanded_offset_xy": offset,
                    "detector_shift_xy": _safe_offset_metadata(offset),
                },
            )
        )
        validate_observation_alignment(observations[-1])

    return truth, tuple(observations)
