"""Trusted identity simulator for early repository validation."""

from __future__ import annotations

import numpy as np

from stitching.contracts import ScenarioConfig, SubApertureObservation, SurfaceTruth
from stitching.trusted.instrument.bias import apply_reference_bias, reference_bias_for_observation
from stitching.trusted.noise.models import add_gaussian_noise, add_outliers, apply_nuisance_terms, apply_retrace_error
from stitching.trusted.scan.transforms import placement_slices
from stitching.trusted.surface.generation import generate_identity_surface
from stitching.trusted.validation import validate_observation_alignment


def _integer_offset(offset_xy: tuple[float, float]) -> tuple[int, int]:
    """Convert scenario offsets into the integer-pixel shift model used in foundation phase."""

    rounded_x = round(float(offset_xy[0]))
    rounded_y = round(float(offset_xy[1]))
    if not np.isclose(float(offset_xy[0]), rounded_x) or not np.isclose(float(offset_xy[1]), rounded_y):
        raise ValueError("Trusted simulator only supports integer scan offsets in foundation phase.")
    return int(rounded_x), int(rounded_y)


def _tile_center(global_shape: tuple[int, int], offset_xy: tuple[float, float]) -> tuple[float, float]:
    """Return the detector-tile geometric center in global pixel-center coordinates."""

    center_x = (global_shape[1] - 1) / 2.0 + offset_xy[0]
    center_y = (global_shape[0] - 1) / 2.0 + offset_xy[1]
    return center_x, center_y


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
    if "subaperture_dc_values" in metadata:
        values = tuple(float(value) for value in metadata["subaperture_dc_values"])
        if index >= len(values):
            raise ValueError("subaperture_dc_values must provide one value per observation.")
        return {"subaperture_dc": values[index]}
    if "subaperture_dc_step" in metadata:
        return {"subaperture_dc": float(metadata["subaperture_dc_step"]) * float(index)}
    if "subaperture_dc" in metadata:
        return {"subaperture_dc": float(metadata["subaperture_dc"])}
    return {"subaperture_dc": 0.0}


def simulate_identity_observations(
    config: ScenarioConfig,
) -> tuple[SurfaceTruth, tuple[SubApertureObservation, ...]]:
    """Simulate identity observations without any stitching optimization."""

    truth = generate_identity_surface(config.grid_shape, config.pixel_size)
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
        global_y, global_x, local_y, local_x = placement_slices(config.grid_shape, tile_shape, center_xy)
        z = np.zeros(tile_shape, dtype=float)
        valid_mask = np.zeros(tile_shape, dtype=bool)
        nuisance_terms = _scenario_nuisance_terms(config, index)
        effective_reference_bias = reference_bias_for_observation(config.reference_bias, index, config.metadata)

        z[local_y, local_x] = truth.z[global_y, global_x]
        valid_mask[local_y, local_x] = truth.valid_mask[global_y, global_x]
        z = apply_reference_bias(z, effective_reference_bias)
        z = apply_nuisance_terms(z, nuisance_terms)
        z = add_gaussian_noise(z, config.gaussian_noise_std, seed=config.seed + index)
        z = add_outliers(z, config.outlier_fraction, magnitude=1.0, seed=config.seed + index, valid_mask=valid_mask)
        z = apply_retrace_error(z, config.retrace_error)
        z = np.where(valid_mask, z, 0.0)
        z = _apply_discrete_rotation(z, rotation_deg)
        valid_mask = _apply_discrete_rotation(valid_mask, rotation_deg).astype(bool)

        observations.append(
            SubApertureObservation(
                observation_id=f"{config.scenario_id}-obs-{index:02d}",
                z=z,
                valid_mask=valid_mask,
                tile_shape=tile_shape,
                center_xy=center_xy,
                global_shape=config.grid_shape,
                rotation_deg=rotation_deg,
                reference_bias=effective_reference_bias,
                nuisance_terms=nuisance_terms,
                metadata={
                    "simulator": "identity",
                    "global_truth_shape": tuple(config.grid_shape),
                    "detector_shift_xy": _integer_offset(offset),
                },
            )
        )
        validate_observation_alignment(observations[-1])

    return truth, tuple(observations)
