"""Trusted identity simulator for early repository validation."""

from __future__ import annotations

import numpy as np

from stitching.contracts import ScenarioConfig, SubApertureObservation, SurfaceTruth
from stitching.trusted.instrument.bias import apply_reference_bias
from stitching.trusted.noise.models import add_gaussian_noise, add_outliers, apply_nuisance_terms, apply_retrace_error
from stitching.trusted.scan.transforms import apply_integer_shift
from stitching.trusted.surface.footprint import circular_pupil_mask
from stitching.trusted.surface.generation import generate_identity_surface
from stitching.trusted.validation import validate_observation_alignment


def _integer_offset(offset_xy: tuple[float, float]) -> tuple[int, int]:
    """Convert scenario offsets into the integer-pixel shift model used in foundation phase."""

    return int(round(offset_xy[0])), int(round(offset_xy[1]))


def _shift_mask(mask: np.ndarray, shift_xy: tuple[int, int]) -> np.ndarray:
    """Shift a boolean mask using the same operator as detector values."""

    return apply_integer_shift(mask.astype(np.uint8), shift_xy).astype(bool)


def simulate_identity_observations(
    config: ScenarioConfig,
) -> tuple[SurfaceTruth, tuple[SubApertureObservation, ...]]:
    """Simulate identity observations without any stitching optimization."""

    truth = generate_identity_surface(config.grid_shape, config.pixel_size)
    detector_footprint = circular_pupil_mask(config.grid_shape)
    truth = SurfaceTruth(
        z=np.where(detector_footprint, truth.z, 0.0),
        valid_mask=np.array(detector_footprint, copy=True),
        pixel_size=truth.pixel_size,
        units=truth.units,
        metadata={"global_truth_shape": tuple(config.grid_shape), "detector_footprint": "circular"},
    )
    observations: list[SubApertureObservation] = []

    for index, offset in enumerate(config.scan_offsets):
        shift_xy = _integer_offset(offset)
        detector_values = np.where(detector_footprint, truth.z, 0.0)
        detector_mask = np.array(detector_footprint, copy=True)

        z = apply_integer_shift(detector_values, shift_xy)
        valid_mask = _shift_mask(detector_mask, shift_xy)
        z = apply_reference_bias(z, config.reference_bias)
        z = apply_nuisance_terms(z, {"subaperture_dc": float(index)})
        z = add_gaussian_noise(z, config.gaussian_noise_std, seed=config.seed + index)
        z = add_outliers(z, config.outlier_fraction, magnitude=1.0, seed=config.seed + index)
        z = apply_retrace_error(z, config.retrace_error)
        z = np.where(valid_mask, z, 0.0)

        if (
            config.reference_bias == 0.0
            and config.gaussian_noise_std == 0.0
            and config.outlier_fraction == 0.0
            and config.retrace_error == 0.0
        ):
            z = apply_integer_shift(detector_values, shift_xy)

        observations.append(
            SubApertureObservation(
                observation_id=f"{config.scenario_id}-obs-{index:02d}",
                z=z,
                valid_mask=valid_mask,
                translation_xy=(float(offset[0]), float(offset[1])),
                rotation_deg=float(config.rotation_deg[min(index, len(config.rotation_deg) - 1)]),
                reference_bias=config.reference_bias,
                nuisance_terms={"subaperture_dc": float(index)},
                metadata={
                    "simulator": "identity",
                    "global_truth_shape": tuple(config.grid_shape),
                    "detector_shift_xy": shift_xy,
                },
            )
        )
        validate_observation_alignment(observations[-1])

    return truth, tuple(observations)
