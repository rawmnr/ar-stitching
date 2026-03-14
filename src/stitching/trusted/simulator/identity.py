"""Trusted identity simulator for early repository validation."""

from __future__ import annotations

import numpy as np

from stitching.contracts import ScenarioConfig, SubApertureObservation, SurfaceTruth
from stitching.trusted.instrument.bias import apply_reference_bias
from stitching.trusted.noise.models import add_gaussian_noise, add_outliers, apply_nuisance_terms, apply_retrace_error
from stitching.trusted.scan.transforms import placement_slices
from stitching.trusted.surface.generation import generate_identity_surface
from stitching.trusted.validation import validate_observation_alignment


def _integer_offset(offset_xy: tuple[float, float]) -> tuple[int, int]:
    """Convert scenario offsets into the integer-pixel shift model used in foundation phase."""

    return int(round(offset_xy[0])), int(round(offset_xy[1]))


def _tile_center(global_shape: tuple[int, int], offset_xy: tuple[float, float]) -> tuple[float, float]:
    """Return the detector-tile center in the global frame."""

    center_x = (global_shape[1] - 1) / 2.0 + offset_xy[0]
    center_y = (global_shape[0] - 1) / 2.0 + offset_xy[1]
    return center_x, center_y


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
        center_xy = _tile_center(config.grid_shape, offset)
        global_y, global_x, local_y, local_x = placement_slices(config.grid_shape, tile_shape, center_xy)
        z = np.zeros(tile_shape, dtype=float)
        valid_mask = np.zeros(tile_shape, dtype=bool)

        z[local_y, local_x] = truth.z[global_y, global_x]
        valid_mask[local_y, local_x] = truth.valid_mask[global_y, global_x]
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
            z = np.zeros(tile_shape, dtype=float)
            z[local_y, local_x] = truth.z[global_y, global_x]

        observations.append(
            SubApertureObservation(
                observation_id=f"{config.scenario_id}-obs-{index:02d}",
                z=z,
                valid_mask=valid_mask,
                tile_shape=tile_shape,
                center_xy=center_xy,
                global_shape=config.grid_shape,
                translation_xy=(float(offset[0]), float(offset[1])),
                rotation_deg=float(config.rotation_deg[min(index, len(config.rotation_deg) - 1)]),
                reference_bias=config.reference_bias,
                nuisance_terms={"subaperture_dc": float(index)},
                metadata={
                    "simulator": "identity",
                    "global_truth_shape": tuple(config.grid_shape),
                    "detector_shift_xy": _integer_offset(offset),
                },
            )
        )
        validate_observation_alignment(observations[-1])

    return truth, tuple(observations)
