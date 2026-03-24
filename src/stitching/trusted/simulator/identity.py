"""Trusted identity simulator for early repository validation."""

from __future__ import annotations

import math

import numpy as np

from stitching.contracts import ScenarioConfig, SubApertureObservation, SurfaceTruth
from stitching.trusted.instrument.bias import (
    apply_reference_bias,
    generate_reference_bias_field,
    reference_bias_for_observation,
)
from stitching.trusted.noise.models import (
    add_gaussian_noise,
    add_low_frequency_noise,
    add_mid_spatial_ripples,
    add_outliers,
    apply_edge_degradation,
    apply_global_drift,
    apply_nuisance_terms,
    apply_optical_psf,
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

    # Robustness evaluation alignment terms (Random per sub-aperture)
    # alignment_term: list of indices (0: Piston, 1: Tip, 2: Tilt, 3: Focus)
    # alignment_random_coeff: magnitude of the random variation (standard deviation)
    if "alignment_term" in metadata:
        term_indices = metadata["alignment_term"]
        if isinstance(term_indices, (int, float)):
            term_indices = [int(term_indices)]
            
        coeff = float(metadata.get("alignment_random_coeff", 1.0))
        
        # Mapping: 0=Piston, 1=Tip, 2=Tilt, 3=Focus
        mapping = ["subaperture_dc", "subaperture_tip", "subaperture_tilt", "subaperture_focus"]
        
        for i in term_indices:
            idx = int(i)
            if idx < len(mapping):
                key = mapping[idx]
                # Unique deterministic seed per observation and term index
                rng = np.random.default_rng(config.seed + index + 70_000 + idx * 100)
                val = rng.normal(0.0, coeff)
                terms[key] = terms.get(key, 0.0) + val

    return terms


def simulate_identity_observations(
    config: ScenarioConfig,
) -> tuple[SurfaceTruth, tuple[SubApertureObservation, ...]]:
    """Simulate identity observations without any stitching optimization."""

    truth = generate_identity_surface(config)
    piece_truth_z = np.array(truth.z, copy=True)
    ripple_mag = float(config.metadata.get("mid_spatial_ripple_std", 0.0))
    if ripple_mag > 0.0:
        piece_truth_z = add_mid_spatial_ripples(piece_truth_z, ripple_mag, seed=config.seed + 50_000)
        piece_truth_z = np.where(truth.valid_mask, piece_truth_z, np.nan)

    truth = SurfaceTruth(
        z=piece_truth_z,
        valid_mask=np.array(truth.valid_mask, copy=True),
        pixel_size=truth.pixel_size,
        units=truth.units,
        metadata={
            "global_truth_shape": tuple(config.grid_shape),
            "surface_model": truth.metadata["surface_model"],
        },
    )
    observations: list[SubApertureObservation] = []
    tile_shape = config.effective_tile_shape

    # 1. Pre-filter piece truth with optical PSF (Fill factor / Optical smoothing).
    # The truth surface carries NaNs outside the physical footprint; filtering those
    # NaNs directly would poison the whole image.
    psf_sigma = float(config.metadata.get("optical_psf_sigma", 0.0))
    if psf_sigma > 0.0:
        effective_truth_z = apply_optical_psf(
            np.where(truth.valid_mask, truth.z, 0.0),
            psf_sigma,
        )
        effective_truth_z = np.where(truth.valid_mask, effective_truth_z, np.nan)
    else:
        effective_truth_z = truth.z

    # 3. Pre-calculate the total surface drift field if coefficients are provided
    drift_coeffs = config.metadata.get("surface_drift_coefficients")
    if drift_coeffs is not None:
        from stitching.trusted.bases.zernike import generate_zernike_surface
        drift_surface_max = generate_zernike_surface(np.array(drift_coeffs, dtype=float), config.grid_shape)
    else:
        # Fallback to legacy single scalar focus bending if present
        deform_mag = float(config.metadata.get("surface_bending_drift", 0.0))
        if deform_mag != 0.0:
            yy, xx = np.indices(config.grid_shape, dtype=float)
            ry = 2.0 * (yy - (config.grid_shape[0]-1)/2.0) / (config.grid_shape[0]-1)
            rx = 2.0 * (xx - (config.grid_shape[1]-1)/2.0) / (config.grid_shape[1]-1)
            drift_surface_max = deform_mag * (rx**2 + ry**2)
        else:
            drift_surface_max = None

    # 4. Pre-calculate static instrument reference bias (detector frame)
    ref_bias_coeffs = config.metadata.get("reference_bias_coefficients")
    hf_amplitude = float(config.metadata.get("reference_bias_hf_amplitude", 0.0))
    radius_frac = None
    if config.metadata.get("detector_pupil") == "circular":
        radius_frac = float(config.metadata.get("detector_radius_fraction", 0.45))

    if ref_bias_coeffs is not None or hf_amplitude > 0.0:
        if ref_bias_coeffs is not None:
            ref_bias_coeffs = np.array(ref_bias_coeffs, dtype=float)
        static_inst_bias = generate_reference_bias_field(
            tile_shape, 
            ref_bias_coeffs,
            radius_fraction=radius_frac,
            hf_amplitude=hf_amplitude,
            seed=config.seed + 80_000
        )
    else:
        # If circular, we still might want a zero field with NaNs if the user wants "reference bias" to be masked
        if radius_frac is not None:
            from stitching.trusted.instrument.bias import stationary_reference_bias
            static_inst_bias = stationary_reference_bias(tile_shape, 0.0, radius_fraction=radius_frac)
        else:
            static_inst_bias = 0.0

    for index, offset in enumerate(config.scan_offsets):
        rotation_deg = float(config.rotation_deg[min(index, len(config.rotation_deg) - 1)])
        center_xy = _tile_center(config.grid_shape, offset)
        realized_center_xy = _realized_pose_error(center_xy, config, index)

        # Time-varying Surface Deformation (Slowly varying modes: Focus, Astig, Coma...)
        if drift_surface_max is not None:
            # Linear drift from 0 to drift_surface_max across the sequence
            time_factor = index / max(len(config.scan_offsets) - 1, 1)
            current_truth_z = effective_truth_z + time_factor * drift_surface_max
        else:
            current_truth_z = effective_truth_z

        # Geometric Retrace Distortion
        geom_retrace_mag = float(config.metadata.get("geometric_retrace_error", 0.0))
        perturbation = None
        if geom_retrace_mag != 0.0:
            yy, xx = np.indices(tile_shape, dtype=float)
            ry = 2.0 * (yy - (tile_shape[0]-1)/2.0) / (tile_shape[0]-1)
            rx = 2.0 * (xx - (tile_shape[1]-1)/2.0) / (tile_shape[1]-1)
            perturbation = np.array([geom_retrace_mag * ry, geom_retrace_mag * rx])

        # Extract
        interp_order = int(config.metadata.get("interpolation_order", 3))
        z, valid_mask = extract_tile(
            current_truth_z, 
            truth.valid_mask, 
            tile_shape, 
            realized_center_xy, 
            rotation_deg=rotation_deg,
            interpolation_order=interp_order,
            coordinate_perturbation_xy=perturbation
        )

        # Apply Retrace Error early (physically surface-dependent, not instrument/detector dependent)
        slope_retrace = float(config.metadata.get("slope_retrace_error", 0.0))
        z = apply_retrace_error(z, config.retrace_error, slope_magnitude=slope_retrace)

        # 1. Apply Detector Pupil Mask & Edge Roll-off
        if config.metadata.get("detector_pupil") == "circular":
            from stitching.trusted.surface.footprint import circular_pupil_mask
            radius_frac = float(config.metadata.get("detector_radius_fraction", 0.45))
            detector_mask = circular_pupil_mask(tile_shape, radius_fraction=radius_frac)
            valid_mask = valid_mask & detector_mask
            
            # Edge roll-off (vignetting/SNR drop at boundaries)
            roll_off = float(config.metadata.get("detector_edge_roll_off", 0.0))
            if roll_off > 0.0:
                z, _ = apply_edge_degradation(z, valid_mask, roll_off_width=roll_off, seed=config.seed + index + 40_000)

        # 2. Add Mid-Spatial Ripples (Polishing marks)
        # (Added to current_truth_z before extraction, so no change here)

        # 3. Add Low-Frequency Noise (Z1-Z15 Fringe)
        lf_magnitude = float(config.metadata.get("low_frequency_noise_std", 0.0))
        if lf_magnitude > 0.0:
            z = add_low_frequency_noise(z, lf_magnitude, seed=config.seed + index + 30_000)

        # 3. Global Drift
        drift_coeffs = {k: float(v) for k, v in config.metadata.items() if k.startswith("drift_")}
        z = apply_global_drift(z, realized_center_xy, config.grid_shape, drift_coeffs)

        nuisance_terms = _scenario_nuisance_terms(config, index)
        effective_reference_bias_scalar = reference_bias_for_observation(config.reference_bias, index, config.metadata)
        
        # If circular, we apply the scalar bias as a masked field
        from stitching.trusted.instrument.bias import stationary_reference_bias
        effective_reference_bias = stationary_reference_bias(
            tile_shape, 
            effective_reference_bias_scalar, 
            radius_fraction=radius_frac
        )

        z = apply_reference_bias(z, effective_reference_bias)
        z = apply_reference_bias(z, static_inst_bias) # Static instrument field-dependent bias
        z = apply_nuisance_terms(z, nuisance_terms)
        z = add_gaussian_noise(z, config.gaussian_noise_std, seed=config.seed + index)
        z = add_outliers(z, config.outlier_fraction, magnitude=1.0, seed=config.seed + index, valid_mask=valid_mask)
        
        z = np.where(valid_mask, z, np.nan)

        observations.append(
            SubApertureObservation(
                observation_id=f"{config.scenario_id}-obs-{index:02d}",
                z=z,
                valid_mask=valid_mask,
                tile_shape=tile_shape,
                center_xy=realized_center_xy,
                global_shape=config.grid_shape,
                rotation_deg=rotation_deg,
                reference_bias=effective_reference_bias_scalar,
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
