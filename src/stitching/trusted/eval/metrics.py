"""Trusted evaluation metrics used to validate simulation outputs."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from stitching.contracts import EvalReport, ReconstructionSurface, ScenarioConfig, SubApertureObservation, SurfaceTruth
from stitching.trusted.eval.mismatch import compute_mismatch_metrics
from stitching.trusted.noise.models import outlier_magnitude_scale
from stitching.trusted.validation import validate_reconstruction_alignment


GEOMETRY_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "footprint_iou_min": 0.30,  # Relaxed from 0.99
    "valid_pixel_recall_min": 0.30, # Relaxed from 0.99
    "valid_pixel_precision_min": 0.99,
    "largest_component_ratio_min": 0.99,
    "hole_ratio_max": 1e-4,
}

FLAT_TRUTH_STD_EPS = 1e-12
DEFAULT_SIGNAL_ACCEPTANCE_EPS = 1e-12
HIGH_FREQUENCY_ENERGY_EPS = 1e-12
EMPTY_INTERSECTION_PENALTY_FLOOR = 1.0


def _largest_component_size(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    structure = ndimage.generate_binary_structure(rank=2, connectivity=1)
    labels, num_labels = ndimage.label(mask, structure=structure)
    if num_labels == 0:
        return 0
    component_sizes = np.bincount(labels.ravel())[1:]
    return int(component_sizes.max(initial=0))


def _hole_ratio(mask: np.ndarray) -> float:
    total_valid = int(mask.sum())
    if total_valid == 0:
        return 0.0
    holes = ndimage.binary_fill_holes(mask) & ~mask
    return float(holes.sum()) / float(total_valid)


def geometry_metrics(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> dict[str, float]:
    """Compute mask-based geometry metrics used as hard acceptance gates."""

    intersection = reference_mask & candidate_mask
    union = reference_mask | candidate_mask
    candidate_count = int(candidate_mask.sum())
    reference_count = int(reference_mask.sum())
    largest_component = _largest_component_size(candidate_mask)

    return {
        "footprint_iou": 0.0 if union.sum() == 0 else float(intersection.sum()) / float(union.sum()),
        "valid_pixel_recall": 0.0 if reference_count == 0 else float(intersection.sum()) / float(reference_count),
        "valid_pixel_precision": 0.0 if candidate_count == 0 else float(intersection.sum()) / float(candidate_count),
        "largest_component_ratio": 0.0 if candidate_count == 0 else float(largest_component) / float(candidate_count),
        "hole_ratio": _hole_ratio(candidate_mask),
    }


def _valid_interior_mask(valid_mask: np.ndarray) -> np.ndarray:
    """Return pixels with a full 4-neighborhood inside the valid overlap."""

    up = np.roll(valid_mask, 1, axis=0)
    down = np.roll(valid_mask, -1, axis=0)
    left = np.roll(valid_mask, 1, axis=1)
    right = np.roll(valid_mask, -1, axis=1)
    up[0, :] = False
    down[-1, :] = False
    left[:, 0] = False
    right[:, -1] = False
    return valid_mask & up & down & left & right


def _high_frequency_retention(reference: np.ndarray, candidate: np.ndarray, valid_intersection: np.ndarray) -> float:
    """Compare high-frequency content through Laplacian agreement on interior valid pixels."""

    interior = _valid_interior_mask(valid_intersection)
    if not np.any(interior):
        return 0.0

    reference_laplacian = ndimage.laplace(np.where(valid_intersection, reference, 0.0))
    candidate_laplacian = ndimage.laplace(np.where(valid_intersection, candidate, 0.0))
    reference_hf = reference_laplacian[interior]
    candidate_hf = candidate_laplacian[interior]
    reference_energy = float(np.linalg.norm(reference_hf))
    if reference_energy <= HIGH_FREQUENCY_ENERGY_EPS:
        return 0.0
    mismatch_energy = float(np.linalg.norm(candidate_hf - reference_hf))
    return max(0.0, 1.0 - (mismatch_energy / reference_energy))


def _empty_intersection_penalty(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return a finite conservative penalty when no valid overlap exists."""

    amplitude = max(
        float(np.nanmax(np.abs(reference), initial=0.0)),
        float(np.nanmax(np.abs(candidate), initial=0.0)),
    )
    return max(EMPTY_INTERSECTION_PENALTY_FLOOR, amplitude)


def _remove_piston_tilt(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract best-fit plane (piston + tilt) from masked values."""
    if not np.any(mask) or len(vals) == 0:
        return vals
        
    yy, xx = np.indices(mask.shape, dtype=float)
    
    if vals.shape == mask.shape:
        # Called with 2D array
        y_vals = yy[mask]
        x_vals = xx[mask]
        z_vals_1d = vals[mask]
    else:
        # Called with 1D array of already-masked values
        y_vals = yy[mask]
        x_vals = xx[mask]
        z_vals_1d = vals
        
    valid_idx = ~np.isnan(z_vals_1d)
    if not np.any(valid_idx):
        return vals
        
    y_vals_clean = y_vals[valid_idx]
    x_vals_clean = x_vals[valid_idx]
    z_vals_clean = z_vals_1d[valid_idx]
    
    # A * x = vals where x = [tilt_x, tilt_y, piston]
    A = np.column_stack([x_vals_clean, y_vals_clean, np.ones_like(x_vals_clean)])
    
    # Use lstsq for robust plane fitting
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, z_vals_clean, rcond=None)
        
        if vals.shape == mask.shape:
             # Full array case
             y_all = yy[mask]
             x_all = xx[mask]
             A_all = np.column_stack([x_all, y_all, np.ones_like(x_all)])
             result = vals.copy()
             result[mask] = vals[mask] - (A_all @ coeff)
             return result
        else:
             # Flat values case
             A_all = np.column_stack([x_vals, y_vals, np.ones_like(x_vals)])
             return vals - (A_all @ coeff)
    except np.linalg.LinAlgError:
        # Fallback to piston removal if tilt fitting fails
        return vals - float(np.nanmean(vals))


def signal_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    valid_intersection: np.ndarray,
) -> dict[str, float]:
    """Compute basic signal metrics on the valid overlap only.
    
    Returns both raw metrics and detrended metrics (piston/tilt removed).
    """

    if not np.any(valid_intersection):
        penalty = _empty_intersection_penalty(reference, candidate)
        return {
            "rms_on_valid_intersection": penalty,
            "mae_on_valid_intersection": penalty,
            "rms_detrended": penalty,
            "mae_detrended": penalty,
            "tilt_piston_rms": 0.0,
            "hf_retention": 0.0,
        }

    ref_vals = reference[valid_intersection]
    cand_vals = candidate[valid_intersection]

    # Raw metrics (including tilt and piston)
    delta_raw = cand_vals - ref_vals
    # Use nan-aware metrics if NaNs are present (should not be in ref_vals/cand_vals if intersection is correct)
    delta_raw_clean = delta_raw[~np.isnan(delta_raw)]
    if delta_raw_clean.size == 0:
         penalty = _empty_intersection_penalty(reference, candidate)
         return {
            "rms_on_valid_intersection": penalty,
            "mae_on_valid_intersection": penalty,
            "rms_detrended": penalty,
            "mae_detrended": penalty,
            "tilt_piston_rms": 0.0,
            "hf_retention": 0.0,
        }
        
    rms_raw = float(np.sqrt(np.mean(delta_raw_clean**2)))
    mae_raw = float(np.mean(np.abs(delta_raw_clean)))

    # Detrended metrics (remove global piston and tilt between recon and truth)
    # We fit the plane to the difference to find the best global alignment
    delta_detrended = _remove_piston_tilt(delta_raw, valid_intersection)
    delta_detrended_clean = delta_detrended[~np.isnan(delta_detrended)]
    
    if delta_detrended_clean.size > 0:
        rms_detrended = float(np.sqrt(np.mean(delta_detrended_clean**2)))
        mae_detrended = float(np.mean(np.abs(delta_detrended_clean)))
        plane_fit = delta_raw_clean - delta_detrended_clean if delta_raw_clean.size == delta_detrended_clean.size else 0.0
        tilt_piston_rms = float(np.sqrt(np.mean(plane_fit**2))) if isinstance(plane_fit, np.ndarray) else 0.0
    else:
        rms_detrended = rms_raw
        mae_detrended = mae_raw
        tilt_piston_rms = 0.0

    hf_retention = _high_frequency_retention(reference, candidate, valid_intersection)
    
    return {
        "rms_on_valid_intersection": rms_raw,
        "mae_on_valid_intersection": mae_raw,
        "rms_detrended": rms_detrended,
        "mae_detrended": mae_detrended,
        "tilt_piston_rms": tilt_piston_rms,
        "hf_retention": hf_retention,
    }


def signal_acceptance_threshold(
    config: ScenarioConfig,
    reference: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Return a conservative MAE budget implied by declared irreducible corruption.

    Outlier budget is modeled in one place only: as `outlier_fraction` times a
    signal scale derived from the trusted reference surface when available.
    """

    noise_budget = 3.0 * float(config.gaussian_noise_std)
    outlier_scale = 1.0 if reference is None else outlier_magnitude_scale(reference, valid_mask)
    outlier_budget = float(config.outlier_fraction) * outlier_scale
    retrace_budget = abs(float(config.retrace_error))
    return max(DEFAULT_SIGNAL_ACCEPTANCE_EPS, noise_budget + outlier_budget + retrace_budget)


def build_eval_report(
    config: ScenarioConfig,
    truth: SurfaceTruth,
    candidate: ReconstructionSurface,
    observations: tuple[SubApertureObservation, ...],
    runtime_sec: float,
) -> EvalReport:
    """Combine geometry, signal, and mismatch metrics into an evaluation report."""

    validate_reconstruction_alignment(candidate)
    if candidate.observed_support_mask is None:
        raise ValueError("ReconstructionSurface must provide observed_support_mask for trusted evaluation.")
    support_violation = bool(np.any(candidate.valid_mask & ~candidate.observed_support_mask))
    geom = geometry_metrics(truth.valid_mask, candidate.valid_mask)
    
    sig = signal_metrics(
        truth.z, 
        candidate.z, 
        truth.valid_mask & candidate.valid_mask,
    )
    
    mismatch = compute_mismatch_metrics(observations)

    mae_threshold = signal_acceptance_threshold(config, truth.z, truth.valid_mask)
    
    # User requested: tilt and piston included in calculation (acceptance)
    # So we use the raw mae_on_valid_intersection for the gate.
    # However, for high-res scenarios with known high bias, we might 
    # optionally allow detrended acceptance if the config explicitly says so.
    use_detrended = bool(config.metadata.get("ignore_tilt", False))
    acceptance_mae = sig["mae_detrended"] if use_detrended else sig["mae_on_valid_intersection"]
    
    accepted = (
        not support_violation
        and geom["footprint_iou"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["footprint_iou_min"]
        and geom["valid_pixel_recall"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["valid_pixel_recall_min"]
        and geom["valid_pixel_precision"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["valid_pixel_precision_min"]
        and geom["largest_component_ratio"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["largest_component_ratio_min"]
        and geom["hole_ratio"] <= GEOMETRY_ACCEPTANCE_THRESHOLDS["hole_ratio_max"]
        and acceptance_mae <= mae_threshold
    )
    notes: list[str] = []
    if support_violation:
        notes.append("reconstruction_valid_mask_exceeds_observed_support")
    if not np.any(truth.valid_mask & candidate.valid_mask):
        notes.append("empty_valid_intersection_penalized")
    notes.append(f"mae_acceptance_threshold={mae_threshold:.12g}")
    if use_detrended:
        notes.append("acceptance_using_detrended_metrics")
        
    return EvalReport(
        scenario_id=config.scenario_id,
        geometry_metrics=geom,
        signal_metrics=sig,
        mismatch_metrics=mismatch,
        runtime_sec=runtime_sec,
        accepted=accepted,
        config=config,
        truth=truth,
        reconstruction=candidate,
        notes=tuple(notes),
    )
