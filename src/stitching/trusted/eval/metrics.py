"""Trusted evaluation metrics used to validate simulation outputs."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from stitching.contracts import EvalReport, ReconstructionSurface, ScenarioConfig, SurfaceTruth
from stitching.trusted.noise.models import outlier_magnitude_scale
from stitching.trusted.validation import validate_reconstruction_alignment


GEOMETRY_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "footprint_iou_min": 1.0,
    "valid_pixel_recall_min": 1.0,
    "valid_pixel_precision_min": 1.0,
    "largest_component_ratio_min": 0.999,
    "hole_ratio_max": 1e-6,
}

FLAT_TRUTH_STD_EPS = 1e-12
DEFAULT_SIGNAL_ACCEPTANCE_EPS = 1e-12
SIMULATOR_OUTLIER_MAGNITUDE = 1.0
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
        float(np.max(np.abs(reference), initial=0.0)),
        float(np.max(np.abs(candidate), initial=0.0)),
    )
    return max(EMPTY_INTERSECTION_PENALTY_FLOOR, amplitude)


def signal_metrics(reference: np.ndarray, candidate: np.ndarray, valid_intersection: np.ndarray) -> dict[str, float]:
    """Compute basic signal metrics on the valid overlap only."""

    if not np.any(valid_intersection):
        penalty = _empty_intersection_penalty(reference, candidate)
        return {
            "rms_on_valid_intersection": penalty,
            "mae_on_valid_intersection": penalty,
            "hf_retention": 0.0,
        }

    delta = candidate[valid_intersection] - reference[valid_intersection]
    rms = float(np.sqrt(np.mean(delta**2)))
    mae = float(np.mean(np.abs(delta)))
    hf_retention = _high_frequency_retention(reference, candidate, valid_intersection)
    return {
        "rms_on_valid_intersection": rms,
        "mae_on_valid_intersection": mae,
        "hf_retention": hf_retention,
    }


def signal_acceptance_threshold(config: ScenarioConfig) -> float:
    """Return a conservative MAE budget implied by declared irreducible corruption."""

    noise_budget = 3.0 * float(config.gaussian_noise_std)
    outlier_budget = float(config.outlier_fraction) * SIMULATOR_OUTLIER_MAGNITUDE
    retrace_budget = abs(float(config.retrace_error))
    return max(DEFAULT_SIGNAL_ACCEPTANCE_EPS, noise_budget + outlier_budget + retrace_budget)


def build_eval_report(
    config: ScenarioConfig,
    truth: SurfaceTruth,
    candidate: ReconstructionSurface,
    runtime_sec: float,
) -> EvalReport:
    """Combine geometry and signal metrics into an evaluation report."""

    validate_reconstruction_alignment(candidate)
    if candidate.observed_support_mask is None:
        raise ValueError("ReconstructionSurface must provide observed_support_mask for trusted evaluation.")
    support_violation = bool(np.any(candidate.valid_mask & ~candidate.observed_support_mask))
    geom = geometry_metrics(truth.valid_mask, candidate.valid_mask)
    sig = signal_metrics(truth.z, candidate.z, truth.valid_mask & candidate.valid_mask)
    mae_threshold = max(
        signal_acceptance_threshold(config),
        DEFAULT_SIGNAL_ACCEPTANCE_EPS,
    )
    if config.outlier_fraction > 0.0:
        mae_threshold += float(config.outlier_fraction) * outlier_magnitude_scale(truth.z, truth.valid_mask)
    accepted = (
        not support_violation
        and
        geom["footprint_iou"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["footprint_iou_min"]
        and geom["valid_pixel_recall"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["valid_pixel_recall_min"]
        and geom["valid_pixel_precision"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["valid_pixel_precision_min"]
        and geom["largest_component_ratio"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["largest_component_ratio_min"]
        and geom["hole_ratio"] <= GEOMETRY_ACCEPTANCE_THRESHOLDS["hole_ratio_max"]
        and sig["mae_on_valid_intersection"] <= mae_threshold
    )
    notes: list[str] = []
    if support_violation:
        notes.append("reconstruction_valid_mask_exceeds_observed_support")
    if not np.any(truth.valid_mask & candidate.valid_mask):
        notes.append("empty_valid_intersection_penalized")
    notes.append(f"mae_acceptance_threshold={mae_threshold:.12g}")
    return EvalReport(
        scenario_id=config.scenario_id,
        geometry_metrics=geom,
        signal_metrics=sig,
        runtime_sec=runtime_sec,
        accepted=accepted,
        notes=tuple(notes),
    )
