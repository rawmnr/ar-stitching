"""Trusted evaluation metrics used to validate simulation outputs."""

from __future__ import annotations

from collections import deque

import numpy as np

from stitching.contracts import EvalReport, ReconstructionSurface, ScenarioConfig, SurfaceTruth
from stitching.trusted.validation import validate_reconstruction_alignment


GEOMETRY_ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "footprint_iou_min": 0.999,
    "valid_pixel_recall_min": 0.999,
    "valid_pixel_precision_min": 0.999,
    "largest_component_ratio_min": 0.999,
    "hole_ratio_max": 1e-6,
}

FLAT_TRUTH_STD_EPS = 1e-12


def _largest_component_size(mask: np.ndarray) -> int:
    visited = np.zeros_like(mask, dtype=bool)
    best = 0
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        visited[y, x] = True
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        best = max(best, size)
    return best


def _hole_ratio(mask: np.ndarray) -> float:
    inverse = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(mask.shape[1]):
        queue.append((0, x))
        queue.append((mask.shape[0] - 1, x))
    for y in range(mask.shape[0]):
        queue.append((y, 0))
        queue.append((y, mask.shape[1] - 1))

    while queue:
        y, x = queue.popleft()
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue
        if visited[y, x] or not inverse[y, x]:
            continue
        visited[y, x] = True
        queue.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))

    holes = inverse & ~visited
    total_valid = int(mask.sum())
    return 0.0 if total_valid == 0 else float(holes.sum()) / float(total_valid)


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


def signal_metrics(reference: np.ndarray, candidate: np.ndarray, valid_intersection: np.ndarray) -> dict[str, float]:
    """Compute basic signal metrics on the valid overlap only."""

    if not np.any(valid_intersection):
        return {
            "rms_on_valid_intersection": float("inf"),
            "mae_on_valid_intersection": float("inf"),
            "hf_retention": 0.0,
        }

    delta = candidate[valid_intersection] - reference[valid_intersection]
    rms = float(np.sqrt(np.mean(delta**2)))
    mae = float(np.mean(np.abs(delta)))
    ref_std = float(np.std(reference[valid_intersection]))
    cand_std = float(np.std(candidate[valid_intersection]))
    hf_retention = 0.0 if ref_std <= FLAT_TRUTH_STD_EPS else cand_std / ref_std
    return {
        "rms_on_valid_intersection": rms,
        "mae_on_valid_intersection": mae,
        "hf_retention": hf_retention,
    }


def build_eval_report(
    config: ScenarioConfig,
    truth: SurfaceTruth,
    candidate: ReconstructionSurface,
    runtime_sec: float,
) -> EvalReport:
    """Combine geometry and signal metrics into an evaluation report."""

    validate_reconstruction_alignment(candidate)
    geom = geometry_metrics(truth.valid_mask, candidate.valid_mask)
    sig = signal_metrics(truth.z, candidate.z, truth.valid_mask & candidate.valid_mask)
    accepted = (
        geom["footprint_iou"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["footprint_iou_min"]
        and geom["valid_pixel_recall"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["valid_pixel_recall_min"]
        and geom["valid_pixel_precision"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["valid_pixel_precision_min"]
        and geom["largest_component_ratio"] >= GEOMETRY_ACCEPTANCE_THRESHOLDS["largest_component_ratio_min"]
        and geom["hole_ratio"] <= GEOMETRY_ACCEPTANCE_THRESHOLDS["hole_ratio_max"]
        and sig["mae_on_valid_intersection"] <= 1e-12
    )
    return EvalReport(
        scenario_id=config.scenario_id,
        geometry_metrics=geom,
        signal_metrics=sig,
        runtime_sec=runtime_sec,
        accepted=accepted,
        notes=(),
    )
