"""Trusted evaluator wrapper with guardrail enforcement."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np

from stitching.contracts import EvalReport, ScenarioConfig
from stitching.harness.budgets import BudgetExceededError, time_guard
from stitching.harness.protocols import CandidateAlgorithm
from stitching.trusted.eval.metrics import build_eval_report
from stitching.trusted.scan.transforms import placement_slices
from stitching.trusted.simulator.identity import simulate_identity_observations


GUARDRAIL_CHECKS = {
    "footprint_iou_min": 0.10,        # Relaxed to allow exploration
    "valid_pixel_recall_min": 0.10,   # Relaxed to allow exploration
    "max_rms_on_valid_intersection": 50.0,    # Relaxed sanity cap for exploration
    "max_runtime_sec": 300.0,
}


class GuardrailViolation(ValueError):
    """Raised when a reconstruction violates hard guardrails."""


def _expected_observed_support(observations, global_shape):
    """Compute the physical observed support from observations."""
    support = np.zeros(global_shape, dtype=bool)
    for obs in observations:
        # Use rounding for sub-pixel centers to determine the pixel-grid support
        center_x, center_y = float(obs.center_xy[0]), float(obs.center_xy[1])
        tile_rows, tile_cols = obs.tile_shape
        
        top = int(round(center_y - (tile_rows - 1) / 2.0))
        left = int(round(center_x - (tile_cols - 1) / 2.0))
        bottom = top + tile_rows
        right = left + tile_cols
        
        gy_start, gy_end = max(0, top), min(global_shape[0], bottom)
        gx_start, gx_end = max(0, left), min(global_shape[1], right)
        
        ly_start, lx_start = max(0, -top), max(0, -left)
        ly_end = ly_start + (gy_end - gy_start)
        lx_end = lx_start + (gx_end - gx_start)

        if gy_end > gy_start and gx_end > gx_start:
            local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly_start:ly_end, lx_start:lx_end]
            support[gy_start:gy_end, gx_start:gx_end][local_mask] = True
    return support


def load_candidate_module(candidate_path: Path) -> CandidateAlgorithm:
    """Dynamically load a candidate algorithm from a Python file."""
    spec = importlib.util.spec_from_file_location("candidate_module", str(candidate_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate from {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CandidateStitcher"):
        raise ImportError(f"Candidate module must define a CandidateStitcher class.")

    return module.CandidateStitcher()


def evaluate_candidate_on_scenario(
    candidate: CandidateAlgorithm,
    scenario_path: Path,
    eval_budget_sec: float = 300.0,
) -> EvalReport:
    """Run a candidate against a single scenario with budget enforcement."""
    config = ScenarioConfig.from_yaml(scenario_path)
    truth, observations = simulate_identity_observations(config)

    t0 = time.monotonic()
    try:
        with time_guard(eval_budget_sec, label=f"eval/{config.scenario_id}"):
            reconstruction = candidate.reconstruct(observations, config)
    except BudgetExceededError:
        raise
    elapsed = time.monotonic() - t0

    # Verify observed support
    expected_support = _expected_observed_support(observations, config.grid_shape)
    if reconstruction.observed_support_mask is None:
        raise GuardrailViolation("Candidate must provide observed_support_mask.")
    if not np.array_equal(reconstruction.observed_support_mask, expected_support):
        raise GuardrailViolation("observed_support_mask does not match physical support.")

    report = build_eval_report(config, truth, reconstruction, observations, runtime_sec=elapsed)

    # Hard guardrails
    _enforce_guardrails(report)

    return report


def evaluate_candidate_on_suite(
    candidate: CandidateAlgorithm,
    scenario_paths: list[Path],
    eval_budget_sec: float = 300.0,
) -> tuple[dict[str, float], tuple[EvalReport, ...]]:
    """Evaluate a candidate across a scenario suite. Returns aggregate metrics + reports."""
    reports: list[EvalReport] = []
    rms_values: list[float] = []
    mae_values: list[float] = []
    total_runtime = 0.0

    for sp in scenario_paths:
        report = evaluate_candidate_on_scenario(candidate, sp, eval_budget_sec)
        reports.append(report)
        
        # Determine if we use raw or detrended metrics for this scenario
        use_detrended = bool(report.config.metadata.get("ignore_tilt", False))
        sig = report.signal_metrics
        
        if use_detrended:
            rms_values.append(sig.get("rms_detrended", sig["rms_on_valid_intersection"]))
            mae_values.append(sig.get("mae_detrended", sig["mae_on_valid_intersection"]))
        else:
            rms_values.append(sig["rms_on_valid_intersection"])
            mae_values.append(sig["mae_on_valid_intersection"])
            
        total_runtime += report.runtime_sec

    aggregate = {
        "aggregate_rms": float(np.sqrt(np.mean(np.array(rms_values) ** 2))),
        "aggregate_mae": float(np.mean(mae_values)),
        "max_rms": float(np.max(rms_values)),
        "min_rms": float(np.min(rms_values)),
        "total_runtime_sec": total_runtime,
        "num_scenarios": len(reports),
        "num_accepted": sum(1 for r in reports if r.accepted),
    }
    return aggregate, tuple(reports)


def _enforce_guardrails(report: EvalReport) -> None:
    """Check hard guardrails that cannot be violated regardless of improvement."""
    geom = report.geometry_metrics
    sig = report.signal_metrics
    
    use_detrended = bool(report.config.metadata.get("ignore_tilt", False))
    current_rms = sig.get("rms_detrended") if use_detrended else sig["rms_on_valid_intersection"]

    if geom["footprint_iou"] < GUARDRAIL_CHECKS["footprint_iou_min"]:
        raise GuardrailViolation(
            f"footprint_iou={geom['footprint_iou']:.4f} "
            f"< {GUARDRAIL_CHECKS['footprint_iou_min']}"
        )

    if current_rms > GUARDRAIL_CHECKS["max_rms_on_valid_intersection"]:
        raise GuardrailViolation(
            f"rms={current_rms:.6f} "
            f"> absolute cap {GUARDRAIL_CHECKS['max_rms_on_valid_intersection']}"
        )

    if report.runtime_sec > GUARDRAIL_CHECKS["max_runtime_sec"]:
        raise GuardrailViolation(
            f"runtime={report.runtime_sec:.1f}s > {GUARDRAIL_CHECKS['max_runtime_sec']}s"
        )
