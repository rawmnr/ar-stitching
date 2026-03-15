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
    "footprint_iou_min": 0.99,
    "valid_pixel_recall_min": 0.99,
    "max_rms_on_valid_intersection": 10.0,    # absolute sanity cap
    "max_runtime_sec": 300.0,
}


class GuardrailViolation(ValueError):
    """Raised when a reconstruction violates hard guardrails."""


def _expected_observed_support(observations, global_shape):
    """Compute the physical observed support from observations."""
    support = np.zeros(global_shape, dtype=bool)
    for obs in observations:
        gy, gx, ly, lx = placement_slices(obs.global_shape, obs.tile_shape, obs.center_xy)
        local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
        support[gy, gx][local_mask] = True
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
        rms_values.append(report.signal_metrics["rms_on_valid_intersection"])
        mae_values.append(report.signal_metrics["mae_on_valid_intersection"])
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

    if geom["footprint_iou"] < GUARDRAIL_CHECKS["footprint_iou_min"]:
        raise GuardrailViolation(
            f"footprint_iou={geom['footprint_iou']:.4f} "
            f"< {GUARDRAIL_CHECKS['footprint_iou_min']}"
        )

    if sig["rms_on_valid_intersection"] > GUARDRAIL_CHECKS["max_rms_on_valid_intersection"]:
        raise GuardrailViolation(
            f"rms={sig['rms_on_valid_intersection']:.6f} "
            f"> absolute cap {GUARDRAIL_CHECKS['max_rms_on_valid_intersection']}"
        )

    if report.runtime_sec > GUARDRAIL_CHECKS["max_runtime_sec"]:
        raise GuardrailViolation(
            f"runtime={report.runtime_sec:.1f}s > {GUARDRAIL_CHECKS['max_runtime_sec']}s"
        )
