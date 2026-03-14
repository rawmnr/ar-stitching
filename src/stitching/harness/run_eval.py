"""Small harness for running trusted identity evaluation on versioned scenarios."""

from __future__ import annotations

from pathlib import Path

from stitching.editable.baseline import baseline_integer_unshift_mean
from stitching.contracts import EvalReport, ScenarioConfig
from stitching.trusted.eval.metrics import build_eval_report
from stitching.trusted.simulator.identity import simulate_identity_observations


def run_baseline_eval(scenario_path: str | Path) -> EvalReport:
    """Load a scenario, reconstruct with the baseline, and evaluate that reconstruction."""

    config = ScenarioConfig.from_yaml(scenario_path)
    truth, observations = simulate_identity_observations(config)
    reconstruction = baseline_integer_unshift_mean(observations)
    return build_eval_report(config, truth, reconstruction, runtime_sec=0.0)


def run_identity_eval(scenario_path: str | Path) -> EvalReport:
    """Backward-compatible wrapper for the current baseline evaluation path."""

    return run_baseline_eval(scenario_path)
