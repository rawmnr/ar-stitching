"""Small harness for running trusted identity evaluation on versioned scenarios."""

from __future__ import annotations

from pathlib import Path

from stitching.editable.baseline import baseline_integer_unshift_mean, baseline_integer_unshift_median
from stitching.contracts import EvalReport, ScenarioConfig
from stitching.trusted.eval.metrics import build_eval_report
from stitching.trusted.simulator.identity import simulate_identity_observations


def _resolve_baseline(baseline_name: str):
    """Resolve a named baseline implementation."""

    baselines = {
        "mean": baseline_integer_unshift_mean,
        "median": baseline_integer_unshift_median,
    }
    if baseline_name not in baselines:
        raise ValueError(f"Unsupported baseline '{baseline_name}'.")
    return baselines[baseline_name]


def run_baseline_eval(scenario_path: str | Path, baseline_name: str | None = None) -> EvalReport:
    """Load a scenario, reconstruct with a selected baseline, and evaluate that reconstruction."""

    config = ScenarioConfig.from_yaml(scenario_path)
    truth, observations = simulate_identity_observations(config)
    selected_baseline = config.baseline_name if baseline_name is None else baseline_name
    reconstruction = _resolve_baseline(selected_baseline)(observations)
    return build_eval_report(config, truth, reconstruction, runtime_sec=0.0)


def run_median_baseline_eval(scenario_path: str | Path) -> EvalReport:
    """Convenience wrapper for the robust median baseline."""

    return run_baseline_eval(scenario_path, baseline_name="median")


def run_identity_eval(scenario_path: str | Path) -> EvalReport:
    """Backward-compatible wrapper for the current baseline evaluation path."""

    return run_baseline_eval(scenario_path)
