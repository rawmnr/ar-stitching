"""Small harness for running trusted identity evaluation on versioned scenarios."""

from __future__ import annotations

from pathlib import Path

from stitching.contracts import EvalReport, ScenarioConfig
from stitching.trusted.eval.metrics import build_eval_report
from stitching.trusted.simulator.identity import simulate_identity_observations


def run_identity_eval(scenario_path: str | Path) -> EvalReport:
    """Load a scenario, simulate identity observations, and evaluate them."""

    config = ScenarioConfig.from_yaml(scenario_path)
    truth, observations = simulate_identity_observations(config)
    return build_eval_report(config, truth, observations[0], runtime_sec=0.0)
