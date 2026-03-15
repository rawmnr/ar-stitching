#!/usr/bin/env python3
"""Entry point: run the autoresearch loop."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from stitching.harness.budgets import IterationBudget
from stitching.harness.loop import AutoresearchLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoresearch optimization loop.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-id", default="exp_default")
    parser.add_argument("--backend", default="codex", choices=["codex", "opencode"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--budget-profile", default="default", choices=["fast", "default", "overnight"])
    parser.add_argument("--scenarios", nargs="*", help="Specific scenario YAML files")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load budget
    budgets_path = args.repo_root / "configs" / "budgets.yaml"
    if budgets_path.exists():
        budgets_cfg = yaml.safe_load(budgets_path.read_text())
        budget_dict = budgets_cfg.get(args.budget_profile, {})
        budget = IterationBudget(**budget_dict)
    else:
        budget = IterationBudget()

    # Resolve scenarios
    scenario_paths = None
    if args.scenarios:
        scenario_paths = [Path(s) for s in args.scenarios]

    # Backend kwargs
    backend_kwargs = {}
    if args.model:
        backend_kwargs["model"] = args.model

    loop = AutoresearchLoop(
        repo_root=args.repo_root,
        experiment_id=args.experiment_id,
        backend_name=args.backend,
        backend_kwargs=backend_kwargs,
        scenario_paths=scenario_paths,
        budget=budget,
        max_iterations=args.max_iterations,
    )
    loop.run()


if __name__ == "__main__":
    main()
