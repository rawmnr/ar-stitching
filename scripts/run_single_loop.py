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
    parser.add_argument("--backend", default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--title", default="autoresearch", help="Title for opencode session")
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

    # Load backend config
    agents_path = args.repo_root / "configs" / "agents.yaml"
    backend_name = args.backend
    backend_kwargs = {}
    
    if agents_path.exists():
        agents_cfg = yaml.safe_load(agents_path.read_text())
        backends = agents_cfg.get("backends", {})
        
        # If the user-specified backend is a key in agents.yaml (e.g. opencode_gpt_oss)
        if backend_name in backends:
            cfg = backends[backend_name]
            # Use the actual backend type (e.g. 'opencode') from the key or a 'type' field
            # In our case, the key starts with the backend type
            if backend_name.startswith("opencode"):
                backend_name = "opencode"
            elif backend_name.startswith("codex"):
                backend_name = "codex"
            
            backend_kwargs.update(cfg)
        elif backend_name == "opencode" and "opencode_gpt_oss" in backends:
             # Default fallback if just 'opencode' is requested
             backend_kwargs.update(backends["opencode_gpt_oss"])

    if args.model:
        backend_kwargs["model"] = args.model

    # Add title for opencode sessions
    if args.title:
        backend_kwargs["title"] = args.title

    loop = AutoresearchLoop(
        repo_root=args.repo_root,
        experiment_id=args.experiment_id,
        backend_name=backend_name,
        backend_kwargs=backend_kwargs,
        scenario_paths=scenario_paths,
        budget=budget,
        max_iterations=args.max_iterations,
    )
    loop.run()


if __name__ == "__main__":
    main()
