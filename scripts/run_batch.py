#!/usr/bin/env python3
"""Run a batch of experiments from the queue."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from stitching.harness.budgets import IterationBudget
from stitching.harness.loop import AutoresearchLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queued experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--queue-dir", type=Path, default=None)
    parser.add_argument("--backend", default="codex")
    parser.add_argument("--max-iterations-per-exp", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    queue_dir = args.queue_dir or (args.repo_root / "experiments" / "queue")
    if not queue_dir.exists():
        logger.error("Queue directory not found: %s", queue_dir)
        return

    for exp_file in sorted(queue_dir.glob("*.yaml")):
        logger.info("Processing experiment: %s", exp_file.name)
        exp_cfg = yaml.safe_load(exp_file.read_text())
        experiment_id = exp_cfg.get("experiment_id", exp_file.stem)
        scenarios = exp_cfg.get("scenarios", None)

        scenario_paths = None
        if scenarios:
            scenario_paths = [args.repo_root / "scenarios" / f"{s}.yaml" for s in scenarios]

        budget_dict = exp_cfg.get("budget", {})
        budget = IterationBudget(**budget_dict) if budget_dict else IterationBudget()

        loop = AutoresearchLoop(
            repo_root=args.repo_root,
            experiment_id=experiment_id,
            backend_name=args.backend,
            scenario_paths=scenario_paths,
            budget=budget,
            max_iterations=args.max_iterations_per_exp,
        )
        loop.run()

        # Move processed experiment out of queue
        done_dir = queue_dir.parent / "done"
        done_dir.mkdir(exist_ok=True)
        exp_file.rename(done_dir / exp_file.name)
        logger.info("Completed: %s", experiment_id)


if __name__ == "__main__":
    main()
