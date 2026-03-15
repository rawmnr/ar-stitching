"""Main autoresearch optimization loop."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stitching.agents.broker import create_backend
from stitching.agents.prompt_builder import build_experiment_context
from stitching.harness.budgets import BudgetExceededError, BudgetTracker, IterationBudget
from stitching.harness.evaluator import (
    GuardrailViolation,
    evaluate_candidate_on_suite,
    load_candidate_module,
)
from stitching.harness.gitops import GitOps, GitOpsError
from stitching.harness.ledger import Ledger
from stitching.harness.protocols import (
    ExperimentContext,
    RunManifest,
    RunResult,
    RunVerdict,
)

logger = logging.getLogger(__name__)


class AutoresearchLoop:
    """Orchestrates the full autoresearch optimization cycle.

    Sequence per iteration:
    1. Load current state and build experiment context
    2. Create isolated git worktree
    3. Invoke agent backend to propose a patch
    4. Apply patch to candidate in worktree
    5. Run trusted evaluation on scenario suite
    6. If improvement + guardrails OK → commit, archive accepted
    7. Otherwise → reject, archive, revert
    """

    def __init__(
        self,
        repo_root: Path,
        experiment_id: str,
        backend_name: str = "codex",
        backend_kwargs: dict[str, Any] | None = None,
        scenario_paths: list[Path] | None = None,
        budget: IterationBudget | None = None,
        max_iterations: int = 100,
        candidate_rel_path: str = "src/stitching/editable/candidate_current.py",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.experiment_id = experiment_id
        self.git = GitOps(self.repo_root)
        self.ledger = Ledger(self.repo_root / "experiments")
        self.backend = create_backend(
            backend_name,
            repo_root=self.repo_root,
            **(backend_kwargs or {}),
        )
        self.scenario_paths = scenario_paths or self._discover_scenarios()
        self.budget = budget or IterationBudget()
        self.max_iterations = max_iterations
        self.candidate_rel_path = candidate_rel_path

        # State
        self._best_metrics: dict[str, float] = self.ledger.load_best_metrics() or {}
        self._previous_diff: str | None = None
        self._previous_summary: str | None = None

    # ---- Public API ----

    def run(self) -> None:
        """Execute the full optimization loop."""
        start_iteration = self.ledger.iteration_count(self.experiment_id)
        logger.info(
            "Starting autoresearch loop: experiment=%s, backend=%s, "
            "scenarios=%d, max_iterations=%d, starting_at=%d",
            self.experiment_id,
            self.backend.name,
            len(self.scenario_paths),
            self.max_iterations,
            start_iteration,
        )

        for iteration in range(start_iteration, start_iteration + self.max_iterations):
            logger.info("=" * 60)
            logger.info("Iteration %d / %d", iteration, start_iteration + self.max_iterations - 1)
            try:
                result = self._run_iteration(iteration)
                self.ledger.record(result)
                logger.info(
                    "Verdict: %s | RMS: %.6f | Elapsed: %.1fs",
                    result.verdict.value,
                    result.metrics.get("aggregate_rms", float("nan")),
                    result.elapsed_sec,
                )
                if result.verdict == RunVerdict.ACCEPTED:
                    self._best_metrics = result.metrics.copy()
                    self._previous_diff = result.diff_patch
                    self._previous_summary = result.hypothesis
                    logger.info("✓ ACCEPTED — new best RMS: %.6f", result.metrics["aggregate_rms"])
                else:
                    self._previous_summary = f"REJECTED ({result.verdict.value}): {result.hypothesis}"
                    logger.info("✗ REJECTED — reason: %s", result.verdict.value)

            except KeyboardInterrupt:
                logger.info("Interrupted by user at iteration %d.", iteration)
                break
            except Exception as exc:
                logger.error("Unexpected error at iteration %d: %s", iteration, exc)
                logger.debug(traceback.format_exc())

    # ---- Iteration logic ----

    def _run_iteration(self, iteration: int) -> RunResult:
        """Execute one complete iteration: propose → evaluate → decide."""
        budget_tracker = BudgetTracker(self.budget)
        budget_tracker.start()
        base_commit = self.git.current_commit()
        candidate_path = self.repo_root / self.candidate_rel_path

        # 1. Evaluate current baseline
        current_metrics = self._evaluate_current(candidate_path)

        # 2. Build context for the agent
        context = build_experiment_context(
            experiment_id=self.experiment_id,
            iteration=iteration,
            current_metrics=current_metrics,
            best_metrics=self._best_metrics or current_metrics,
            candidate_path=candidate_path,
            previous_diff=self._previous_diff,
            previous_summary=self._previous_summary,
            scenario_ids=tuple(p.stem for p in self.scenario_paths),
            time_budget_sec=self.budget.eval_time_sec,
        )

        manifest = RunManifest(
            experiment_id=self.experiment_id,
            iteration=iteration,
            agent_backend=self.backend.name,
            prompt_hash=GitOps.prompt_hash(context.candidate_source),
            source_commit=base_commit,
            scenario_ids=context.scenario_ids,
            time_budget_sec=self.budget.total_time_sec,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            seed=iteration,
        )

        # 3. Ask agent for a patch
        try:
            proposal = self.backend.propose_patch(context)
        except Exception as exc:
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_CRASH,
                current_metrics, str(exc), budget_tracker.elapsed,
            )

        # 4. Save the patch (agent modifies worktree directly for Codex;
        #    for other backends, we apply the diff)
        backup_source = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else ""

        # 5. Evaluate the patched candidate
        try:
            budget_tracker.check()
            new_metrics, reports = evaluate_candidate_on_suite(
                load_candidate_module(candidate_path),
                self.scenario_paths,
                eval_budget_sec=self.budget.eval_time_sec,
            )
        except BudgetExceededError:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_TIMEOUT,
                current_metrics, "Evaluation timeout", budget_tracker.elapsed,
                hypothesis=proposal.hypothesis,
                diff=proposal.diff,
            )
        except GuardrailViolation as exc:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_GUARDRAIL,
                current_metrics, str(exc), budget_tracker.elapsed,
                hypothesis=proposal.hypothesis,
                diff=proposal.diff,
            )
        except Exception as exc:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_CRASH,
                current_metrics, str(exc), budget_tracker.elapsed,
                hypothesis=proposal.hypothesis,
                diff=proposal.diff,
            )

        # 6. Accept / reject decision
        diff_patch = self.git.diff_against(base_commit, [self.candidate_rel_path])
        improved = self._is_improvement(current_metrics, new_metrics)

        if improved:
            # Commit the improvement
            commit_msg = (
                f"autoresearch: iter={iteration} rms={new_metrics['aggregate_rms']:.8f}\n\n"
                f"Hypothesis: {proposal.hypothesis}\n"
                f"Agent: {self.backend.name}"
            )
            self.git.stage_and_commit([self.candidate_rel_path], commit_msg)
            verdict = RunVerdict.ACCEPTED
            final_metrics = new_metrics
        else:
            # Revert
            self._restore_candidate(candidate_path, backup_source)
            verdict = RunVerdict.REJECTED_REGRESSION
            final_metrics = current_metrics

        return RunResult(
            manifest=manifest,
            verdict=verdict,
            metrics=final_metrics,
            eval_reports=reports if improved else (),
            hypothesis=proposal.hypothesis,
            diff_patch=diff_patch,
            elapsed_sec=budget_tracker.elapsed,
            notes=tuple(
                f"{r.scenario_id}: rms={r.signal_metrics['rms_on_valid_intersection']:.6f}"
                for r in (reports if improved else ())
            ),
        )

    # ---- Helpers ----

    def _evaluate_current(self, candidate_path: Path) -> dict[str, float]:
        """Evaluate the current candidate to establish baseline metrics."""
        try:
            candidate = load_candidate_module(candidate_path)
            metrics, _ = evaluate_candidate_on_suite(
                candidate, self.scenario_paths, self.budget.eval_time_sec,
            )
            return metrics
        except Exception:
            return {"aggregate_rms": float("inf"), "num_scenarios": 0}

    def _is_improvement(
        self,
        current: dict[str, float],
        proposed: dict[str, float],
    ) -> bool:
        """Strict improvement: lower aggregate RMS and all scenarios accepted."""
        current_rms = current.get("aggregate_rms", float("inf"))
        proposed_rms = proposed.get("aggregate_rms", float("inf"))
        all_accepted = proposed.get("num_accepted", 0) == proposed.get("num_scenarios", 0)
        return proposed_rms < current_rms and all_accepted

    def _restore_candidate(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def _discover_scenarios(self) -> list[Path]:
        scenarios_dir = self.repo_root / "scenarios"
        return sorted(scenarios_dir.glob("s[0-9]*.yaml"))

    def _build_failure_result(
        self,
        manifest: RunManifest,
        verdict: RunVerdict,
        metrics: dict[str, float],
        error: str,
        elapsed: float,
        hypothesis: str = "",
        diff: str = "",
    ) -> RunResult:
        return RunResult(
            manifest=manifest,
            verdict=verdict,
            metrics=metrics,
            eval_reports=(),
            hypothesis=hypothesis or "N/A",
            diff_patch=diff,
            elapsed_sec=elapsed,
            notes=(f"error: {error[:500]}",),
        )
