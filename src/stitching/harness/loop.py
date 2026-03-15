"""Main autoresearch optimization loop."""

from __future__ import annotations

import logging
import time
import traceback
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
from stitching.harness.gitops import GitOps
from stitching.harness.ledger import Ledger
from stitching.harness.protocols import (
    ExperimentContext,
    RunManifest,
    RunResult,
    RunVerdict,
)

logger = logging.getLogger(__name__)


class AutoresearchLoop:
    """Orchestrates the full autoresearch optimization cycle."""

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

        self._best_metrics: dict[str, float] = self.ledger.load_best_metrics() or {}
        self._previous_diff: str | None = None
        self._previous_summary: str | None = None

    def run(self) -> None:
        start_iteration = self.ledger.iteration_count(self.experiment_id)
        logger.info(
            "Starting autoresearch loop: experiment=%s, backend=%s, "
            "scenarios=%d, max_iterations=%d, starting_at=%d",
            self.experiment_id, self.backend.name,
            len(self.scenario_paths), self.max_iterations, start_iteration,
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
                    logger.info(
                        "✓ ACCEPTED — new best RMS: %.6f",
                        result.metrics["aggregate_rms"],
                    )
                else:
                    self._previous_summary = (
                        f"REJECTED ({result.verdict.value}): {result.hypothesis}"
                    )
                    logger.info("✗ REJECTED — reason: %s", result.verdict.value)

            except KeyboardInterrupt:
                logger.info("Interrupted by user at iteration %d.", iteration)
                break
            except Exception as exc:
                logger.error("Unexpected error at iteration %d: %s", iteration, exc)
                logger.debug(traceback.format_exc())

    def _run_iteration(self, iteration: int) -> RunResult:
        budget_tracker = BudgetTracker(self.budget)
        budget_tracker.start()
        base_commit = self.git.current_commit()
        candidate_path = self.repo_root / self.candidate_rel_path

        # ──────────────────────────────────────────────────────
        # FIX 1: backup BEFORE calling the agent
        # ──────────────────────────────────────────────────────
        backup_source = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists()
            else ""
        )

        # 1. Evaluate current baseline
        current_metrics = self._evaluate_current(candidate_path)

        # 2. Build context
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

        # ──────────────────────────────────────────────────────
        # FIX 2: apply the patch to disk
        # ──────────────────────────────────────────────────────
        try:
            self._apply_patch(candidate_path, backup_source, proposal)
        except _PatchEmpty:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_EMPTY,
                current_metrics, "Agent produced no code changes.",
                budget_tracker.elapsed, hypothesis=proposal.hypothesis,
            )
        except _PatchSyntaxError as exc:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_SYNTAX,
                current_metrics, str(exc), budget_tracker.elapsed,
                hypothesis=proposal.hypothesis,
                diff=proposal.diff,
            )

        # 4. Evaluate the patched candidate
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
                hypothesis=proposal.hypothesis, diff=proposal.diff,
            )
        except GuardrailViolation as exc:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_GUARDRAIL,
                current_metrics, str(exc), budget_tracker.elapsed,
                hypothesis=proposal.hypothesis, diff=proposal.diff,
            )
        except Exception as exc:
            self._restore_candidate(candidate_path, backup_source)
            return self._build_failure_result(
                manifest, RunVerdict.REJECTED_CRASH,
                current_metrics, str(exc), budget_tracker.elapsed,
                hypothesis=proposal.hypothesis, diff=proposal.diff,
            )

        # 5. Accept / reject
        diff_patch = self.git.diff_against(base_commit, [self.candidate_rel_path])
        improved = self._is_improvement(current_metrics, new_metrics)

        if improved:
            commit_msg = (
                f"autoresearch: iter={iteration} "
                f"rms={new_metrics['aggregate_rms']:.8f}\n\n"
                f"Hypothesis: {proposal.hypothesis}\n"
                f"Agent: {self.backend.name}"
            )
            self.git.stage_and_commit([self.candidate_rel_path], commit_msg)
            verdict = RunVerdict.ACCEPTED
            final_metrics = new_metrics
        else:
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
                f"{r.scenario_id}: "
                f"rms={r.signal_metrics['rms_on_valid_intersection']:.6f}"
                for r in (reports if improved else ())
            ),
        )

    # ──────────────────────────────────────────────────────────
    # New: patch application with syntax validation
    # ──────────────────────────────────────────────────────────

    def _apply_patch(
        self,
        candidate_path: Path,
        backup_source: str,
        proposal: "PatchProposal",
    ) -> None:
        """Write the proposed code to disk and validate it compiles."""
        from stitching.harness.protocols import PatchProposal

        disk_content = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists()
            else ""
        )

        if proposal.changes_pre_applied:
            # Codex already wrote to disk — check the file actually changed
            new_source = disk_content
            if new_source.strip() == backup_source.strip():
                raise _PatchEmpty("Backend claimed pre-applied but file unchanged.")
        elif proposal.full_source:
            # API backend provided the full code — write it
            new_source = _strip_bom(proposal.full_source)
            candidate_path.write_text(new_source, encoding="utf-8")
        else:
            # No full_source and no pre-applied changes — nothing to apply
            raise _PatchEmpty("PatchProposal has no full_source and changes_pre_applied=False.")

        # Validate Python syntax before evaluation
        new_source = candidate_path.read_text(encoding="utf-8")
        try:
            compile(new_source, str(candidate_path), "exec")
        except SyntaxError as exc:
            raise _PatchSyntaxError(
                f"Proposed code has syntax error: {exc.msg} "
                f"(line {exc.lineno})"
            ) from exc

        # Verify the file defines CandidateStitcher
        if "class CandidateStitcher" not in new_source:
            raise _PatchSyntaxError(
                "Proposed code does not define class CandidateStitcher."
            )

        logger.info(
            "Patch applied: %d → %d bytes",
            len(backup_source),
            len(new_source),
        )

    # ──────────────────────────────────────────────────────────

    def _evaluate_current(self, candidate_path: Path) -> dict[str, float]:
        try:
            candidate = load_candidate_module(candidate_path)
            metrics, _ = evaluate_candidate_on_suite(
                candidate, self.scenario_paths, self.budget.eval_time_sec,
            )
            return metrics
        except Exception as exc:
            logger.warning("Baseline evaluation failed: %s", exc)
            return {"aggregate_rms": float("inf"), "num_scenarios": 0}

    def _is_improvement(
        self, current: dict[str, float], proposed: dict[str, float],
    ) -> bool:
        current_rms = current.get("aggregate_rms", float("inf"))
        proposed_rms = proposed.get("aggregate_rms", float("inf"))
        all_accepted = (
            proposed.get("num_accepted", 0) == proposed.get("num_scenarios", 0)
        )
        return proposed_rms < current_rms and all_accepted

    def _restore_candidate(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def _discover_scenarios(self) -> list[Path]:
        scenarios_dir = self.repo_root / "scenarios"
        return sorted(scenarios_dir.glob("s[0-9]*.yaml"))

    def _build_failure_result(
        self, manifest, verdict, metrics, error, elapsed,
        hypothesis="", diff="",
    ) -> RunResult:
        return RunResult(
            manifest=manifest, verdict=verdict, metrics=metrics,
            eval_reports=(), hypothesis=hypothesis or "N/A",
            diff_patch=diff, elapsed_sec=elapsed,
            notes=(f"error: {error[:500]}",),
        )


# ── Internal exceptions (not exported) ──

class _PatchEmpty(Exception):
    pass

class _PatchSyntaxError(Exception):
    pass

def _strip_bom(source: str) -> str:
    """Remove UTF-8 BOM and leading whitespace artifacts."""
    return source.lstrip("\ufeff").lstrip("\ufffe")
