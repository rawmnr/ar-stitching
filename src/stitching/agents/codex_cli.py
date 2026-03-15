"""Codex CLI agent backend using `codex exec` for non-interactive runs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from stitching.harness.protocols import AgentBackend, ExperimentContext, PatchProposal


class CodexCliBackend(AgentBackend):
    """Agent backend wrapping OpenAI Codex CLI (`codex exec`)."""

    def __init__(
        self,
        repo_root: Path,
        model: str = "o4-mini",
        approval_mode: str = "full-auto",
        timeout_sec: float = 120.0,
    ) -> None:
        self.repo_root = repo_root
        self.model = model
        self.approval_mode = approval_mode
        self.timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return f"codex-cli/{self.model}"

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        """Call `codex exec` with a structured prompt and parse its output."""
        prompt = self._build_prompt(context)

        result = subprocess.run(
            [
                "codex",
                "exec",
                f"--model={self.model}",
                f"--approval-mode={self.approval_mode}",
                prompt,
            ],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )

        if result.returncode != 0:
            raise RuntimeError(f"codex exec failed (rc={result.returncode}): {result.stderr[:500]}")

        return self._parse_output(result.stdout, context)

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        """Ask Codex to analyze why a previous iteration failed."""
        prompt = (
            f"Analyze this autoresearch failure for optical stitching:\n\n"
            f"Error: {error}\n\n"
            f"Previous hypothesis: {context.extra.get('last_hypothesis', 'N/A')}\n"
            f"Current RMS: {context.current_metrics.get('aggregate_rms', 'N/A')}\n"
            f"Suggest a different approach."
        )
        result = subprocess.run(
            ["codex", "exec", f"--model={self.model}", f"--approval-mode={self.approval_mode}", prompt],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        return result.stdout[:2000] if result.returncode == 0 else f"Analysis failed: {result.stderr[:500]}"

    def _build_prompt(self, ctx: ExperimentContext) -> str:
        """Construct a structured prompt for Codex."""
        lines = [
            "You are optimizing an optical sub-aperture stitching algorithm.",
            "",
            "## Editable files (you may ONLY modify these):",
            *[f"  - {p}" for p in ctx.editable_paths],
            "",
            "## Forbidden paths (NEVER touch these):",
            *[f"  - {p}" for p in ctx.forbidden_paths],
            "",
            f"## Current aggregate RMS: {ctx.current_metrics.get('aggregate_rms', 'N/A')}",
            f"## Best aggregate RMS so far: {ctx.best_metrics.get('aggregate_rms', 'N/A')}",
            f"## Time budget: {ctx.time_budget_sec:.0f}s per scenario",
            "",
        ]
        if ctx.previous_diff:
            lines.extend([
                "## Previous diff (last iteration):",
                "```diff",
                ctx.previous_diff[:3000],
                "```",
                "",
            ])
        if ctx.previous_summary:
            lines.extend([
                f"## Previous attempt summary: {ctx.previous_summary}",
                "",
            ])
        lines.extend([
            "## Domain notes:",
            ctx.domain_notes[:2000],
            "",
            "## Current candidate source:",
            "```python",
            ctx.candidate_source[:6000],
            "```",
            "",
            "## Instructions:",
            "1. State your mathematical HYPOTHESIS explicitly.",
            "2. Modify ONLY the candidate file to implement it.",
            "3. Use vectorized NumPy/SciPy, no Python loops on pixels.",
            "4. The reconstruction must call `reconstruct(observations, config)` and return a ReconstructionSurface.",
            "5. Do NOT touch the scorer, seeds, or trusted stack.",
        ])
        return "\n".join(lines)

    def _parse_output(self, stdout: str, ctx: ExperimentContext) -> PatchProposal:
        """Extract a PatchProposal from Codex output."""
        # Codex exec applies changes directly in the worktree.
        # We read the modified file back.
        hypothesis = "Agent-proposed optimization (extracted from Codex output)"
        # Try to extract hypothesis from stdout
        for line in stdout.splitlines():
            if line.strip().lower().startswith("hypothesis:"):
                hypothesis = line.split(":", 1)[1].strip()
                break

        return PatchProposal(
            hypothesis=hypothesis,
            diff=stdout[:5000],
            target_files=ctx.editable_paths,
            reasoning=stdout[:2000],
        )
