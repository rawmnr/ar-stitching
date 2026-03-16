"""OpenCode CLI agent backend for local-model-driven experiments."""

from __future__ import annotations

import subprocess
from pathlib import Path

from stitching.harness.protocols import AgentBackend, ExperimentContext, PatchProposal


class OpenCodeCliBackend(AgentBackend):
    """Agent backend wrapping sst1/opencode CLI."""

    def __init__(
        self,
        repo_root: Path,
        provider: str = "llama-swap",
        model: str = "gpt-oss-prod",
        timeout_sec: float = 120.0,
    ) -> None:
        self.repo_root = repo_root
        self.provider = provider
        self.model = model
        self.timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return f"opencode/{self.provider}/{self.model}"

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        """Invoke opencode with a structured prompt."""
        prompt = self._build_prompt(context)

        # Build command with explicit model if provider/model are set
        cmd = ["opencode", "--non-interactive"]
        if self.provider and self.model:
            model_id = f"{self.provider}/{self.model}"
            cmd.extend(["-m", model_id])

        # OpenCode uses stdin for prompts in non-interactive mode
        result = subprocess.run(
            cmd,
            input=prompt,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )

        if result.returncode != 0:
            raise RuntimeError(f"opencode failed: {result.stderr[:500]}")

        hypothesis = "Agent-proposed optimization (OpenCode)"
        for line in result.stdout.splitlines():
            if "hypothesis" in line.lower():
                hypothesis = line.strip()
                break

        return PatchProposal(
            hypothesis=hypothesis,
            diff=result.stdout[:5000],
            target_files=context.editable_paths,
            reasoning=result.stdout[:2000],
        )

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        prompt = f"/inspect-failure\nError: {error}\nRMS: {context.current_metrics.get('aggregate_rms')}"
        
        cmd = ["opencode", "--non-interactive"]
        if self.provider and self.model:
            model_id = f"{self.provider}/{self.model}"
            cmd.extend(["-m", model_id])

        result = subprocess.run(
            cmd,
            input=prompt,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        return result.stdout[:2000]

    def _build_prompt(self, ctx: ExperimentContext) -> str:
        lines = [
            "/improve-candidate",
            f"Current RMS: {ctx.current_metrics.get('aggregate_rms', 'N/A')}",
            f"Best RMS: {ctx.best_metrics.get('aggregate_rms', 'N/A')}",
            f"Budget: {ctx.time_budget_sec}s",
            f"Iteration: {ctx.iteration}",
            "",
            "Editable: " + ", ".join(ctx.editable_paths),
            "",
            "Candidate source:",
            ctx.candidate_source[:6000],
        ]
        if ctx.previous_summary:
            lines.append(f"\nPrevious: {ctx.previous_summary}")
        lines.append(f"\nDomain: {ctx.domain_notes[:1500]}")
        return "\n".join(lines)