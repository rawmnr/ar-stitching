"""Codex CLI agent backend using `codex exec`."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

from stitching.harness.protocols import AgentBackend, ExperimentContext, PatchProposal

logger = logging.getLogger(__name__)

# Detect platform for executable name
_CODEX_CMD = "codex.cmd" if platform.system() == "Windows" else "codex"


class CodexCliBackend:
    """Agent backend wrapping OpenAI Codex CLI (`codex exec`)."""

    def __init__(
        self,
        repo_root: Path,
        model: str = "o4-mini",
        approval_mode: str = "full-auto",
        timeout_sec: float = 300.0,
        **kwargs: Any,
    ) -> None:
        self.repo_root = repo_root
        self.model = model
        self.approval_mode = approval_mode
        self.timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return f"codex-cli/{self.model}"

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        """Write a task file, call codex exec, detect file changes."""
        candidate_path = self.repo_root / context.editable_paths[0]

        # Save pre-state to detect changes
        pre_content = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists()
            else ""
        )

        # Write detailed task to a file Codex can read
        task_path = self.repo_root / ".codex_task.md"
        task_text = self._build_task(context)
        task_path.write_text(task_text, encoding="utf-8")

        # Build a SHORT command-line prompt that references the task file
        rms = context.current_metrics.get("aggregate_rms", "unknown")
        cli_prompt = (
            f"Read .codex_task.md for full instructions. "
            f"Current RMS={rms}. "
            f"Edit {context.editable_paths[0]} to improve the stitching algorithm. "
            f"Do not ask questions. Just edit the file."
        )

        logger.info(
            "Calling %s exec (model=%s, prompt=%d chars, task=%d chars)",
            _CODEX_CMD, self.model, len(cli_prompt), len(task_text),
        )

        try:
            result = subprocess.run(
                [
                    _CODEX_CMD, "exec",
                    "--full-auto",
                    "--model", self.model,
                    cli_prompt,
                ],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_sec,
            )
        finally:
            # Clean up task file
            task_path.unlink(missing_ok=True)

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0:
            logger.warning(
                "codex exec returned %d: %s", result.returncode, stderr[:300],
            )

        # Detect if Codex actually changed the file
        post_content = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists()
            else ""
        )

        file_changed = post_content.strip() != pre_content.strip()

        if file_changed:
            # Validate the new content is at least valid Python
            if "class CandidateStitcher" not in post_content:
                logger.warning("Codex wrote file without CandidateStitcher. Reverting.")
                candidate_path.write_text(pre_content, encoding="utf-8")
                raise RuntimeError(
                    "Codex wrote non-conforming code (no CandidateStitcher)."
                )

            hypothesis = self._extract_hypothesis(stdout)
            logger.info("Codex modified file (%d → %d bytes)", len(pre_content), len(post_content))

            return PatchProposal(
                hypothesis=hypothesis,
                diff=stdout[:3000],
                target_files=context.editable_paths,
                reasoning=stdout[:2000],
                full_source=post_content,
                changes_pre_applied=True,
            )
        else:
            # Codex did NOT edit the file — extract code from stdout
            logger.warning("Codex did not modify the file. Trying to extract code from stdout.")
            full_source = self._extract_code_from_output(stdout)
            if full_source and "class CandidateStitcher" in full_source:
                return PatchProposal(
                    hypothesis=self._extract_hypothesis(stdout),
                    diff=stdout[:3000],
                    target_files=context.editable_paths,
                    reasoning=stdout[:2000],
                    full_source=full_source,
                    changes_pre_applied=False,
                )
            raise RuntimeError(
                f"Codex produced no usable code. "
                f"stdout={stdout[:300]}, stderr={stderr[:300]}"
            )

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        prompt = (
            f"Analyze failure: {error[:200]}. "
            f"RMS={context.current_metrics.get('aggregate_rms')}"
        )
        try:
            result = subprocess.run(
                [_CODEX_CMD, "exec", f"--model={self.model}", prompt],
                cwd=str(self.repo_root),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=60.0,
            )
            return result.stdout[:2000]
        except Exception as exc:
            return f"Analysis failed: {exc}"

    def _build_task(self, ctx: ExperimentContext) -> str:
        """Write a full task file for Codex to read."""
        lines = [
            "# Autoresearch Task",
            "",
            "## Objective",
            f"Improve `{ctx.editable_paths[0]}` to reduce aggregate RMS.",
            "",
            "## Current Metrics",
            f"- Aggregate RMS: {ctx.current_metrics.get('aggregate_rms', 'N/A')}",
            f"- Best RMS: {ctx.best_metrics.get('aggregate_rms', 'N/A')}",
            f"- Scenarios: {', '.join(ctx.scenario_ids)}",
            "",
            "## Domain Notes",
            ctx.domain_notes[:3000],
            "",
        ]
        if ctx.previous_summary:
            lines.extend([
                "## Previous Attempt",
                ctx.previous_summary[:1000],
                "",
            ])
        lines.extend([
            "## Rules",
            "- ONLY modify the candidate file.",
            "- Keep class CandidateStitcher with method reconstruct().",
            "- Return a ReconstructionSurface with correct observed_support_mask.",
            "- Use NumPy/SciPy vectorized operations.",
            "- State your hypothesis as a code comment at the top.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _extract_hypothesis(stdout: str) -> str:
        for line in stdout.splitlines():
            low = line.lower().strip()
            if low.startswith("hypothesis:") or low.startswith("# hypothesis"):
                return line.split(":", 1)[-1].strip()
        return "Agent-proposed optimization (Codex)"

    @staticmethod
    def _extract_code_from_output(stdout: str) -> str | None:
        """Try to extract a Python code block from Codex stdout."""
        import re
        blocks = re.findall(r"```python\s*\n(.*?)```", stdout, re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()
        # Fallback: find lines that look like Python
        lines = stdout.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip().startswith(("from ", "import ", '"""', "class ")):
                start = i
                break
        if start is not None:
            return "\n".join(lines[start:])
        return None
