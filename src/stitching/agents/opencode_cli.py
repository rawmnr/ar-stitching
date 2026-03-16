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

        # Build command: opencode run -y -m provider/model "prompt"
        # -y is the 'YOLO' flag to auto-approve file modifications
        cmd = ["opencode", "run", "-y"]
        if self.provider and self.model:
            model_id = f"{self.provider}/{self.model}"
            cmd.extend(["-m", model_id])
        
        cmd.append(prompt)

        # Before calling, record the file state to detect direct changes
        target_path = Path(self.repo_root) / "src/stitching/editable/candidate_current.py"
        mtime_before = target_path.stat().st_mtime if target_path.exists() else 0

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            shell=True,
        )

        if result.returncode != 0:
            # If -y is not supported, it might fail, we catch and log
            if "unknown flag: -y" in result.stderr or "unknown flag: --yes" in result.stderr:
                # Fallback: try without -y if the version is older
                cmd.remove("-y")
                result = subprocess.run(
                    cmd,
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    shell=True,
                )
            else:
                raise RuntimeError(f"opencode failed: {result.stderr[:500]}")

        # Check if file was modified on disk directly
        mtime_after = target_path.stat().st_mtime if target_path.exists() else 0
        changes_pre_applied = (mtime_after > mtime_before)

        # Extract code block if present in stdout (fallback)
        full_source = self._extract_python_code(result.stdout)
        
        # If no code block found, but file modified, we are happy
        if not full_source and changes_pre_applied:
            full_source = target_path.read_text(encoding="utf-8")

        hypothesis = "Agent-proposed optimization (OpenCode)"
        # ... logic to find hypothesis ...

    def _build_prompt(self, ctx: ExperimentContext) -> str:
        lines = [
            "ACT AS AN AUTONOMOUS OPTIMIZATION AGENT.",
            "DIRECTLY EDIT the file src/stitching/editable/candidate_current.py to improve RMS.",
            "USE YOUR TOOLS to write the updated code to disk.",
            "DO NOT JUST EXPLAIN, APPLY THE CHANGES.",
            "",
            f"Current RMS: {ctx.current_metrics.get('aggregate_rms', 'N/A')}",
            f"Best RMS: {ctx.best_metrics.get('aggregate_rms', 'N/A')}",
            "",
            "Editable: " + ", ".join(ctx.editable_paths),
            "",
            "Candidate source:",
            ctx.candidate_source[:6000],
        ]
        # ... rest of prompt ...