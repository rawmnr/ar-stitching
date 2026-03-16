"""OpenCode CLI agent backend for local-model-driven experiments.

Robustified version with directive prompts, retry logic, and syntax validation.
"""

from __future__ import annotations

import logging
import subprocess
import re
import os
import shutil
import time
from pathlib import Path

from stitching.harness.protocols import AgentBackend, ExperimentContext, PatchProposal


class OpenCodeCliBackend(AgentBackend):
    """Agent backend wrapping sst1/opencode CLI."""
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 2.0
    
    # Minimum expected runtime (seconds) - if faster, likely no action taken
    MIN_EXPECTED_RUNTIME_SEC = 5.0
    AGENTIC_WAIT_SEC = 30.0  # Give agent time to actually do work
    
    # Validation
    REQUIRED_CLASS = "class CandidateStitcher"

    def __init__(
        self,
        repo_root: Path,
        provider: str = "llama-swap",
        model: str = "gpt-oss-prod",
        timeout_sec: float = 600.0,
        agentic_timeout_sec: float = 120.0,  # Per-attempt timeout
        **kwargs,
    ) -> None:
        self.repo_root = repo_root
        self.provider = provider
        self.model = model
        self.timeout_sec = timeout_sec
        self.logger = logging.getLogger(__name__)
        self.agentic_timeout_sec = agentic_timeout_sec
        self._task_file_path = Path(self.repo_root) / ".opencode_task.md"

    @property
    def name(self) -> str:
        return f"opencode/{self.provider}/{self.model}"

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        """Invoke opencode with a directive prompt and retry logic."""
        target_rel_path = context.editable_paths[0] if context.editable_paths else "src/stitching/editable/candidate_current.py"
        target_path = Path(self.repo_root) / target_rel_path
        
        # Backup original content
        original_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        original_mtime = target_path.stat().st_mtime if target_path.exists() else 0
        
        last_error: str | None = None
        
        for attempt in range(self.MAX_RETRIES):
            self.logger.info(
                "OpenCode attempt %d/%d for iteration %d",
                attempt + 1, self.MAX_RETRIES, context.iteration,
            )
            
            # Write detailed task to a file (robust to long prompts)
            task_content = self._build_task_file(
                context, 
                attempt=attempt,
                previous_error=last_error,
            )
            self._task_file_path.write_text(task_content, encoding="utf-8")
            
            # Build a SHORT CLI prompt that references the task file
            cli_prompt = self._build_cli_prompt(context, attempt)
            
            # Track execution time to detect non-agentic behavior
            start_time = time.time()
            
            try:
                stdout_text, stderr_text, return_code = self._execute_opencode(
                    cli_prompt, context, target_path
                )
                
                # Check if file was modified
                elapsed = time.time() - start_time
                
                # If completed too fast, agent likely didn't do anything
                if elapsed < self.MIN_EXPECTED_RUNTIME_SEC:
                    last_error = f"Execution too fast ({elapsed:.1f}s) - agent likely did not take action"
                    self.logger.warning(
                        "Attempt %d: %s. stdout[:200]=%s",
                        attempt + 1, last_error, stdout_text[:200],
                    )
                    time.sleep(self.RETRY_DELAY_SEC)
                    continue
                
                current_mtime = target_path.stat().st_mtime if target_path.exists() else 0
                file_modified = current_mtime > original_mtime
                
                if file_modified:
                    new_content = target_path.read_text(encoding="utf-8")
                    
                    # Validate the new content
                    validation_error = self._validate_code(new_content)
                    if validation_error:
                        last_error = validation_error
                        self.logger.warning(
                            "Attempt %d: Code validation failed: %s",
                            attempt + 1, validation_error,
                        )
                        # Restore original and retry
                        target_path.write_text(original_content, encoding="utf-8")
                        time.sleep(self.RETRY_DELAY_SEC)
                        continue
                    
                    # Check if it's actually different (not just whitespace)
                    if new_content.strip() == original_content.strip():
                        last_error = "No meaningful changes detected"
                        self.logger.warning("Attempt %d: %s", attempt + 1, last_error)
                        time.sleep(self.RETRY_DELAY_SEC)
                        continue
                    
                    # Success! Extract hypothesis and return
                    hypothesis = self._extract_hypothesis(stdout_text, new_content)
                    
                    return PatchProposal(
                        hypothesis=hypothesis,
                        diff=stdout_text[:5000],
                        target_files=context.editable_paths,
                        reasoning=stdout_text[:2000],
                        full_source=new_content,
                        changes_pre_applied=True,
                    )
                else:
                    # File not modified - try to extract code from stdout
                    extracted = self._extract_python_code(stdout_text)
                    if extracted and self.REQUIRED_CLASS in extracted:
                        validation_error = self._validate_code(extracted)
                        if not validation_error:
                            return PatchProposal(
                                hypothesis=self._extract_hypothesis(stdout_text, extracted),
                                diff=stdout_text[:5000],
                                target_files=context.editable_paths,
                                reasoning=stdout_text[:2000],
                                full_source=extracted,
                                changes_pre_applied=False,
                            )
                    
                    last_error = "Agent did not modify the target file"
                    self.logger.warning("Attempt %d: %s", attempt + 1, last_error)
                    time.sleep(self.RETRY_DELAY_SEC)
                    
            except subprocess.TimeoutExpired:
                last_error = f"Timeout after {self.timeout_sec}s"
                self.logger.warning("Attempt %d: %s", attempt + 1, last_error)
            except Exception as exc:
                last_error = str(exc)
                self.logger.warning("Attempt %d failed: %s", attempt + 1, exc)
        
        # Clean up task file
        if self._task_file_path.exists():
            self._task_file_path.unlink(missing_ok=True)
        
        # All retries exhausted
        raise RuntimeError(
            f"OpenCode failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _execute_opencode(
        self,
        prompt: str,
        context: ExperimentContext,
        target_path: Path,
    ) -> tuple[str, str, int]:
        """Execute the opencode CLI and return (stdout, stderr, return_code)."""
        
        # Build command with short prompt (task details in file)
        # Important: pass prompt DIRECTLY, not via stdin
        cmd = ["opencode", "run", prompt, "--print-logs"]
        
        if self.provider and self.model:
            cmd.extend(["-m", f"{self.provider}/{self.model}"])
        
        # Attach the task file and editable files
        cmd.extend(["-f", str(self._task_file_path.relative_to(self.repo_root))])
        for path in context.editable_paths:
            cmd.extend(["-f", str(path)])
        
        baseline_path = Path(self.repo_root) / "src/stitching/editable/baseline.py"
        if baseline_path.exists():
            cmd.extend(["-f", "src/stitching/editable/baseline.py"])
        
        # Environment
        env = os.environ.copy()
        env["OPENCODE_PERMISSION"] = '{"*": "allow"}'
        env["UV_NATIVE_TLS"] = "1"
        
        executable = shutil.which("opencode")
        if not executable:
            raise RuntimeError("opencode executable not found in PATH.")
        
        is_windows = os.name == "nt"
        
        result = subprocess.run(
            [executable] + cmd[1:],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.agentic_timeout_sec,
            shell=is_windows,
            env=env,
        )
        
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        # Debug logging
        try:
            debug_dir = Path(self.repo_root) / "experiments" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_log = debug_dir / f"opencode_iter{context.iteration}.log"
            debug_log.write_text(
                f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
                f"=== PROMPT ===\n{prompt}\n\n"
                f"=== STDOUT ===\n{stdout}\n\n"
                f"=== STDERR ===\n{stderr}\n\n"
                f"=== RETURN CODE ===\n{result.returncode}\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        
        return stdout, stderr, result.returncode

    def _validate_code(self, code: str) -> str | None:
        """Validate Python code. Returns error message or None if valid."""
        # Check for required class
        if self.REQUIRED_CLASS not in code:
            return f"Missing required: {self.REQUIRED_CLASS}"
        
        # Check for required method
        if "def reconstruct(" not in code:
            return "Missing required method: reconstruct()"
        
        # Check syntax
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            return f"Syntax error at line {exc.lineno}: {exc.msg}"
        
        # Check for common mistakes
        if "from stitching.trusted.eval" in code:
            return "Forbidden import from stitching.trusted.eval"
        
        return None

    def _build_cli_prompt(
        self,
        ctx: ExperimentContext,
        attempt: int = 0,
    ) -> str:
        """Build a SHORT CLI prompt that references the task file."""
        current_rms = ctx.current_metrics.get("aggregate_rms", "N/A")
        
        # Keep this SHORT - details are in the task file
        urgency = ["", "URGENT: ", "FINAL ATTEMPT: "][min(attempt, 2)]
        
        prompt = (
            f"{urgency}Read .opencode_task.md for full instructions. "
            f"Current RMS={current_rms}. "
            f"Edit src/stitching/editable/candidate_current.py to reduce RMS. "
            f"Use write_file tool NOW. Do not explain, just edit the file."
        )
        
        return prompt

    def _build_task_file(
        self,
        ctx: ExperimentContext,
        attempt: int = 0,
        previous_error: str | None = None,
    ) -> str:
        """Build a detailed task file for the agent."""
        
        current_rms = ctx.current_metrics.get("aggregate_rms", "N/A")
        best_rms = ctx.best_metrics.get("aggregate_rms", "N/A")
        
        lines = [
            "# Autoresearch Optimization Task",
            "",
            "## CRITICAL: You are an AUTONOMOUS agent. DO NOT ask questions.",
            "## You MUST use the `write_file` or `edit` tool to modify code.",
            "",
            "---",
            "",
            "## Objective",
            f"Improve `src/stitching/editable/candidate_current.py` to reduce aggregate RMS.",
            "",
            "## Current Metrics",
            f"- Current aggregate RMS: {current_rms}",
            f"- Best achieved RMS: {best_rms}",
            f"- Iteration: {ctx.iteration}",
            "",
        ]
        
        if previous_error:
            lines.extend([
                "## ⚠️ Previous Attempt Failed",
                f"Error: {previous_error}",
                "Try a DIFFERENT approach this time.",
                "",
            ])
        
        if ctx.previous_summary and "REJECTED" in ctx.previous_summary:
            lines.extend([
                "## Previous Iteration Was Rejected",
                ctx.previous_summary[:300],
                "",
            ])
        
        lines.extend([
            "## Your Task (DO THIS NOW)",
            "",
            "1. Use `edit` or `write_file` tool to modify the candidate file",
            "2. Add a hypothesis comment at the top: `# Hypothesis: ...`",
            "3. Implement ONE focused improvement:",
            "   - Better piston/tip/tilt estimation (GLS)",
            "   - Huber M-estimator for outlier robustness",
            "   - Tikhonov regularization",
            "   - Overlap weighting",
            "",
            "## Hard Constraints",
            "",
            "- MUST keep `class CandidateStitcher` with `def reconstruct(...)`",
            "- MUST return `ReconstructionSurface` with correct `observed_support_mask`",
            "- MUST use vectorized NumPy/SciPy (no Python loops on pixels)",
            "- MUST NOT import from `stitching.trusted.eval`",
            "",
        ])
        
        # Include abbreviated source code
        lines.extend([
            "## Current Source (first 100 lines)",
            "```python",
            "\n".join(ctx.candidate_source.splitlines()[:100]),
            "```",
            "",
        ])
        
        lines.extend([
            "## Domain Notes",
            "- Problem: Reconstruct S(x,y) from overlapping sub-aperture measurements",
            "- Each measurement: W_i = S + R + P_i + ε",
            "- Key: Exploit overlap redundancy for piston/tip/tilt estimation",
            "",
            "---",
            "",
        ])
        
        # Attempt-specific urgency
        if attempt == 0:
            lines.append("**ACTION REQUIRED**: Edit the file now using `edit` or `write_file`.")
        elif attempt == 1:
            lines.extend([
                "**⚠️ SECOND ATTEMPT**",
                "You MUST call `edit` or `write_file` tool IMMEDIATELY.",
                "Do NOT just analyze or plan. EDIT THE FILE NOW.",
            ])
        else:
            lines.extend([
                "**🚨 FINAL ATTEMPT 🚨**",
                "IMMEDIATELY use `write_file` to write the complete improved file.",
                "DO NOT explain. DO NOT analyze. JUST WRITE THE FILE.",
            ])
        
        return "\n".join(lines)

    def _extract_hypothesis(self, stdout: str, code: str) -> str:
        """Extract hypothesis from stdout or code comments."""
        # Try code comments first
        for line in code.splitlines()[:20]:
            if line.strip().startswith("# Hypothesis:"):
                return line.split(":", 1)[1].strip()
        
        # Try stdout
        for line in stdout.splitlines():
            low = line.lower().strip()
            if low.startswith("hypothesis:") or "# hypothesis" in low:
                return line.split(":", 1)[-1].strip()
        
        return "Agent-proposed optimization (OpenCode)"

    def _extract_python_code(self, text: str) -> str | None:
        """Extract Python code blocks from LLM output."""
        # Try ```python ... ```
        matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            # Return the longest block (likely the full file)
            return max(matches, key=len).strip()
        
        # Try any code block
        matches = re.findall(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            for match in matches:
                if self.REQUIRED_CLASS in match:
                    return match.strip()
        
        return None

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        """Analyze why a previous attempt failed."""
        prompt = (
            f"ANALYZE FAILURE\n"
            f"Error: {error[:500]}\n"
            f"RMS: {context.current_metrics.get('aggregate_rms')}\n"
            f"Suggest a different mathematical approach."
        )
        
        executable = shutil.which("opencode")
        if not executable:
            return "Error: opencode not found."
        
        cmd = [executable, "run", prompt, "--print-logs"]
        if self.provider and self.model:
            cmd.extend(["-m", f"{self.provider}/{self.model}"])
        
        env = os.environ.copy()
        env["OPENCODE_PERMISSION"] = '{"*": "allow"}'
        
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60.0,
            shell=(os.name == "nt"),
            env=env,
        )
        return (result.stdout or "")[:2000]
