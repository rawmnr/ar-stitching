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
    
    # Validation
    REQUIRED_CLASS = "class CandidateStitcher"

    def __init__(
        self,
        repo_root: Path,
        provider: str = "llama-swap",
        model: str = "gpt-oss-prod",
        timeout_sec: float = 600.0,
        **kwargs,
    ) -> None:
        self.repo_root = repo_root
        self.provider = provider
        self.model = model
        self.timeout_sec = timeout_sec
        self.logger = logging.getLogger(__name__)

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
            
            # Build increasingly directive prompts on retry
            prompt = self._build_directive_prompt(
                context, 
                attempt=attempt,
                previous_error=last_error,
            )
            
            try:
                stdout_text, stderr_text, return_code = self._execute_opencode(
                    prompt, context, target_path
                )
                
                # Check if file was modified
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
        
        # Build command
        cmd = ["opencode", "run", prompt, "--print-logs"]
        if self.provider and self.model:
            cmd.extend(["-m", f"{self.provider}/{self.model}"])
        
        # Attach files
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
            timeout=self.timeout_sec,
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

    def _build_directive_prompt(
        self,
        ctx: ExperimentContext,
        attempt: int = 0,
        previous_error: str | None = None,
    ) -> str:
        """Build an increasingly directive prompt based on attempt number."""
        
        current_rms = ctx.current_metrics.get("aggregate_rms", "N/A")
        best_rms = ctx.best_metrics.get("aggregate_rms", "N/A")
        
        # Core directive - gets more urgent with each retry
        urgency = ["", "IMPORTANT: ", "CRITICAL: "][min(attempt, 2)]
        
        lines = [
            f"{urgency}YOU ARE AN AUTONOMOUS CODE OPTIMIZATION AGENT.",
            "",
            "=" * 60,
            "MISSION: Improve the optical stitching algorithm to reduce RMS error.",
            "=" * 60,
            "",
            "## CURRENT METRICS",
            f"- Current aggregate RMS: {current_rms}",
            f"- Best achieved RMS: {best_rms}",
            f"- Iteration: {ctx.iteration}",
            "",
        ]
        
        # Add error feedback if this is a retry
        if previous_error:
            lines.extend([
                "## ⚠️ PREVIOUS ATTEMPT FAILED",
                f"Error: {previous_error}",
                "You MUST fix this issue and try a DIFFERENT approach.",
                "",
            ])
        
        # Add previous rejection info
        if ctx.previous_summary and "REJECTED" in ctx.previous_summary:
            lines.extend([
                "## PREVIOUS ITERATION REJECTED",
                ctx.previous_summary[:500],
                "DO NOT repeat the same mistake. Try a different mathematical approach.",
                "",
            ])
        
        # Direct action instructions
        lines.extend([
            "## REQUIRED ACTIONS (IN ORDER)",
            "",
            "1. **IMMEDIATELY** use `write_file` to edit:",
            f"   `{ctx.editable_paths[0]}`",
            "",
            "2. **STATE YOUR HYPOTHESIS** as a comment at the top of the file:",
            "   `# Hypothesis: <one-sentence mathematical hypothesis>`",
            "",
            "3. **IMPLEMENT ONE FOCUSED CHANGE** that should reduce RMS:",
            "   - GLS piston+tip+tilt estimation",
            "   - Huber M-estimator for outlier robustness",
            "   - Tikhonov regularization (λ ∈ [1e-10, 1e-2])",
            "   - SNR-aware overlap weighting",
            "",
            "4. **VERIFY SYNTAX** by running:",
            "   `uv run python -m py_compile src/stitching/editable/candidate_current.py`",
            "",
        ])
        
        # Constraints
        lines.extend([
            "## CONSTRAINTS (VIOLATIONS = REJECTION)",
            "",
            "- MUST keep `class CandidateStitcher` with `def reconstruct(...)`",
            "- MUST return `ReconstructionSurface` with correct `observed_support_mask`",
            "- MUST use vectorized NumPy/SciPy (NO Python loops on pixels)",
            "- MUST NOT import from `stitching.trusted.eval`",
            "- MUST NOT apply aggressive smoothing (destroys high frequencies)",
            "- MUST NOT reduce valid_mask coverage to game metrics",
            "",
        ])
        
        # Include the current source code directly
        lines.extend([
            "## CURRENT SOURCE CODE",
            "```python",
            ctx.candidate_source,
            "```",
            "",
        ])
        
        # Domain knowledge (abbreviated)
        lines.extend([
            "## DOMAIN NOTES",
            "- Problem: Reconstruct S(x,y) from overlapping sub-aperture measurements",
            "- Each measurement: W_i = S + R + P_i + ε (reference bias, piston error, noise)",
            "- Key: Exploit overlap redundancy for piston/tip/tilt estimation",
            "- Use scipy.sparse for large systems",
            "",
        ])
        
        # Final directive based on attempt
        if attempt == 0:
            lines.append("NOW: Edit the file and implement your improvement.")
        elif attempt == 1:
            lines.extend([
                "⚠️ SECOND ATTEMPT: You MUST write code to the file.",
                "DO NOT explain or plan. IMMEDIATELY use write_file.",
            ])
        else:
            lines.extend([
                "🚨 FINAL ATTEMPT: Write the complete improved file NOW.",
                "Use write_file IMMEDIATELY. No planning, no explanation.",
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
