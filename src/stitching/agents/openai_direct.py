"""Direct OpenAI agent backend using the official Python SDK."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from stitching.harness.protocols import AgentBackend, ExperimentContext, PatchProposal

logger = logging.getLogger(__name__)


class OpenAiDirectBackend:
    """Agent backend calling OpenAI Chat Completions API directly."""

    def __init__(
        self,
        repo_root: Path,
        model: str = "gpt-4o",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        from openai import OpenAI

        self.repo_root = repo_root
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    @property
    def name(self) -> str:
        return f"openai-direct/{self.model}"

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        prompt = self._build_prompt(context)

        logger.info("Sending %d-char prompt to %s", len(prompt), self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=8192,
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from OpenAI API.")

        logger.debug("Response length: %d chars", len(content))
        return self._parse_response(content, context)

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        prompt = (
            f"An optical stitching algorithm failed evaluation.\n"
            f"Error: {error}\n"
            f"Current RMS: {context.current_metrics.get('aggregate_rms')}\n"
            f"Suggest a different mathematical approach."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content or "No analysis."

    def _build_prompt(self, ctx: ExperimentContext) -> str:
        parts = [
            "# Task",
            "",
            "Improve the optical stitching algorithm below.",
            "Return the **complete** updated Python file.",
            "",
            "## Current Performance",
            f"- Aggregate RMS: {ctx.current_metrics.get('aggregate_rms', 'N/A')}",
            f"- Best RMS ever: {ctx.best_metrics.get('aggregate_rms', 'N/A')}",
            f"- Scenarios: {', '.join(ctx.scenario_ids)}",
            f"- Iteration: {ctx.iteration}",
            "",
        ]

        if ctx.previous_summary:
            parts.extend([
                "## Previous Attempt",
                ctx.previous_summary[:1000],
                "",
            ])

        parts.extend([
            "## Domain Knowledge",
            ctx.domain_notes,
            "",
            "## Constraints",
            "- You MUST keep `class CandidateStitcher` with method `reconstruct`.",
            "- You MUST return a `ReconstructionSurface` with correct `observed_support_mask`.",
            "- You MUST use only: numpy, scipy, and imports from `stitching.contracts` / `stitching.trusted.scan.transforms`.",
            "- DO NOT import from `stitching.trusted.eval` or modify scoring.",
            "- Use vectorized NumPy/SciPy. No Python loops on individual pixels.",
            "- Runtime must stay under 300s per scenario.",
            "",
            "## Current Source Code",
            "```python",
            ctx.candidate_source,
            "```",
            "",
            "## Required Response Format",
            "",
            "HYPOTHESIS: <one-sentence mathematical hypothesis>",
            "",
            "REASONING: <brief explanation>",
            "",
            "```python",
            "<complete updated source code>",
            "```",
        ])
        return "\n".join(parts)

    def _parse_response(
        self, content: str, ctx: ExperimentContext,
    ) -> PatchProposal:
        """Extract hypothesis, reasoning, and code from the LLM response."""
        hypothesis = "Agent-proposed optimization"
        reasoning = ""

        # Extract hypothesis
        hyp_match = re.search(
            r"HYPOTHESIS:\s*(.+?)(?:\n|REASONING:)",
            content,
            re.DOTALL,
        )
        if hyp_match:
            hypothesis = hyp_match.group(1).strip()

        # Extract reasoning
        reason_match = re.search(
            r"REASONING:\s*(.+?)(?:```python)",
            content,
            re.DOTALL,
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()

        # Extract code — take the LAST python block (most likely the full file)
        code_blocks = re.findall(
            r"```python\s*\n(.*?)```",
            content,
            re.DOTALL,
        )

        if not code_blocks:
            raise RuntimeError(
                "No ```python``` code block found in LLM response. "
                f"Response starts with: {content[:200]}"
            )

        # Pick the longest code block (heuristic: full file > snippet)
        full_source = max(code_blocks, key=len).strip()

        # Sanity: must define CandidateStitcher
        if "class CandidateStitcher" not in full_source:
            raise RuntimeError(
                "Extracted code does not define CandidateStitcher. "
                f"Code starts with: {full_source[:200]}"
            )

        logger.info(
            "Parsed proposal: hypothesis='%s', code=%d chars",
            hypothesis[:80],
            len(full_source),
        )

        return PatchProposal(
            hypothesis=hypothesis,
            diff=content[:3000],
            target_files=ctx.editable_paths,
            reasoning=reasoning[:2000],
            full_source=full_source,
            changes_pre_applied=False,
        )


_SYSTEM_PROMPT = """\
You are an expert in optical metrology and computational interferometry.
You optimize sub-aperture stitching algorithms in Python.

Rules:
- Always provide a complete, runnable Python file.
- Always use the exact class name CandidateStitcher.
- Always provide HYPOTHESIS and REASONING before the code.
- Prefer scipy.sparse for large systems.
- Never smooth aggressively or mask pixels to game metrics.
- When estimating nuisance parameters (piston, tip, tilt), solve them
  simultaneously using global least squares on overlap zones.
"""
