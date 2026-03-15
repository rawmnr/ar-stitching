"""Simulated agent backend for testing the autoresearch loop."""

from __future__ import annotations

import time
from pathlib import Path

from stitching.harness.protocols import AgentBackend, ExperimentContext, PatchProposal


class SimulatedAgentBackend(AgentBackend):
    """Agent backend that returns a hardcoded (but valid) improvement."""

    def __init__(self, repo_root: Path, **kwargs) -> None:
        self.repo_root = repo_root

    @property
    def name(self) -> str:
        return "simulated/baseline-piston"

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        """Propose a simple improvement: use median instead of mean for pistons."""
        time.sleep(1) # Simulate thinking
        
        # A real patch would use a diff, but for the simulated one 
        # we return a slightly modified version of the candidate.
        # Here we just return a hypothesis to test the flow.
        return PatchProposal(
            hypothesis="Using nanmedian instead of nanmean in piston estimation should be more robust to small outliers.",
            diff="", # In a real scenario, this would be a diff
            target_files=context.editable_paths,
            reasoning="Median is a better estimator of the DC shift when the overlap region has edge artifacts.",
        )

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        return "Simulated analysis: try checking the overlap area size."
