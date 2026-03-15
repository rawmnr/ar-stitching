"""Core protocols defining the contracts between harness, agents, and algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from stitching.contracts import (
    EvalReport,
    ReconstructionSurface,
    ScenarioConfig,
    SubApertureObservation,
)


# ---------------------------------------------------------------------------
# Algorithm protocol – what every candidate must implement
# ---------------------------------------------------------------------------

@runtime_checkable
class CandidateAlgorithm(Protocol):
    """Contract that every editable stitching candidate must satisfy."""

    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        ...


# ---------------------------------------------------------------------------
# Agent protocol – what every backend (Codex, OpenCode, …) must expose
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentContext:
    """Read-only context handed to the agent for a single iteration."""

    experiment_id: str
    iteration: int
    current_metrics: dict[str, float]
    best_metrics: dict[str, float]
    previous_diff: str | None
    previous_summary: str | None
    candidate_source: str  # current candidate_current.py content
    editable_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    time_budget_sec: float
    domain_notes: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchProposal:
    """Structured output from an agent backend."""

    hypothesis: str  # mandatory scientific hypothesis
    diff: str  # unified diff or full file content
    target_files: tuple[str, ...]
    reasoning: str
    estimated_improvement: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PatchApplicationMode(Enum):
    UNIFIED_DIFF = "unified_diff"
    FULL_REPLACE = "full_replace"


@runtime_checkable
class AgentBackend(Protocol):
    """Contract for interchangeable agent backends."""

    @property
    def name(self) -> str:
        ...

    def propose_patch(self, context: ExperimentContext) -> PatchProposal:
        ...

    def analyze_failure(self, context: ExperimentContext, error: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Run manifest and result schemas
# ---------------------------------------------------------------------------

class RunVerdict(Enum):
    ACCEPTED = "accepted"
    REJECTED_REGRESSION = "rejected_regression"
    REJECTED_GUARDRAIL = "rejected_guardrail"
    REJECTED_TIMEOUT = "rejected_timeout"
    REJECTED_CRASH = "rejected_crash"
    REJECTED_EMPTY = "rejected_empty"


@dataclass(frozen=True)
class RunManifest:
    """Immutable record of what was attempted in a single iteration."""

    experiment_id: str
    iteration: int
    agent_backend: str
    prompt_hash: str
    source_commit: str
    scenario_ids: tuple[str, ...]
    time_budget_sec: float
    timestamp_utc: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """Immutable record of what happened in a single iteration."""

    manifest: RunManifest
    verdict: RunVerdict
    metrics: dict[str, float]
    eval_reports: tuple[EvalReport, ...]
    hypothesis: str
    diff_patch: str
    elapsed_sec: float
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
