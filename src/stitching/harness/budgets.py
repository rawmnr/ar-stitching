"""Time and resource budget enforcement for autoresearch iterations."""

from __future__ import annotations

import signal
import time
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


class BudgetExceededError(RuntimeError):
    """Raised when a time or resource budget is exceeded."""


@dataclass(frozen=True)
class IterationBudget:
    """Resource limits for a single autoresearch iteration."""

    agent_time_sec: float = 120.0      # Time for the agent to propose a patch
    eval_time_sec: float = 300.0       # Time for trusted evaluation
    total_time_sec: float = 480.0      # Total wall-clock budget
    max_memory_mb: float = 8192.0      # Peak memory cap
    max_candidate_lines: int = 2000    # Prevent bloated candidates


@contextmanager
def time_guard(budget_sec: float, label: str = "operation") -> Generator[None, None, None]:
    """Context manager that raises BudgetExceededError after budget_sec."""

    # Note: signal.SIGALRM is not available on Windows.
    # On Windows, this falls back to a no-op guard, but the BudgetTracker
    # will still catch budget overruns at check-points.
    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum: int, frame: object) -> None:
        raise BudgetExceededError(f"{label} exceeded {budget_sec:.1f}s budget.")

    # Using SIGALRM for Unix-like systems
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, budget_sec)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class BudgetTracker:
    """Tracks elapsed time and enforces per-iteration budgets."""

    def __init__(self, budget: IterationBudget) -> None:
        self.budget = budget
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget.total_time_sec - self.elapsed)

    def check(self) -> None:
        if self.elapsed > self.budget.total_time_sec:
            raise BudgetExceededError(
                f"Total budget {self.budget.total_time_sec:.0f}s exceeded "
                f"(elapsed: {self.elapsed:.1f}s)."
            )
