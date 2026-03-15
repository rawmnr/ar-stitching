"""Leaderboard management: tracks and ranks all accepted candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LeaderboardEntry:
    iteration: int
    aggregate_rms: float
    max_rms: float
    total_runtime_sec: float
    num_accepted: int
    hypothesis: str
    commit: str
    timestamp: str


class Leaderboard:
    """Maintains a sorted leaderboard of accepted candidates."""

    def __init__(self, experiments_dir: Path) -> None:
        self.leaderboard_path = experiments_dir / "leaderboard.json"
        self._entries: list[LeaderboardEntry] = self._load()

    def _load(self) -> list[LeaderboardEntry]:
        if not self.leaderboard_path.exists():
            return []
        data = json.loads(self.leaderboard_path.read_text(encoding="utf-8"))
        return [LeaderboardEntry(**e) for e in data]

    def add(self, entry: LeaderboardEntry) -> int:
        """Add an entry and return its rank (0-indexed)."""
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.aggregate_rms)
        self._save()
        return self._entries.index(entry)

    def top(self, n: int = 10) -> list[LeaderboardEntry]:
        return self._entries[:n]

    def best(self) -> LeaderboardEntry | None:
        return self._entries[0] if self._entries else None

    def to_markdown(self) -> str:
        lines = [
            "# Autoresearch Leaderboard",
            "",
            "| Rank | Iteration | Agg RMS | Max RMS | Runtime | Hypothesis |",
            "|------|-----------|---------|---------|---------|------------|",
        ]
        for rank, entry in enumerate(self._entries[:20], 1):
            lines.append(
                f"| {rank} | {entry.iteration} | {entry.aggregate_rms:.8f} | "
                f"{entry.max_rms:.8f} | {entry.total_runtime_sec:.1f}s | "
                f"{entry.hypothesis[:60]} |"
            )
        return "\n".join(lines)

    def _save(self) -> None:
        data = [
            {
                "iteration": e.iteration,
                "aggregate_rms": e.aggregate_rms,
                "max_rms": e.max_rms,
                "total_runtime_sec": e.total_runtime_sec,
                "num_accepted": e.num_accepted,
                "hypothesis": e.hypothesis,
                "commit": e.commit,
                "timestamp": e.timestamp,
            }
            for e in self._entries
        ]
        self.leaderboard_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )
