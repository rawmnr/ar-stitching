"""Persistent experiment ledger: append-only scientific log."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stitching.harness.protocols import RunManifest, RunResult, RunVerdict


class Ledger:
    """Append-only ledger persisting structured run results to disk."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.runs_dir = experiments_dir / "runs"
        self.accepted_dir = experiments_dir / "accepted"
        self.rejected_dir = experiments_dir / "rejected"
        for d in (self.runs_dir, self.accepted_dir, self.rejected_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, result: RunResult) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        name = f"{ts}_{result.manifest.experiment_id}_iter{result.manifest.iteration:04d}"
        return self.runs_dir / name

    def record(self, result: RunResult) -> Path:
        """Persist a complete run result to disk."""
        run_dir = self._run_dir(result)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Manifest
        (run_dir / "manifest.json").write_text(
            json.dumps(_serialize(asdict(result.manifest)), indent=2),
            encoding="utf-8",
        )

        # Metrics
        (run_dir / "metrics.json").write_text(
            json.dumps(result.metrics, indent=2),
            encoding="utf-8",
        )

        # Diff
        (run_dir / "diff.patch").write_text(result.diff_patch, encoding="utf-8")

        # Summary
        summary_lines = [
            f"# Run Summary",
            f"",
            f"- **Experiment**: {result.manifest.experiment_id}",
            f"- **Iteration**: {result.manifest.iteration}",
            f"- **Agent**: {result.manifest.agent_backend}",
            f"- **Verdict**: {result.verdict.value}",
            f"- **Elapsed**: {result.elapsed_sec:.2f}s",
            f"- **Hypothesis**: {result.hypothesis}",
            f"",
            f"## Metrics",
            f"```json",
            json.dumps(result.metrics, indent=2),
            f"```",
        ]
        if result.notes:
            summary_lines.extend(["", "## Notes"])
            summary_lines.extend(f"- {n}" for n in result.notes)

        (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

        # Symlink into accepted/rejected
        target_dir = self.accepted_dir if result.verdict == RunVerdict.ACCEPTED else self.rejected_dir
        link = target_dir / run_dir.name
        if not link.exists():
            # Use relative symlink if possible, or just copy on Windows if symlinks are restricted
            try:
                link.symlink_to(run_dir.resolve(), target_is_directory=True)
            except OSError:
                # Fallback: write a small text file pointing to the run_dir
                (target_dir / f"{run_dir.name}.ptr").write_text(str(run_dir.resolve()))

        return run_dir

    def load_best_metrics(self) -> dict[str, float] | None:
        """Load the best accepted metrics so far."""
        best: dict[str, float] | None = None
        best_rms = float("inf")
        for accepted_link in sorted(self.accepted_dir.iterdir()):
            if accepted_link.is_dir():
                metrics_path = accepted_link / "metrics.json"
            elif accepted_link.suffix == ".ptr":
                metrics_path = Path(accepted_link.read_text().strip()) / "metrics.json"
            else:
                continue

            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rms = metrics.get("aggregate_rms", float("inf"))
            if rms < best_rms:
                best_rms = rms
                best = metrics
        return best

    def iteration_count(self, experiment_id: str) -> int:
        """Count completed iterations for a given experiment."""
        count = 0
        if self.runs_dir.exists():
            for run_dir in self.runs_dir.iterdir():
                if experiment_id in run_dir.name:
                    count += 1
        return count


def _serialize(obj: Any) -> Any:
    """Make dataclass dicts JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj
