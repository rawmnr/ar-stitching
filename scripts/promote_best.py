#!/usr/bin/env python3
"""Promote the best accepted candidate to the baseline archive."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from stitching.analysis.leaderboard import Leaderboard


def main() -> None:
    repo_root = Path.cwd()
    lb = Leaderboard(repo_root / "experiments")
    best = lb.best()

    if best is None:
        print("No accepted candidates found.")
        return

    candidate_path = repo_root / "src" / "stitching" / "editable" / "candidate_current.py"
    archive_dir = repo_root / "src" / "stitching" / "editable" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"candidate_rms{best.aggregate_rms:.6f}_{ts}.py"
    shutil.copy2(candidate_path, archive_dir / archive_name)
    print(f"Promoted: {archive_name} (RMS={best.aggregate_rms:.8f})")

    # Update leaderboard markdown
    reports_dir = repo_root / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "leaderboard.md").write_text(
        lb.to_markdown(), encoding="utf-8",
    )
    print("Leaderboard updated.")


if __name__ == "__main__":
    main()
