"""Git operations for the autoresearch loop: worktrees, commits, reverts."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GitOpsError(RuntimeError):
    """Raised when a git operation fails."""


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the result."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise GitOpsError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


class GitOps:
    """Encapsulates all git operations for the autoresearch harness."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        if not (self.repo_root / ".git").exists():
            raise GitOpsError(f"Not a git repository: {self.repo_root}")

    # ---- State queries ----

    def current_commit(self) -> str:
        result = _run_git(["rev-parse", "HEAD"], self.repo_root)
        return result.stdout.strip()

    def current_branch(self) -> str:
        result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], self.repo_root)
        return result.stdout.strip()

    def is_clean(self) -> bool:
        result = _run_git(["status", "--porcelain"], self.repo_root)
        return result.stdout.strip() == ""

    def diff_against(self, commit: str, paths: list[str] | None = None) -> str:
        args = ["diff", commit, "--"]
        if paths:
            args.extend(paths)
        result = _run_git(args, self.repo_root)
        return result.stdout

    # ---- Worktree management ----

    def create_worktree(self, worktree_path: Path, branch_name: str) -> Path:
        """Create an isolated worktree for one experiment iteration."""
        base_commit = self.current_commit()
        # Create a new branch from current HEAD
        _run_git(["branch", branch_name, base_commit], self.repo_root, check=False)
        _run_git(
            ["worktree", "add", str(worktree_path), branch_name],
            self.repo_root,
        )
        return worktree_path

    def remove_worktree(self, worktree_path: Path) -> None:
        _run_git(["worktree", "remove", "--force", str(worktree_path)], self.repo_root, check=False)

    def list_worktrees(self) -> list[str]:
        result = _run_git(["worktree", "list", "--porcelain"], self.repo_root)
        return [
            line.split(" ", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("worktree ")
        ]

    # ---- Commit / revert ----

    def stage_and_commit(
        self,
        paths: list[str],
        message: str,
        cwd: Path | None = None,
    ) -> str:
        """Stage specific paths and commit. Returns the new commit hash."""
        target = cwd or self.repo_root
        for p in paths:
            _run_git(["add", p], target)
        _run_git(["commit", "-m", message], target)
        return _run_git(["rev-parse", "HEAD"], target).stdout.strip()

    def revert_to(self, commit: str, cwd: Path | None = None) -> None:
        target = cwd or self.repo_root
        _run_git(["reset", "--hard", commit], target)

    def cherry_pick(self, commit: str, cwd: Path | None = None) -> None:
        target = cwd or self.repo_root
        _run_git(["cherry-pick", commit], target)

    def tag(self, tag_name: str, message: str = "") -> None:
        args = ["tag"]
        if message:
            args.extend(["-a", tag_name, "-m", message])
        else:
            args.append(tag_name)
        _run_git(args, self.repo_root)

    # ---- Diff extraction ----

    def generate_patch(self, from_commit: str, to_commit: str) -> str:
        result = _run_git(["diff", from_commit, to_commit], self.repo_root)
        return result.stdout

    # ---- Utility ----

    @staticmethod
    def timestamp_branch_name(experiment_id: str, iteration: int) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"autoresearch/{experiment_id}/iter_{iteration:04d}_{ts}"

    @staticmethod
    def prompt_hash(prompt_text: str) -> str:
        return hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
