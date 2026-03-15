"""Agent backends for the stitching optimization harness."""

from stitching.agents.broker import create_backend
from stitching.agents.opencode_cli import OpenCodeCliBackend
from stitching.agents.codex_cli import CodexCliBackend

__all__ = ["create_backend", "OpenCodeCliBackend", "CodexCliBackend"]
