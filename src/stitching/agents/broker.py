"""Agent broker: selects and routes to the appropriate backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stitching.agents.codex_cli import CodexCliBackend
from stitching.agents.opencode_cli import OpenCodeCliBackend
from stitching.harness.protocols import AgentBackend


BACKEND_REGISTRY: dict[str, type] = {
    "codex": CodexCliBackend,
    "opencode": OpenCodeCliBackend,
}


def create_backend(
    backend_name: str,
    repo_root: Path,
    **kwargs: Any,
) -> AgentBackend:
    """Instantiate an agent backend by name."""
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    return BACKEND_REGISTRY[backend_name](repo_root=repo_root, **kwargs)
