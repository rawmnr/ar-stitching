"""Typed data contracts shared across trusted and editable boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ArrayF64 = np.ndarray
ArrayBool = np.ndarray


@dataclass(frozen=True)
class SurfaceTruth:
    """Reference surface and valid footprint used as ground truth."""

    z: ArrayF64
    valid_mask: ArrayBool
    pixel_size: float
    units: str = "arb"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubApertureObservation:
    """Single sub-aperture measurement in detector coordinates."""

    observation_id: str
    z: ArrayF64
    valid_mask: ArrayBool
    translation_xy: tuple[float, float]
    rotation_deg: float
    reference_bias: float = 0.0
    nuisance_terms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionSurface:
    """Reconstructed surface candidate produced by editable algorithms."""

    z: ArrayF64
    valid_mask: ArrayBool
    source_observation_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioConfig:
    """Canonical experiment configuration loaded from versioned scenario files."""

    scenario_id: str
    description: str
    grid_shape: tuple[int, int]
    pixel_size: float
    scan_offsets: tuple[tuple[float, float], ...]
    rotation_deg: tuple[float, ...] = (0.0,)
    reference_bias: float = 0.0
    gaussian_noise_std: float = 0.0
    outlier_fraction: float = 0.0
    retrace_error: float = 0.0
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        """Load a scenario config from a YAML file."""

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            scenario_id=payload["scenario_id"],
            description=payload["description"],
            grid_shape=tuple(payload["grid_shape"]),
            pixel_size=float(payload["pixel_size"]),
            scan_offsets=tuple(tuple(offset) for offset in payload["scan_offsets"]),
            rotation_deg=tuple(payload.get("rotation_deg", [0.0])),
            reference_bias=float(payload.get("reference_bias", 0.0)),
            gaussian_noise_std=float(payload.get("gaussian_noise_std", 0.0)),
            outlier_fraction=float(payload.get("outlier_fraction", 0.0)),
            retrace_error=float(payload.get("retrace_error", 0.0)),
            seed=int(payload.get("seed", 0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvalReport:
    """Structured evaluation output for a single scenario run."""

    scenario_id: str
    geometry_metrics: dict[str, float]
    signal_metrics: dict[str, float]
    runtime_sec: float
    accepted: bool
    notes: tuple[str, ...] = ()
