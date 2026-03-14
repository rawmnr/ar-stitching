"""Typed data contracts shared across trusted and editable boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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
    """Single local detector tile with known pose in the global frame."""

    observation_id: str
    z: ArrayF64
    valid_mask: ArrayBool
    tile_shape: tuple[int, int]
    center_xy: tuple[float, float]
    global_shape: tuple[int, int]
    rotation_deg: float
    reference_bias: float = 0.0
    nuisance_terms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def translation_xy(self) -> tuple[float, float]:
        """Return a derived convenience translation, not independent state.

        `ScenarioConfig.scan_offsets` defines requested scan motion at scenario level.
        `SubApertureObservation` stores the realized pose as `center_xy` in the global
        frame, and `translation_xy` is derived from that pose for consumers that still
        want center-relative offsets.
        """

        global_center_x = (self.global_shape[1] - 1) / 2.0
        global_center_y = (self.global_shape[0] - 1) / 2.0
        return self.center_xy[0] - global_center_x, self.center_xy[1] - global_center_y


@dataclass(frozen=True)
class ReconstructionSurface:
    """Reconstructed surface candidate produced by editable algorithms."""

    z: ArrayF64
    valid_mask: ArrayBool
    source_observation_ids: tuple[str, ...]
    observed_support_mask: ArrayBool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioConfig:
    """Canonical experiment configuration loaded from versioned scenario files."""

    scenario_id: str
    description: str
    grid_shape: tuple[int, int]
    pixel_size: float
    scan_offsets: tuple[tuple[float, float], ...]
    tile_shape: tuple[int, int] | None = None
    baseline_name: str = "mean"
    rotation_deg: tuple[float, ...] = (0.0,)
    reference_bias: float = 0.0
    gaussian_noise_std: float = 0.0
    outlier_fraction: float = 0.0
    retrace_error: float = 0.0
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _normalize_field(name: str, value: Any) -> Any:
        """Normalize YAML values into the dataclass field contract."""

        tuple_2d_fields = {"grid_shape", "tile_shape"}
        tuple_seq_fields = {"rotation_deg"}
        nested_tuple_fields = {"scan_offsets"}
        float_fields = {"pixel_size", "reference_bias", "gaussian_noise_std", "outlier_fraction", "retrace_error"}
        int_fields = {"seed"}
        str_fields = {"scenario_id", "description", "baseline_name"}

        if name in tuple_2d_fields:
            return None if value is None else tuple(value)
        if name in tuple_seq_fields:
            return tuple(value)
        if name in nested_tuple_fields:
            return tuple(tuple(item) for item in value)
        if name in float_fields:
            return float(value)
        if name in int_fields:
            return int(value)
        if name in str_fields:
            return str(value)
        if name == "metadata":
            return dict(value)
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        """Load a scenario config from a YAML file."""

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        field_names = {field_info.name for field_info in fields(cls)}
        config_kwargs: dict[str, Any] = {}

        for field_info in fields(cls):
            field_name = field_info.name
            if field_name == "metadata":
                continue
            if field_name in payload:
                config_kwargs[field_name] = cls._normalize_field(field_name, payload[field_name])

        extra_metadata = {key: value for key, value in payload.items() if key not in field_names}
        declared_metadata = dict(payload.get("metadata", {}))
        config_kwargs["metadata"] = {**declared_metadata, **extra_metadata}
        return cls(**config_kwargs)

    @property
    def effective_tile_shape(self) -> tuple[int, int]:
        """Return the configured detector tile shape or default to the global grid."""

        return self.grid_shape if self.tile_shape is None else self.tile_shape


@dataclass(frozen=True)
class EvalReport:
    """Structured evaluation output for a single scenario run."""

    scenario_id: str
    geometry_metrics: dict[str, float]
    signal_metrics: dict[str, float]
    runtime_sec: float
    accepted: bool
    notes: tuple[str, ...] = ()
