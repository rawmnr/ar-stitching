"""Optional adapters for circular-pupil Zernike or Fringe basis generation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


def _normalized_pupil_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized x/y coordinates and unit-disk mask."""

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    x = 2.0 * xx / max(cols - 1, 1) - 1.0
    y = 2.0 * yy / max(rows - 1, 1) - 1.0
    mask = x**2 + y**2 <= 1.0
    return x, y, mask


def _resolve_backend(backend: str) -> str:
    """Resolve an installed optional backend."""

    if backend != "auto":
        return backend

    for candidate in ("optiland", "prysm"):
        try:
            import_module(candidate)
            return candidate
        except ImportError:
            continue
    raise ImportError("No optional Zernike backend is installed. Install the 'optics' or 'zernike_alt' extra.")


def _generate_with_optiland(coefficients: np.ndarray, shape: tuple[int, int], indexing: str) -> np.ndarray:
    """Call an Optiland backend if available."""

    optiland = import_module("optiland")
    x, y, mask = _normalized_pupil_grid(shape)

    if hasattr(optiland, "Zernike"):
        generator = optiland.Zernike(coefficients=coefficients.tolist(), indexing=indexing)
        surface = np.asarray(generator(x=x, y=y), dtype=float)
    else:
        raise ImportError("Installed Optiland package does not expose the expected Zernike API.")

    return np.where(mask, surface, 0.0)


def _generate_with_prysm(coefficients: np.ndarray, shape: tuple[int, int], indexing: str) -> np.ndarray:
    """Call a Prysm backend if available."""

    polynomials = import_module("prysm.polynomials")
    x, y, mask = _normalized_pupil_grid(shape)
    surface = np.zeros(shape, dtype=float)

    for term_index, coefficient in enumerate(np.asarray(coefficients, dtype=float).ravel(), start=1):
        if coefficient == 0.0:
            continue
        if indexing not in {"noll", "fringe", "ansi"}:
            raise ValueError(f"Unsupported Prysm indexing '{indexing}'.")
        if hasattr(polynomials, "zernike_nm"):
            # Adapter intentionally keeps the interface narrow; exact index mapping is backend-specific.
            raise NotImplementedError("Prysm backend support is not wired yet for indexed coefficient generation.")
        raise ImportError("Installed Prysm package does not expose the expected zernike API.")

    return np.where(mask, surface, 0.0)


def generate_zernike_surface(
    coefficients: np.ndarray,
    shape: tuple[int, int],
    indexing: str = "noll",
    backend: str = "auto",
) -> np.ndarray:
    """Generate a circular-pupil surface using an optional backend adapter."""

    resolved_backend = _resolve_backend(backend)
    coeffs = np.asarray(coefficients, dtype=float)

    if resolved_backend == "optiland":
        return _generate_with_optiland(coeffs, shape, indexing)
    if resolved_backend == "prysm":
        return _generate_with_prysm(coeffs, shape, indexing)
    raise ValueError(f"Unsupported Zernike backend '{resolved_backend}'.")
