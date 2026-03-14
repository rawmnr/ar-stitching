"""Optional adapters for circular-pupil Zernike or Fringe basis generation."""

from __future__ import annotations

from importlib import import_module

import numpy as np
from scipy.special import factorial


def _normalized_pupil_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized x/y coordinates and unit-disk mask."""

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    x = 2.0 * xx / max(cols - 1, 1) - 1.0
    y = 2.0 * yy / max(rows - 1, 1) - 1.0
    mask = x**2 + y**2 <= 1.0
    return x, y, mask


def _polar_pupil_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized polar coordinates and a unit-disk mask."""

    x, y, mask = _normalized_pupil_grid(shape)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta, mask


def _ansi_pairs(num_terms: int) -> list[tuple[int, int]]:
    """Enumerate ANSI/OSA Zernike (n, m) pairs."""

    pairs: list[tuple[int, int]] = []
    n = 0
    while len(pairs) < num_terms:
        for m in range(-n, n + 1, 2):
            pairs.append((n, m))
            if len(pairs) == num_terms:
                break
        n += 1
    return pairs


def _noll_pairs(num_terms: int) -> list[tuple[int, int]]:
    """Enumerate Noll Zernike (n, m) pairs."""

    pairs: list[tuple[int, int]] = []
    n = 0
    while len(pairs) < num_terms:
        if n % 2 == 0:
            ordered_m = [0]
            for abs_m in range(2, n + 1, 2):
                ordered_m.extend((-abs_m, abs_m))
        else:
            ordered_m = []
            for abs_m in range(1, n + 1, 2):
                ordered_m.extend((-abs_m, abs_m))
        for m in ordered_m:
            pairs.append((n, m))
            if len(pairs) == num_terms:
                break
        n += 1
    return pairs


def _fringe_pairs(num_terms: int) -> list[tuple[int, int]]:
    """Enumerate a simple Fringe-style ordering."""

    pairs: list[tuple[int, int]] = []
    n = 0
    while len(pairs) < num_terms:
        ordered_m = [0] if n % 2 == 0 else []
        for abs_m in range(1 if n % 2 else 2, n + 1, 2):
            ordered_m.extend((abs_m, -abs_m))
        for m in ordered_m:
            pairs.append((n, m))
            if len(pairs) == num_terms:
                break
        n += 1
    return pairs


def _index_pairs(indexing: str, num_terms: int) -> list[tuple[int, int]]:
    """Resolve coefficient indexing to (n, m) mode pairs."""

    if indexing == "ansi":
        return _ansi_pairs(num_terms)
    if indexing == "noll":
        return _noll_pairs(num_terms)
    if indexing == "fringe":
        return _fringe_pairs(num_terms)
    raise ValueError(f"Unsupported Zernike indexing '{indexing}'.")


def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Evaluate the radial Zernike polynomial."""

    abs_m = abs(m)
    if (n - abs_m) % 2 != 0:
        return np.zeros_like(rho, dtype=float)

    radial = np.zeros_like(rho, dtype=float)
    max_k = (n - abs_m) // 2
    for k in range(max_k + 1):
        coefficient = (
            (-1.0) ** k
            * factorial(n - k, exact=False)
            / (
                factorial(k, exact=False)
                * factorial((n + abs_m) // 2 - k, exact=False)
                * factorial((n - abs_m) // 2 - k, exact=False)
            )
        )
        radial = radial + coefficient * rho ** (n - 2 * k)
    return radial


def _zernike_mode(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate one real-valued Zernike mode on a polar grid."""

    radial = _radial_polynomial(n, m, rho)
    if m == 0:
        return radial
    if m > 0:
        return radial * np.cos(m * theta)
    return radial * np.sin(abs(m) * theta)


def _generate_with_internal(coefficients: np.ndarray, shape: tuple[int, int], indexing: str) -> np.ndarray:
    """Generate Zernike surfaces with an internal SciPy-backed implementation."""

    coeffs = np.asarray(coefficients, dtype=float).ravel()
    rho, theta, mask = _polar_pupil_grid(shape)
    surface = np.zeros(shape, dtype=float)

    for coefficient, (n, m) in zip(coeffs, _index_pairs(indexing, len(coeffs)), strict=False):
        if coefficient == 0.0:
            continue
        surface = surface + float(coefficient) * _zernike_mode(n, m, rho, theta)

    return np.where(mask, surface, 0.0)


def _resolve_backend(backend: str) -> str:
    """Resolve an installed optional backend.

    `auto` currently considers only fully wired backends. Prysm is intentionally
    excluded until indexed coefficient generation is implemented end-to-end.
    """

    if backend != "auto":
        return backend

    for candidate in ("optiland",):
        try:
            import_module(candidate)
            return candidate
        except ImportError:
            continue
    return "internal"


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
    """Fail explicitly until the Prysm adapter is fully wired."""

    raise NotImplementedError(
        "Prysm backend is not part of the public trusted API yet. "
        "Use backend='internal', backend='optiland', or backend='auto'."
    )


def generate_zernike_surface(
    coefficients: np.ndarray,
    shape: tuple[int, int],
    indexing: str = "noll",
    backend: str = "auto",
) -> np.ndarray:
    """Generate a circular-pupil surface using a supported backend adapter.

    Publicly supported backends are:
    - `internal`
    - `optiland` if installed
    - `auto`, which currently resolves to `optiland` or falls back to `internal`

    `prysm` remains intentionally unavailable until its adapter is fully implemented.
    """

    resolved_backend = _resolve_backend(backend)
    coeffs = np.asarray(coefficients, dtype=float)

    if resolved_backend == "internal":
        return _generate_with_internal(coeffs, shape, indexing)
    if resolved_backend == "optiland":
        return _generate_with_optiland(coeffs, shape, indexing)
    if resolved_backend == "prysm":
        return _generate_with_prysm(coeffs, shape, indexing)
    raise ValueError(f"Unsupported Zernike backend '{resolved_backend}'.")
