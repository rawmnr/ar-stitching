from importlib import import_module

import numpy as np
import pytest

from stitching.trusted.bases import zernike
from stitching.trusted.bases.zernike import generate_zernike_surface
from stitching.trusted.surface.generation import generate_identity_surface, surface_from_basis


def test_surface_from_basis_legendre_returns_surface_truth() -> None:
    coefficients = np.zeros((2, 2), dtype=float)
    coefficients[0, 1] = 0.5

    truth = surface_from_basis(shape=(5, 5), pixel_size=1.0, basis_name="legendre", coefficients=coefficients)

    assert truth.z.shape == (5, 5)
    assert truth.valid_mask.shape == (5, 5)
    assert truth.metadata["surface_model"] == "legendre"


def test_surface_from_basis_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError):
        surface_from_basis(shape=(3, 3), pixel_size=1.0, basis_name="unknown", coefficients=np.ones((1, 1)))


def test_identity_surface_uses_legendre_generation() -> None:
    truth = generate_identity_surface((5, 5), pixel_size=1.0)

    assert truth.metadata["surface_model"] == "legendre_structured_low_order"
    assert np.std(truth.z) > 0.0


def test_identity_surface_has_nontrivial_edge_variation() -> None:
    truth = generate_identity_surface((9, 9), pixel_size=1.0)

    top_edge = truth.z[0, :]
    bottom_edge = truth.z[-1, :]
    assert np.std(top_edge) > 0.0
    assert np.std(bottom_edge) > 0.0
    assert not np.allclose(top_edge, bottom_edge)


def test_zernike_internal_backend_generates_circular_surface() -> None:
    surface = generate_zernike_surface(coefficients=np.array([0.1, 0.2]), shape=(9, 9), backend="internal")

    assert surface.shape == (9, 9)
    assert np.std(surface) > 0.0
    assert surface[0, 0] == 0.0


def test_zernike_internal_basic_modes_are_consistent() -> None:
    piston = generate_zernike_surface(coefficients=np.array([1.0]), shape=(9, 9), indexing="noll", backend="internal")
    tilt = generate_zernike_surface(coefficients=np.array([0.0, 1.0]), shape=(9, 9), indexing="noll", backend="internal")
    mask = piston != 0.0

    assert np.allclose(piston[mask], 1.0)
    assert np.isclose(np.mean(tilt[mask]), 0.0, atol=1e-12)
    assert np.allclose(tilt[:, 4], -np.flipud(tilt[:, 4]))


def test_zernike_auto_backend_falls_back_to_internal_when_optional_backends_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):
        if name in {"optiland", "prysm"}:
            raise ImportError(name)
        return import_module(name)

    monkeypatch.setattr(zernike, "import_module", fake_import_module)

    surface = generate_zernike_surface(coefficients=np.array([0.1, 0.2]), shape=(9, 9), backend="auto")

    assert surface.shape == (9, 9)
    assert np.std(surface) > 0.0


def test_zernike_auto_backend_prefers_first_available_optional_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOptiland:
        pass

    def fake_import_module(name: str):
        if name == "optiland":
            return FakeOptiland()
        raise ImportError(name)

    monkeypatch.setattr(zernike, "import_module", fake_import_module)

    assert zernike._resolve_backend("auto") == "optiland"


def test_zernike_prysm_backend_is_explicitly_unavailable() -> None:
    with pytest.raises(NotImplementedError):
        generate_zernike_surface(coefficients=np.array([0.1, 0.2]), shape=(9, 9), backend="prysm")


def test_surface_from_basis_supports_zernike() -> None:
    truth = surface_from_basis(
        shape=(9, 9),
        pixel_size=1.0,
        basis_name="zernike",
        coefficients=np.array([0.1, 0.2]),
    )

    assert truth.z.shape == (9, 9)
    assert truth.valid_mask.shape == (9, 9)
    assert truth.valid_mask[0, 0] is np.False_
    assert truth.metadata["surface_model"] == "zernike"


def test_zernike_internal_matches_optiland_when_available() -> None:
    try:
        import_module("optiland")
    except ImportError:
        pytest.skip("optiland extra not installed")

    coefficients = np.array([0.05, -0.02, 0.01], dtype=float)
    internal = generate_zernike_surface(coefficients=coefficients, shape=(33, 33), backend="internal")
    external = generate_zernike_surface(coefficients=coefficients, shape=(33, 33), backend="optiland")

    assert internal.shape == external.shape
    assert np.allclose(internal, external, atol=1e-6, rtol=1e-6)
