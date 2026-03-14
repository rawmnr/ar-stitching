import numpy as np
import pytest

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

    assert truth.metadata["surface_model"] == "legendre_tilt_plus_weak_quadratic"
    assert np.std(truth.z) > 0.0


def test_zernike_adapter_fails_explicitly_without_backend() -> None:
    with pytest.raises((ImportError, NotImplementedError)):
        generate_zernike_surface(coefficients=np.array([0.1, 0.2]), shape=(9, 9), backend="auto")
