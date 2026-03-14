import numpy as np

from stitching.trusted.bases.legendre import evaluate_legendre_basis_1d, generate_legendre_surface, sample_legendre_basis_2d
from stitching.trusted.surface.generation import generate_identity_surface


def test_legendre_basis_1d_matches_low_order_closed_forms() -> None:
    basis = evaluate_legendre_basis_1d(length=5, max_degree=2)
    axis = np.linspace(-1.0, 1.0, num=5)

    assert np.allclose(basis[0], 1.0)
    assert np.allclose(basis[1], axis)
    assert np.allclose(basis[2], 0.5 * (3.0 * axis**2 - 1.0))


def test_tensor_legendre_basis_has_expected_shape() -> None:
    basis = sample_legendre_basis_2d(shape=(4, 6), max_degree_y=2, max_degree_x=3)

    assert basis.shape == (3, 4, 4, 6)


def test_generate_legendre_surface_from_single_tilt_term() -> None:
    coefficients = np.zeros((2, 2), dtype=float)
    coefficients[0, 1] = 1.0

    surface = generate_legendre_surface(coefficients, shape=(3, 5))

    expected = np.tile(np.linspace(-1.0, 1.0, num=5), (3, 1))
    assert np.allclose(surface, expected)


def test_identity_surface_is_nonflat_legendre_surface() -> None:
    truth = generate_identity_surface((7, 7), pixel_size=1.0)

    assert truth.metadata["surface_model"] == "legendre_structured_low_order"
    assert np.std(truth.z) > 0.0


def test_tensor_legendre_basis_is_approximately_discrete_orthogonal_on_small_grid() -> None:
    basis = sample_legendre_basis_2d(shape=(9, 9), max_degree_y=2, max_degree_x=2)
    flattened = basis.reshape(9, -1)
    gram = flattened @ flattened.T
    diagonal = np.diag(gram)

    assert np.all(diagonal > 0.0)
    for row in range(gram.shape[0]):
        for col in range(gram.shape[1]):
            if row == col:
                continue
            assert abs(gram[row, col]) < 0.25 * np.sqrt(diagonal[row] * diagonal[col])
