import numpy as np
import pytest

from nvspin.noise import (
    DecoherenceModel,
)

from nvspin.solvers import (
    evolve_density_matrix,
    population,
    propagate_density_matrix,
    pure_state_density_matrix,
)


def test_pure_state_density_matrix():

    psi = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    rho = pure_state_density_matrix(
        psi
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(
        rho,
        expected,
        atol=1e-12,
    )


def test_pure_state_is_normalized():

    psi = np.array(
        [
            2.0,
            0.0,
        ],
        dtype=complex,
    )

    rho = pure_state_density_matrix(
        psi
    )

    np.testing.assert_allclose(
        np.trace(rho),
        1.0,
        atol=1e-12,
    )


def test_population():

    rho = np.array(
        [
            [0.75, 0.0],
            [0.0, 0.25],
        ],
        dtype=complex,
    )

    assert np.isclose(
        population(rho, 0),
        0.75,
    )

    assert np.isclose(
        population(rho, 1),
        0.25,
    )


def test_no_decoherence_matches_unitary_backend():

    H = np.array(
        [
            [0.0, 0.005],
            [0.005, 0.0],
        ],
        dtype=complex,
    )

    rho = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    duration = 25.0

    expected = propagate_density_matrix(
        rho,
        H,
        duration,
    )

    result = evolve_density_matrix(
        rho,
        H,
        duration,
        decoherence=None,
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
    )


def test_empty_decoherence_model_matches_unitary():

    H = np.array(
        [
            [0.0, 0.005],
            [0.005, 0.0],
        ],
        dtype=complex,
    )

    rho = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    duration = 25.0

    expected = evolve_density_matrix(
        rho,
        H,
        duration,
        decoherence=None,
    )

    result = evolve_density_matrix(
        rho,
        H,
        duration,
        decoherence=DecoherenceModel(),
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
    )


def test_T1_through_evolution_backend():

    T1 = 1000.0

    H = np.zeros(
        (2, 2),
        dtype=complex,
    )

    rho_excited = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    result = evolve_density_matrix(
        rho_excited,
        H,
        duration=T1,
        decoherence=DecoherenceModel(
            T1=T1
        ),
    )

    np.testing.assert_allclose(
        population(
            result,
            1,
        ),
        np.exp(-1.0),
        atol=1e-12,
    )


def test_invalid_population_index():

    rho = np.eye(
        2,
        dtype=complex,
    ) / 2.0

    with pytest.raises(
        ValueError
    ):
        population(
            rho,
            2,
        )