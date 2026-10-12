import numpy as np

from nvspin.noise import (
    DecoherenceModel,
)

from nvspin.solvers import (
    propagate_density_matrix,
    propagate_master_equation,
)


def test_no_collapse_matches_unitary_density_matrix():

    H = np.array(
        [
            [0.0, 0.010],
            [0.010, 0.0],
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

    unitary_result = (
        propagate_density_matrix(
            rho,
            H,
            duration,
        )
    )

    master_result = (
        propagate_master_equation(
            rho,
            H,
            duration,
            collapse_operators=[],
        )
    )

    np.testing.assert_allclose(
        master_result,
        unitary_result,
        atol=1e-12,
    )


def test_master_equation_preserves_trace():

    H = np.zeros(
        (2, 2),
        dtype=complex,
    )

    rho = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    noise = DecoherenceModel(
        T1=1000.0,
        T_phi=500.0,
    )

    result = propagate_master_equation(
        rho,
        H,
        duration=500.0,
        collapse_operators=(
            noise.collapse_operators()
        ),
    )

    np.testing.assert_allclose(
        np.trace(result),
        1.0,
        atol=1e-12,
    )


def test_T1_population_decay():

    T1 = 1000.0

    H = np.zeros(
        (2, 2),
        dtype=complex,
    )

    # Initial excited state |1><1|
    rho = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    noise = DecoherenceModel(
        T1=T1,
    )

    result = propagate_master_equation(
        rho,
        H,
        duration=T1,
        collapse_operators=(
            noise.collapse_operators()
        ),
    )

    expected_excited_population = (
        np.exp(-1.0)
    )

    np.testing.assert_allclose(
        np.real(result[1, 1]),
        expected_excited_population,
        atol=1e-12,
    )


def test_pure_dephasing_does_not_change_population():

    H = np.zeros(
        (2, 2),
        dtype=complex,
    )

    rho = 0.5 * np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=complex,
    )

    noise = DecoherenceModel(
        T_phi=1000.0,
    )

    result = propagate_master_equation(
        rho,
        H,
        duration=1000.0,
        collapse_operators=(
            noise.collapse_operators()
        ),
    )

    np.testing.assert_allclose(
        np.real(
            np.diag(result)
        ),
        [0.5, 0.5],
        atol=1e-12,
    )


def test_pure_dephasing_coherence_at_Tphi():

    T_phi = 1000.0

    H = np.zeros(
        (2, 2),
        dtype=complex,
    )

    rho = 0.5 * np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=complex,
    )

    noise = DecoherenceModel(
        T_phi=T_phi,
    )

    result = propagate_master_equation(
        rho,
        H,
        duration=T_phi,
        collapse_operators=(
            noise.collapse_operators()
        ),
    )

    expected_coherence = (
        0.5
        * np.exp(-1.0)
    )

    np.testing.assert_allclose(
        np.real(
            result[0, 1]
        ),
        expected_coherence,
        atol=1e-12,
    )


def test_T1_contributes_half_rate_to_coherence():

    T1 = 1000.0

    H = np.zeros(
        (2, 2),
        dtype=complex,
    )

    rho = 0.5 * np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=complex,
    )

    noise = DecoherenceModel(
        T1=T1,
    )

    result = propagate_master_equation(
        rho,
        H,
        duration=T1,
        collapse_operators=(
            noise.collapse_operators()
        ),
    )

    expected_coherence = (
        0.5
        * np.exp(-0.5)
    )

    np.testing.assert_allclose(
        np.real(
            result[0, 1]
        ),
        expected_coherence,
        atol=1e-12,
    )