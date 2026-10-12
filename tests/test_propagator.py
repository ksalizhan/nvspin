import numpy as np
import pytest

from nvspin.solvers import (
    propagate_density_matrix,
    propagate_state,
    unitary_propagator,
)


def test_zero_duration_identity():

    H = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    U = unitary_propagator(
        H,
        0.0,
    )

    np.testing.assert_allclose(
        U,
        np.eye(2),
        atol=1e-12,
    )


def test_propagator_is_unitary():

    H = np.array(
        [
            [0.0, 0.1],
            [0.1, 1.0],
        ],
        dtype=complex,
    )

    U = unitary_propagator(
        H,
        25.0,
    )

    np.testing.assert_allclose(
        U.conj().T @ U,
        np.eye(2),
        atol=1e-12,
    )


def test_known_phase_accumulation():

    # State |1> has energy 0.010 GHz.
    #
    # After 50 ns:
    #
    # phase = 2*pi*(0.010)*(50) = pi
    #
    # therefore amplitude -> -1.

    H = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.010],
        ],
        dtype=complex,
    )

    psi = np.array(
        [
            0.0,
            1.0,
        ],
        dtype=complex,
    )

    final_state = propagate_state(
        psi,
        H,
        50.0,
    )

    expected = np.array(
        [
            0.0,
            -1.0,
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(
        final_state,
        expected,
        atol=1e-12,
    )


def test_state_norm_preserved():

    H = np.array(
        [
            [0.0, 0.02],
            [0.02, 0.01],
        ],
        dtype=complex,
    )

    psi = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    final_state = propagate_state(
        psi,
        H,
        100.0,
    )

    np.testing.assert_allclose(
        np.vdot(
            final_state,
            final_state,
        ),
        1.0,
        atol=1e-12,
    )


def test_density_matrix_trace_preserved():

    H = np.array(
        [
            [0.0, 0.01],
            [0.01, 0.0],
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

    final_rho = propagate_density_matrix(
        rho,
        H,
        50.0,
    )

    np.testing.assert_allclose(
        np.trace(final_rho),
        1.0,
        atol=1e-12,
    )


def test_negative_duration_rejected():

    H = np.eye(2)

    with pytest.raises(
        ValueError
    ):
        unitary_propagator(
            H,
            -1.0,
        )