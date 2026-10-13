import numpy as np

from nvspin.control import (
    bloch_vector,
    rotate_density_matrix,
    rotation_unitary,
)


def test_ground_state_bloch_vector():

    rho = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    vector = bloch_vector(
        rho
    )

    np.testing.assert_allclose(
        vector,
        [0.0, 0.0, 1.0],
        atol=1e-12,
    )


def test_pi_x_rotation():

    rho = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    final = rotate_density_matrix(
        rho,
        "X",
        np.pi,
    )

    vector = bloch_vector(
        final
    )

    np.testing.assert_allclose(
        vector,
        [0.0, 0.0, -1.0],
        atol=1e-12,
    )


def test_pi_over_2_x_rotation():

    rho = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    final = rotate_density_matrix(
        rho,
        "X",
        np.pi / 2.0,
    )

    vector = bloch_vector(
        final
    )

    np.testing.assert_allclose(
        vector,
        [0.0, -1.0, 0.0],
        atol=1e-12,
    )


def test_rotation_unitary_is_unitary():

    U = rotation_unitary(
        "Y",
        np.pi,
    )

    np.testing.assert_allclose(
        U.conj().T @ U,
        np.eye(2),
        atol=1e-12,
    )