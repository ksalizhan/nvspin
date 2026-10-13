"""Bloch-vector utilities for two-level spin control."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


SIGMA_X = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ],
    dtype=complex,
)

SIGMA_Y = np.array(
    [
        [0.0, -1j],
        [1j, 0.0],
    ],
    dtype=complex,
)

SIGMA_Z = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0],
    ],
    dtype=complex,
)


def bloch_vector(
    density_matrix: ComplexArray,
) -> FloatArray:
    """
    Convert a 2x2 density matrix into a Bloch vector.

    Returns
    -------
    array
        [<sigma_x>, <sigma_y>, <sigma_z>]
    """

    rho = np.asarray(
        density_matrix,
        dtype=complex,
    )

    if rho.shape != (2, 2):
        raise ValueError(
            "density_matrix must have shape (2, 2)."
        )

    x = np.real(
        np.trace(
            rho @ SIGMA_X
        )
    )

    y = np.real(
        np.trace(
            rho @ SIGMA_Y
        )
    )

    z = np.real(
        np.trace(
            rho @ SIGMA_Z
        )
    )

    return np.array(
        [x, y, z],
        dtype=float,
    )


def rotation_unitary(
    axis: str,
    angle: float,
) -> ComplexArray:
    """
    Construct an ideal single-qubit rotation.

    Parameters
    ----------
    axis
        "X" or "Y".

    angle
        Rotation angle in radians.
    """

    axis = axis.upper()

    if axis == "X":
        sigma = SIGMA_X

    elif axis == "Y":
        sigma = SIGMA_Y

    else:
        raise ValueError(
            "axis must be 'X' or 'Y'."
        )

    identity = np.eye(
        2,
        dtype=complex,
    )

    return (
        np.cos(angle / 2.0)
        * identity
        -
        1j
        * np.sin(angle / 2.0)
        * sigma
    )


def rotate_density_matrix(
    density_matrix: ComplexArray,
    axis: str,
    angle: float,
) -> ComplexArray:
    """Apply an ideal rotation to a density matrix."""

    U = rotation_unitary(
        axis,
        angle,
    )

    rho = np.asarray(
        density_matrix,
        dtype=complex,
    )

    return (
        U
        @ rho
        @ U.conj().T
    )