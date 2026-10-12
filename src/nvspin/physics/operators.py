"""Spin operators used throughout nvspin."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


def spin_operators(spin: float) -> tuple[
    ComplexMatrix,
    ComplexMatrix,
    ComplexMatrix,
]:
    """
    Return Cartesian spin operators Sx, Sy, Sz.

    Parameters
    ----------
    spin
        Spin quantum number S. Must be a non-negative integer
        or half-integer.

    Returns
    -------
    Sx, Sy, Sz
        Dimensionless spin matrices in the basis

            |S>, |S-1>, ..., |-S>.

    Notes
    -----
    hbar is set to 1. Hamiltonian modules are responsible for
    attaching the appropriate frequency/energy units.
    """

    if spin < 0:
        raise ValueError("spin must be non-negative.")

    two_spin = 2.0 * spin

    if not np.isclose(two_spin, round(two_spin)):
        raise ValueError(
            "spin must be an integer or half-integer."
        )

    dimension = int(round(two_spin)) + 1

    m_values = np.arange(
        spin,
        -spin - 1.0,
        -1.0,
        dtype=float,
    )

    sz = np.diag(m_values).astype(complex)

    s_plus = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )

    for column in range(1, dimension):
        m = m_values[column]

        s_plus[column - 1, column] = np.sqrt(
            spin * (spin + 1.0)
            - m * (m + 1.0)
        )

    s_minus = s_plus.conj().T

    sx = 0.5 * (s_plus + s_minus)
    sy = (s_plus - s_minus) / (2.0j)

    return sx, sy, sz


def identity_for_spin(spin: float) -> ComplexMatrix:
    """Return the identity operator for a spin-S Hilbert space."""

    dimension = int(round(2.0 * spin)) + 1

    return np.eye(
        dimension,
        dtype=complex,
    )


def nv_electron_operators() -> tuple[
    ComplexMatrix,
    ComplexMatrix,
    ComplexMatrix,
]:
    """Return Sx, Sy, Sz for the NV electronic spin S=1."""

    return spin_operators(1.0)