"""Unitary propagation utilities for nvspin."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


ComplexArray = NDArray[np.complex128]


def unitary_propagator(
    hamiltonian: ComplexArray,
    duration: float,
) -> ComplexArray:
    """
    Calculate unitary evolution for a time-independent
    Hamiltonian.

    Parameters
    ----------
    hamiltonian
        Hamiltonian in GHz.

    duration
        Evolution duration in ns.

    Returns
    -------
    U
        Unitary propagator

            U = exp(-i 2 pi H t)

    Notes
    -----
    nvspin uses:

        H : GHz
        t : ns

    Since

        1 GHz = 1 / ns,

    the phase accumulated by a frequency H over time t is

        2 pi H t.
    """

    H = np.asarray(
        hamiltonian,
        dtype=complex,
    )

    if H.ndim != 2:
        raise ValueError(
            "Hamiltonian must be a matrix."
        )

    if H.shape[0] != H.shape[1]:
        raise ValueError(
            "Hamiltonian must be square."
        )

    if duration < 0:
        raise ValueError(
            "duration must be non-negative."
        )

    return expm(
        -1j
        * 2.0
        * np.pi
        * H
        * duration
    )


def propagate_state(
    state: ComplexArray,
    hamiltonian: ComplexArray,
    duration: float,
) -> ComplexArray:
    """Propagate a state vector."""

    U = unitary_propagator(
        hamiltonian,
        duration,
    )

    psi = np.asarray(
        state,
        dtype=complex,
    )

    return U @ psi


def propagate_density_matrix(
    density_matrix: ComplexArray,
    hamiltonian: ComplexArray,
    duration: float,
) -> ComplexArray:
    """
    Propagate a density matrix coherently.

        rho(t) = U rho(0) U^dagger
    """

    U = unitary_propagator(
        hamiltonian,
        duration,
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