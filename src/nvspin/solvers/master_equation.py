"""Lindblad master-equation solvers for nvspin."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


ComplexArray = NDArray[np.complex128]


def liouvillian(
    hamiltonian: ComplexArray,
    collapse_operators: list[ComplexArray] | None = None,
) -> ComplexArray:
    """
    Construct the Liouvillian superoperator.

    Parameters
    ----------
    hamiltonian
        Hamiltonian in GHz.

    collapse_operators
        Lindblad collapse operators with rates expressed
        in 1/ns through their amplitudes.

    Notes
    -----
    nvspin uses

        H in GHz
        t in ns

    so the coherent part is

        -i 2*pi [H, rho].

    Vectorization uses column-major ordering.
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

    dimension = H.shape[0]

    identity = np.eye(
        dimension,
        dtype=complex,
    )

    # ---------------------------------------------------------
    # Coherent contribution
    #
    # vec(H rho) = (I kron H) vec(rho)
    #
    # vec(rho H) = (H^T kron I) vec(rho)
    # ---------------------------------------------------------

    L = (
        -1j
        * 2.0
        * np.pi
        * (
            np.kron(
                identity,
                H,
            )
            -
            np.kron(
                H.T,
                identity,
            )
        )
    )

    if collapse_operators is None:
        collapse_operators = []

    # ---------------------------------------------------------
    # Lindblad dissipators
    # ---------------------------------------------------------

    for collapse in collapse_operators:

        C = np.asarray(
            collapse,
            dtype=complex,
        )

        if C.shape != H.shape:
            raise ValueError(
                "Collapse operator dimensions must match "
                "the Hamiltonian."
            )

        C_dagger_C = (
            C.conj().T
            @ C
        )

        jump = np.kron(
            C.conj(),
            C,
        )

        left = np.kron(
            identity,
            C_dagger_C,
        )

        right = np.kron(
            C_dagger_C.T,
            identity,
        )

        L += (
            jump
            - 0.5 * left
            - 0.5 * right
        )

    return L


def propagate_master_equation(
    density_matrix: ComplexArray,
    hamiltonian: ComplexArray,
    duration: float,
    collapse_operators: list[ComplexArray] | None = None,
) -> ComplexArray:
    """
    Propagate a density matrix under a time-independent
    Lindblad master equation.
    """

    if duration < 0:
        raise ValueError(
            "duration must be non-negative."
        )

    rho = np.asarray(
        density_matrix,
        dtype=complex,
    )

    H = np.asarray(
        hamiltonian,
        dtype=complex,
    )

    if rho.shape != H.shape:
        raise ValueError(
            "Density matrix and Hamiltonian dimensions "
            "must match."
        )

    L = liouvillian(
        H,
        collapse_operators,
    )

    rho_vector = rho.reshape(
        -1,
        order="F",
    )

    propagator = expm(
        L * duration
    )

    final_vector = (
        propagator
        @ rho_vector
    )

    final_rho = final_vector.reshape(
        rho.shape,
        order="F",
    )

    return final_rho