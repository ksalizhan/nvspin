"""Hamiltonian construction for NV-center spin systems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .nv_center import NVCenter
from .operators import spin_operators


ComplexMatrix = NDArray[np.complex128]


def tensor_product(operators: list[ComplexMatrix]) -> ComplexMatrix:
    """Kronecker product of a list of operators."""

    result = np.array([[1.0]], dtype=complex)

    for operator in operators:
        result = np.kron(result, operator)

    return result


def embedded_operator(
    operator: ComplexMatrix,
    subsystem: int,
    dimensions: list[int],
) -> ComplexMatrix:
    """
    Embed an operator into the complete Hilbert space.

    subsystem=0 corresponds to the NV electronic spin.
    Nuclear spins follow in NVCenter.nuclei order.
    """

    operators = []

    for index, dimension in enumerate(dimensions):
        if index == subsystem:
            operators.append(operator)
        else:
            operators.append(
                np.eye(dimension, dtype=complex)
            )

    return tensor_product(operators)


def build_hamiltonian(system: NVCenter) -> ComplexMatrix:
    """
    Construct the complete ground-state spin Hamiltonian.

    Hamiltonian units are GHz.

    Includes
    --------
    - NV zero-field splitting
    - transverse strain parameter E
    - electron Zeeman interaction
    - nuclear Zeeman interactions
    - full hyperfine tensors
    - axial nuclear quadrupole interactions
    """

    nuclei = system.nuclei

    dimensions = [
        3,
        *[nucleus.dimension for nucleus in nuclei],
    ]

    total_dimension = int(np.prod(dimensions))

    H = np.zeros(
        (total_dimension, total_dimension),
        dtype=complex,
    )

    # ---------------------------------------------------------
    # NV electron operators
    # ---------------------------------------------------------

    sx0, sy0, sz0 = spin_operators(1.0)

    S = [
        embedded_operator(sx0, 0, dimensions),
        embedded_operator(sy0, 0, dimensions),
        embedded_operator(sz0, 0, dimensions),
    ]

    Sx, Sy, Sz = S

    # ---------------------------------------------------------
    # Zero-field splitting
    #
    # D Sz^2
    #
    # This differs from the traceless convention
    # D [Sz^2 - S(S+1)/3], but only by a constant energy
    # shift and therefore gives identical transition
    # frequencies.
    # ---------------------------------------------------------

    H += system.D * (Sz @ Sz)

    # ---------------------------------------------------------
    # Transverse strain / ZFS
    #
    # E (Sx^2 - Sy^2)
    # ---------------------------------------------------------

    H += system.E * (
        Sx @ Sx
        - Sy @ Sy
    )

    # ---------------------------------------------------------
    # Electron Zeeman
    #
    # gamma_e B . S
    # ---------------------------------------------------------

    Bx, By, Bz = system.B

    H += system.gamma_e * (
        Bx * Sx
        + By * Sy
        + Bz * Sz
    )

    # ---------------------------------------------------------
    # Nuclear terms
    # ---------------------------------------------------------

    for nucleus_index, nucleus in enumerate(
        nuclei,
        start=1,
    ):

        ix0, iy0, iz0 = spin_operators(
            nucleus.spin
        )

        I = [
            embedded_operator(
                ix0,
                nucleus_index,
                dimensions,
            ),
            embedded_operator(
                iy0,
                nucleus_index,
                dimensions,
            ),
            embedded_operator(
                iz0,
                nucleus_index,
                dimensions,
            ),
        ]

        Ix, Iy, Iz = I

        # Nuclear Zeeman:
        #
        # -gamma_n B . I

        H += -nucleus.gamma * (
            Bx * Ix
            + By * Iy
            + Bz * Iz
        )

        # Hyperfine:
        #
        # S . A . I
        #
        # Sum_ab S_a A_ab I_b

        for a in range(3):
            for b in range(3):
                H += (
                    nucleus.A[a, b]
                    * (S[a] @ I[b])
                )

        # Axial quadrupole:
        #
        # Q Iz^2
        #
        # A constant offset is irrelevant for transition
        # frequencies.

        if nucleus.quadrupole != 0.0:
            H += (
                nucleus.quadrupole
                * (Iz @ Iz)
            )

    return H


def eigensystem(
    system: NVCenter,
) -> tuple[
    NDArray[np.float64],
    ComplexMatrix,
]:
    """Return sorted eigenvalues and eigenvectors."""

    H = build_hamiltonian(system)

    energies, states = np.linalg.eigh(H)

    return energies, states