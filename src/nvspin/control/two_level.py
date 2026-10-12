"""Effective two-level rotating-frame control Hamiltonians."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


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


def rotating_frame_hamiltonian(
    transition_frequency: float,
    microwave_frequency: float,
    rabi_frequency: float,
    phase: float = 0.0,
) -> ComplexMatrix:
    """
    Construct an effective two-level rotating-frame Hamiltonian.

    All frequencies are in GHz.

    Parameters
    ----------
    transition_frequency
        Unperturbed transition frequency f0.

    microwave_frequency
        Applied microwave carrier frequency f_MW.

    rabi_frequency
        On-resonance Rabi frequency f_R.

    phase
        Microwave phase in radians.

        phase = 0
            X rotation.

        phase = pi/2
            Y rotation.

    Returns
    -------
    H_rot
        2x2 rotating-frame Hamiltonian in GHz.

    Notes
    -----
    We define

        detuning = f0 - f_MW

    and

        H_rot =
            detuning/2 * sigma_z
            +
            f_R/2 * (
                cos(phi) sigma_x
                +
                sin(phi) sigma_y
            )

    nvspin uses

        U = exp(-i 2 pi H t)

    with H in GHz and t in ns.
    """

    if transition_frequency < 0:
        raise ValueError(
            "transition_frequency must be non-negative."
        )

    if microwave_frequency < 0:
        raise ValueError(
            "microwave_frequency must be non-negative."
        )

    if rabi_frequency < 0:
        raise ValueError(
            "rabi_frequency must be non-negative."
        )

    detuning = (
        transition_frequency
        - microwave_frequency
    )

    transverse_drive = (
        np.cos(phase) * SIGMA_X
        +
        np.sin(phase) * SIGMA_Y
    )

    H_rot = (
        0.5 * detuning * SIGMA_Z
        +
        0.5 * rabi_frequency
        * transverse_drive
    )

    return H_rot