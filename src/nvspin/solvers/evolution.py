"""Unified evolution backend for nvspin experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nvspin.noise import DecoherenceModel

from .master_equation import (
    propagate_master_equation,
)

from .propagator import (
    propagate_density_matrix,
)


ComplexArray = NDArray[np.complex128]


def evolve_density_matrix(
    density_matrix: ComplexArray,
    hamiltonian: ComplexArray,
    duration: float,
    decoherence: DecoherenceModel | None = None,
) -> ComplexArray:
    """
    Evolve a density matrix using the appropriate solver.

    Parameters
    ----------
    density_matrix
        Initial density matrix.

    hamiltonian
        Hamiltonian in GHz.

    duration
        Evolution duration in ns.

    decoherence
        Optional DecoherenceModel.

        If None:
            coherent unitary evolution is used.

        If provided:
            Lindblad master-equation evolution is used.

    Returns
    -------
    rho
        Final density matrix.

    Notes
    -----
    This function is intended to be the common evolution
    interface used by experiments.

    Experiments should not need to decide whether to call
    the unitary or Lindblad solver directly.
    """

    rho = np.asarray(
        density_matrix,
        dtype=complex,
    )

    H = np.asarray(
        hamiltonian,
        dtype=complex,
    )

    if rho.ndim != 2:
        raise ValueError(
            "density_matrix must be a matrix."
        )

    if rho.shape[0] != rho.shape[1]:
        raise ValueError(
            "density_matrix must be square."
        )

    if H.ndim != 2:
        raise ValueError(
            "hamiltonian must be a matrix."
        )

    if H.shape[0] != H.shape[1]:
        raise ValueError(
            "hamiltonian must be square."
        )

    if rho.shape != H.shape:
        raise ValueError(
            "density_matrix and hamiltonian "
            "dimensions must match."
        )

    if duration < 0:
        raise ValueError(
            "duration must be non-negative."
        )

    # ---------------------------------------------------------
    # Ideal coherent evolution
    # ---------------------------------------------------------

    if decoherence is None:

        return propagate_density_matrix(
            rho,
            H,
            duration,
        )

    # ---------------------------------------------------------
    # Open-system evolution
    # ---------------------------------------------------------

    collapse_operators = (
        decoherence.collapse_operators()
    )

    # If the model contains no active channels, deliberately
    # use the unitary backend. This guarantees exact agreement
    # with the ideal solver rather than merely numerical
    # agreement through an empty Liouvillian calculation.

    if len(collapse_operators) == 0:

        return propagate_density_matrix(
            rho,
            H,
            duration,
        )

    return propagate_master_equation(
        rho,
        H,
        duration,
        collapse_operators=(
            collapse_operators
        ),
    )


def population(
    density_matrix: ComplexArray,
    state_index: int,
) -> float:
    """Return the population of a basis state."""

    rho = np.asarray(
        density_matrix,
        dtype=complex,
    )

    if state_index < 0:
        raise ValueError(
            "state_index must be non-negative."
        )

    if state_index >= rho.shape[0]:
        raise ValueError(
            "state_index is outside the Hilbert space."
        )

    return float(
        np.real(
            rho[
                state_index,
                state_index,
            ]
        )
    )


def pure_state_density_matrix(
    state: ComplexArray,
) -> ComplexArray:
    """Construct |psi><psi| from a state vector."""

    psi = np.asarray(
        state,
        dtype=complex,
    )

    if psi.ndim != 1:
        raise ValueError(
            "state must be a vector."
        )

    norm = np.vdot(
        psi,
        psi,
    )

    if np.isclose(
        norm,
        0.0,
    ):
        raise ValueError(
            "state must have non-zero norm."
        )

    psi = (
        psi
        / np.sqrt(norm)
    )

    return np.outer(
        psi,
        psi.conj(),
    )