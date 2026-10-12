"""Markovian decoherence models for nvspin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


SIGMA_MINUS = np.array(
    [
        [0.0, 1.0],
        [0.0, 0.0],
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


@dataclass
class DecoherenceModel:
    """
    Markovian two-level decoherence model.

    Parameters
    ----------
    T1
        Longitudinal relaxation time in ns.

        If None, longitudinal relaxation is disabled.

    T_phi
        Pure-dephasing time in ns.

        If None, pure dephasing is disabled.

    Notes
    -----
    For this model:

        1/T2 = 1/(2*T1) + 1/T_phi

    whenever the corresponding channels are present.
    """

    T1: float | None = None
    T_phi: float | None = None

    def __post_init__(self) -> None:

        if self.T1 is not None:
            if self.T1 <= 0:
                raise ValueError(
                    "T1 must be positive."
                )

        if self.T_phi is not None:
            if self.T_phi <= 0:
                raise ValueError(
                    "T_phi must be positive."
                )

    @property
    def gamma1(self) -> float:
        """
        Longitudinal relaxation rate in 1/ns.
        """

        if self.T1 is None:
            return 0.0

        return 1.0 / self.T1

    @property
    def gamma_phi(self) -> float:
        """
        Pure-dephasing rate in 1/ns.
        """

        if self.T_phi is None:
            return 0.0

        return 1.0 / self.T_phi

    @property
    def T2(self) -> float:
        """
        Effective transverse coherence time in ns.

        Returns infinity when no transverse decoherence
        channel is active.
        """

        rate = (
            0.5 * self.gamma1
            + self.gamma_phi
        )

        if np.isclose(
            rate,
            0.0,
        ):
            return np.inf

        return 1.0 / rate

    def collapse_operators(
        self,
    ) -> list[ComplexMatrix]:
        """
        Construct Lindblad collapse operators.

        Returns
        -------
        list of matrices
            Collapse operators with rates in 1/ns.

        Notes
        -----
        For T1 relaxation:

            L1 = sqrt(1/T1) sigma_-

        For pure dephasing:

            Lphi = sqrt(1/(2*T_phi)) sigma_z

        With this convention, off-diagonal density-matrix
        elements decay as exp(-t/T_phi) from the pure
        dephasing channel.
        """

        operators: list[
            ComplexMatrix
        ] = []

        if self.T1 is not None:

            operators.append(
                np.sqrt(
                    1.0 / self.T1
                )
                * SIGMA_MINUS
            )

        if self.T_phi is not None:

            operators.append(
                np.sqrt(
                    1.0
                    / (
                        2.0
                        * self.T_phi
                    )
                )
                * SIGMA_Z
            )

        return operators