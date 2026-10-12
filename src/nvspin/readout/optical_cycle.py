"""Rate-equation model of the NV optical cycle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


FloatArray = NDArray[np.float64]


@dataclass
class NVOpticalRates:
    """
    Five-level NV optical-cycle rates.

    States
    ------
    0 : |g, ms=0>
    1 : |g, ms=+/-1>
    2 : |e, ms=0>
    3 : |e, ms=+/-1>
    4 : metastable singlet manifold

    All rates are in 1/ns.

    Parameters
    ----------
    excitation
        Spin-conserving optical excitation rate.

    radiative_ms0
        Radiative decay rate e0 -> g0.

    radiative_ms1
        Radiative decay rate e1 -> g1.

    isc_ms0
        Intersystem-crossing rate e0 -> singlet.

    isc_ms1
        Intersystem-crossing rate e1 -> singlet.

    singlet_to_ms0
        Singlet decay rate -> ground ms=0.

    singlet_to_ms1
        Singlet decay rate -> ground ms=+/-1.
    """

    excitation: float

    radiative_ms0: float
    radiative_ms1: float

    isc_ms0: float
    isc_ms1: float

    singlet_to_ms0: float
    singlet_to_ms1: float

    def __post_init__(self) -> None:

        for name, value in vars(self).items():

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )


class NVOpticalCycle:
    """
    Five-level NV optical rate-equation model.

    Population dynamics obey

        dp/dt = M p

    where p is a five-component population vector.
    """

    dimension = 5

    def __init__(
        self,
        rates: NVOpticalRates,
    ) -> None:

        self.rates = rates

    def rate_matrix(self) -> FloatArray:
        """
        Construct the population-rate matrix M.

        Convention:

            dp/dt = M p

        Therefore column j contains rates flowing out of
        state j and into the other states.
        """

        r = self.rates

        M = np.zeros(
            (5, 5),
            dtype=float,
        )

        # =====================================================
        # Ground ms=0
        # =====================================================

        # g0 -> e0
        M[2, 0] += r.excitation
        M[0, 0] -= r.excitation

        # =====================================================
        # Ground ms=+/-1
        # =====================================================

        # g1 -> e1
        M[3, 1] += r.excitation
        M[1, 1] -= r.excitation

        # =====================================================
        # Excited ms=0
        # =====================================================

        # e0 -> g0 radiative
        M[0, 2] += r.radiative_ms0
        M[2, 2] -= r.radiative_ms0

        # e0 -> singlet
        M[4, 2] += r.isc_ms0
        M[2, 2] -= r.isc_ms0

        # =====================================================
        # Excited ms=+/-1
        # =====================================================

        # e1 -> g1 radiative
        M[1, 3] += r.radiative_ms1
        M[3, 3] -= r.radiative_ms1

        # e1 -> singlet
        M[4, 3] += r.isc_ms1
        M[3, 3] -= r.isc_ms1

        # =====================================================
        # Singlet
        # =====================================================

        # singlet -> g0
        M[0, 4] += r.singlet_to_ms0
        M[4, 4] -= r.singlet_to_ms0

        # singlet -> g1
        M[1, 4] += r.singlet_to_ms1
        M[4, 4] -= r.singlet_to_ms1

        return M

    def propagate(
        self,
        populations: FloatArray,
        duration: float,
    ) -> FloatArray:
        """
        Propagate optical populations for duration in ns.
        """

        if duration < 0:
            raise ValueError(
                "duration must be non-negative."
            )

        p = np.asarray(
            populations,
            dtype=float,
        )

        if p.shape != (5,):
            raise ValueError(
                "populations must have shape (5,)."
            )

        if np.any(p < 0):
            raise ValueError(
                "populations must be non-negative."
            )

        total = np.sum(p)

        if np.isclose(total, 0.0):
            raise ValueError(
                "population vector cannot be zero."
            )

        p = p / total

        return (
            expm(
                self.rate_matrix()
                * duration
            )
            @ p
        )

    def steady_state(self) -> FloatArray:
        """
        Calculate the normalized steady-state populations.

        Solves

            M p = 0

        together with

            sum(p) = 1.
        """

        M = self.rate_matrix()

        A = M.copy()
        b = np.zeros(
            5,
            dtype=float,
        )

        # Replace final equation with normalization.
        A[-1, :] = 1.0
        b[-1] = 1.0

        populations = np.linalg.solve(
            A,
            b,
        )

        return populations