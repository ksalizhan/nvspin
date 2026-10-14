"""DD spectroscopy with an explicit 13C and classical bath noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvspin.experiments.dd_spectroscopy import (
    SingleCarbon13,
    SingleCarbonDDSpectroscopy,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


FloatArray = NDArray[np.float64]


@dataclass
class NoisyDDSpectroscopyResult:
    """
    Result of DD spectroscopy with one explicit 13C and
    independent classical longitudinal noise.
    """

    pulse_spacings: FloatArray

    carbon_coherence: FloatArray
    bath_coherence: FloatArray
    total_coherence: FloatArray

    preservation_probability: FloatArray

    total_times: FloatArray


class SingleCarbonNoisyDDSpectroscopy:
    """
    Single-13C DD spectroscopy with an unresolved OU bath.

    Physics
    -------
    The resolved 13C is treated explicitly through the
    conditional nuclear Hamiltonians

        H0 = f_L I_z

        H1 = (f_L + A_parallel) I_z
             + A_perpendicular I_x.

    The unresolved environment is represented independently
    by classical longitudinal Ornstein-Uhlenbeck frequency
    noise

        H_noise(t) = delta_f(t) sigma_z / 2.

    Because the OU process is Gaussian and independent of
    the resolved 13C, the electron coherence factorizes:

        L_total = L_13C * L_bath.

    The OU contribution is evaluated analytically from its
    phase variance under the DD switching function:

        L_bath = exp[-Var(phi) / 2].

    Parameters
    ----------
    carbon
        Explicit weakly coupled 13C nucleus.

    number_of_pulses
        Number of ideal instantaneous electron pi pulses.

    spacing_range
        Center-to-center pi-pulse spacing range in ns.

    noise
        Ornstein-Uhlenbeck longitudinal frequency noise.

    points
        Number of spectroscopy points.

    Notes
    -----
    This model assumes:

    - ideal instantaneous pi pulses,
    - independent resolved nucleus and classical bath,
    - purely longitudinal classical OU noise,
    - Gaussian OU statistics,
    - no T1 relaxation,
    - no finite pulse duration,
    - no pulse errors.

    No Monte Carlo sampling is required for the OU bath.
    """

    def __init__(
        self,
        carbon: SingleCarbon13,
        number_of_pulses: int,
        spacing_range: tuple[float, float],
        noise: OrnsteinUhlenbeckNoise,
        points: int = 500,
    ) -> None:

        if number_of_pulses < 1:
            raise ValueError(
                "number_of_pulses must be at least 1."
            )

        spacing_min, spacing_max = (
            spacing_range
        )

        if spacing_min <= 0:
            raise ValueError(
                "Minimum pulse spacing must be positive."
            )

        if spacing_max <= spacing_min:
            raise ValueError(
                "spacing_range must satisfy max > min."
            )

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        self.carbon = carbon

        self.number_of_pulses = int(
            number_of_pulses
        )

        self.spacing_range = (
            float(spacing_min),
            float(spacing_max),
        )

        self.noise = noise

        self.points = int(
            points
        )

    # ========================================================
    # DD timing
    # ========================================================

    def _pulse_times(
        self,
        pulse_spacing: float,
    ) -> FloatArray:
        """
        Return centers of the ideal pi pulses.

        For N pulses:

            t_k = (k + 1/2) * pulse_spacing

        where

            k = 0, ..., N - 1.

        The total ideal sequence duration is

            T = N * pulse_spacing.
        """

        return (
            (
                np.arange(
                    self.number_of_pulses,
                    dtype=float,
                )
                + 0.5
            )
            * pulse_spacing
        )

    # ========================================================
    # OU bath coherence
    # ========================================================

    def _bath_coherence_at_spacing(
        self,
        pulse_spacing: float,
    ) -> float:
        """
        Calculate exact OU bath coherence for one DD spacing.

        For the ideal switching function y(t),

            phi =
                2*pi integral
                y(t) delta_f(t) dt.

        Since OU noise is zero-mean Gaussian,

            L_bath =
                <exp(i phi)>

        is exactly

            L_bath =
                exp[-Var(phi) / 2].

        OrnsteinUhlenbeckNoise.coherence_under_dd()
        evaluates this quantity analytically interval by
        interval.
        """

        total_time = (
            self.number_of_pulses
            * pulse_spacing
        )

        pulse_times = (
            self._pulse_times(
                pulse_spacing
            )
        )

        return (
            self.noise.coherence_under_dd(
                pulse_times=pulse_times,
                total_time=total_time,
            )
        )

    # ========================================================
    # Simulation
    # ========================================================

    def simulate(
        self,
    ) -> NoisyDDSpectroscopyResult:
        """
        Run the hybrid single-13C + OU spectroscopy scan.
        """

        pulse_spacings = np.linspace(
            self.spacing_range[0],
            self.spacing_range[1],
            self.points,
        )

        total_times = (
            self.number_of_pulses
            * pulse_spacings
        )

        # ----------------------------------------------------
        # Explicit coherent 13C contribution
        # ----------------------------------------------------

        carbon_experiment = (
            SingleCarbonDDSpectroscopy(
                carbon=self.carbon,
                number_of_pulses=(
                    self.number_of_pulses
                ),
                spacing_range=(
                    self.spacing_range
                ),
                points=self.points,
            )
        )

        carbon_result = (
            carbon_experiment.simulate()
        )

        carbon_coherence = (
            carbon_result.coherence
        )

        # ----------------------------------------------------
        # Analytic OU bath contribution
        # ----------------------------------------------------

        bath_coherence = np.zeros(
            self.points,
            dtype=float,
        )

        for index, spacing in enumerate(
            pulse_spacings
        ):

            bath_coherence[index] = (
                self._bath_coherence_at_spacing(
                    float(spacing)
                )
            )

        # ----------------------------------------------------
        # Independent-environment factorization
        #
        # L_total(tau)
        #
        #     = L_13C(tau)
        #       * L_OU(tau)
        # ----------------------------------------------------

        total_coherence = (
            carbon_coherence
            * bath_coherence
        )

        # ----------------------------------------------------
        # Electron preservation probability
        #
        #     P_x = (1 + L_total) / 2
        #
        # Note that total_coherence may be negative.
        # That is a coherent phase inversion and is physical.
        # ----------------------------------------------------

        preservation_probability = (
            0.5
            * (
                1.0
                + total_coherence
            )
        )

        return NoisyDDSpectroscopyResult(
            pulse_spacings=(
                pulse_spacings
            ),
            carbon_coherence=(
                carbon_coherence
            ),
            bath_coherence=(
                bath_coherence
            ),
            total_coherence=(
                total_coherence
            ),
            preservation_probability=(
                preservation_probability
            ),
            total_times=(
                total_times
            ),
        )