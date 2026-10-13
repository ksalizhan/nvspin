"""Dynamical-decoupling simulations under classical noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvspin.control import (
    DDSequence,
    bloch_vector,
    rotate_density_matrix,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)

from nvspin.solvers import (
    evolve_density_matrix,
)


FloatArray = NDArray[np.float64]


SIGMA_Z = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0],
    ],
    dtype=complex,
)


@dataclass
class DynamicalDecouplingResult:
    """Result of a stochastic DD simulation."""

    total_time: float
    average_bloch_vector: FloatArray

    coherence: float
    coherence_x: float
    coherence_y: float

    realizations: int


class DynamicalDecoupling:
    """
    Ideal-pulse DD simulation under classical frequency noise.

    Two timing modes are supported.

    Spectroscopy mode
    -----------------

        tau is specified directly.

    For N pulses:

        T = N * tau

    This will eventually be useful when scanning tau for
    nuclear-spin spectroscopy.

    Coherence mode
    --------------

        total_time is specified directly.

    For N pulses:

        tau = T / N

    This is useful for comparing different pulse numbers at
    the same total storage time.

    Sequence timing
    ---------------

        tau/2 - pi - tau - pi - ... - pi - tau/2

    Parameters
    ----------
    sequence
        DDSequence defining the phases of the refocusing
        pulses.

    noise
        Ornstein-Uhlenbeck frequency-noise model.

    tau
        Optional interpulse spacing in ns.

    total_time
        Optional total free-evolution time in ns.

    realizations
        Number of stochastic noise realizations.

    dt
        Noise/evolution integration step in ns.

    seed
        Random seed for reproducibility.

    Notes
    -----
    Exactly one of tau or total_time must be supplied.

    A zero-pulse sequence is allowed only in total_time mode.

    Refocusing pulses are ideal and instantaneous.
    """

    def __init__(
        self,
        sequence: DDSequence,
        noise: OrnsteinUhlenbeckNoise,
        tau: float | None = None,
        total_time: float | None = None,
        realizations: int = 100,
        dt: float = 5.0,
        seed: int | None = None,
    ) -> None:

        # =====================================================
        # Timing validation
        # =====================================================

        if (
            tau is None
            and total_time is None
        ):
            raise ValueError(
                "Specify either tau or total_time."
            )

        if (
            tau is not None
            and total_time is not None
        ):
            raise ValueError(
                "Specify tau or total_time, not both."
            )

        if tau is not None:

            if tau <= 0:
                raise ValueError(
                    "tau must be positive."
                )

        if total_time is not None:

            if total_time <= 0:
                raise ValueError(
                    "total_time must be positive."
                )

        # Zero-pulse sequences cannot define T from tau.
        if (
            sequence.number_of_pulses == 0
            and tau is not None
        ):
            raise ValueError(
                "Zero-pulse sequences require total_time."
            )

        if realizations < 1:
            raise ValueError(
                "realizations must be at least 1."
            )

        if dt <= 0:
            raise ValueError(
                "dt must be positive."
            )

        # =====================================================
        # Store parameters
        # =====================================================

        self.sequence = sequence
        self.noise = noise

        self._tau = (
            None
            if tau is None
            else float(tau)
        )

        self._total_time = (
            None
            if total_time is None
            else float(total_time)
        )

        self.realizations = int(
            realizations
        )

        self.dt = float(
            dt
        )

        self.seed = seed

    # =========================================================
    # Timing
    # =========================================================

    @property
    def total_time(self) -> float:
        """
        Total free-evolution time in ns.

        If total_time was supplied directly, return it.

        Otherwise:

            T = N * tau
        """

        if self._total_time is not None:

            return self._total_time

        return (
            self.sequence.number_of_pulses
            * self._tau
        )

    @property
    def tau(self) -> float | None:
        """
        Interpulse spacing in ns.

        Returns None for a zero-pulse free-induction
        sequence.

        Otherwise:

            tau = T / N

        when total_time mode is used.
        """

        number_of_pulses = (
            self.sequence.number_of_pulses
        )

        if number_of_pulses == 0:

            return None

        if self._tau is not None:

            return self._tau

        return (
            self.total_time
            / number_of_pulses
        )

    @property
    def pulse_times(self) -> FloatArray:
        """
        Centers of the ideal pi pulses.

        For N > 0:

            t_k = (k + 1/2) tau

        For N = 0:

            returns an empty array.
        """

        number_of_pulses = (
            self.sequence.number_of_pulses
        )

        if number_of_pulses == 0:

            return np.array(
                [],
                dtype=float,
            )

        return (
            (
                np.arange(
                    number_of_pulses,
                    dtype=float,
                )
                + 0.5
            )
            * self.tau
        )

    # =========================================================
    # Initial state
    # =========================================================

    def _initial_density_matrix(self):
        """
        Prepare |0>, then apply pi/2_X.

        The resulting Bloch vector lies on the equator.
        """

        rho = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=complex,
        )

        return rotate_density_matrix(
            rho,
            "X",
            np.pi / 2.0,
        )

    # =========================================================
    # Noise Hamiltonian
    # =========================================================

    @staticmethod
    def _noise_hamiltonian(
        detuning: float,
    ):
        """
        Classical frequency-noise Hamiltonian.

            H = delta_f / 2 sigma_z

        detuning is in GHz.
        """

        return (
            0.5
            * detuning
            * SIGMA_Z
        )

    # =========================================================
    # Free evolution
    # =========================================================

    def _propagate_interval(
        self,
        rho,
        start_time: float,
        end_time: float,
        noise_times: FloatArray,
        noise_values: FloatArray,
    ):
        """
        Propagate through one free-evolution interval.

        The OU noise is treated as piecewise constant over
        integration steps.
        """

        current_time = (
            start_time
        )

        while (
            current_time
            < end_time - 1e-12
        ):

            step = min(
                self.dt,
                end_time
                - current_time,
            )

            # -------------------------------------------------
            # Find the most recent sampled noise value.
            # -------------------------------------------------

            index = int(
                np.searchsorted(
                    noise_times,
                    current_time,
                    side="right",
                )
                - 1
            )

            index = max(
                index,
                0,
            )

            index = min(
                index,
                len(noise_values) - 1,
            )

            detuning = (
                noise_values[index]
            )

            H = self._noise_hamiltonian(
                detuning
            )

            rho = evolve_density_matrix(
                rho,
                H,
                step,
                decoherence=None,
            )

            current_time += step

        return rho

    # =========================================================
    # One stochastic realization
    # =========================================================

    def _simulate_realization(
        self,
        noise_times: FloatArray,
        noise_values: FloatArray,
    ) -> FloatArray:
        """Simulate one stochastic realization."""

        rho = (
            self._initial_density_matrix()
        )

        current_time = 0.0

        # -----------------------------------------------------
        # DD pulse train
        # -----------------------------------------------------

        for pulse_time, pulse in zip(
            self.pulse_times,
            self.sequence.pulses,
        ):

            # Free evolution up to the pulse.
            rho = self._propagate_interval(
                rho,
                current_time,
                pulse_time,
                noise_times,
                noise_values,
            )

            # Ideal instantaneous pi pulse.
            rho = rotate_density_matrix(
                rho,
                pulse.axis,
                np.pi,
            )

            current_time = (
                pulse_time
            )

        # -----------------------------------------------------
        # Final free interval
        #
        # For zero-pulse free induction this automatically
        # evolves from 0 to total_time.
        # -----------------------------------------------------

        rho = self._propagate_interval(
            rho,
            current_time,
            self.total_time,
            noise_times,
            noise_values,
        )

        return bloch_vector(
            rho
        )

    # =========================================================
    # Ensemble simulation
    # =========================================================

    def simulate(
        self,
    ) -> DynamicalDecouplingResult:
        """Average the DD signal over noise realizations."""

        rng = np.random.default_rng(
            self.seed
        )

        final_vectors = np.zeros(
            (
                self.realizations,
                3,
            ),
            dtype=float,
        )

        for realization in range(
            self.realizations
        ):

            noise_times, noise_values = (
                self.noise.generate(
                    duration=self.total_time,
                    dt=self.dt,
                    rng=rng,
                )
            )

            final_vectors[
                realization
            ] = self._simulate_realization(
                noise_times,
                noise_values,
            )

        # -----------------------------------------------------
        # Ensemble average
        # -----------------------------------------------------

        average = np.mean(
            final_vectors,
            axis=0,
        )

        coherence_x = float(
            average[0]
        )

        coherence_y = float(
            average[1]
        )

        coherence = float(
            np.sqrt(
                coherence_x**2
                + coherence_y**2
            )
        )

        return DynamicalDecouplingResult(
            total_time=(
                self.total_time
            ),
            average_bloch_vector=(
                average
            ),
            coherence=(
                coherence
            ),
            coherence_x=(
                coherence_x
            ),
            coherence_y=(
                coherence_y
            ),
            realizations=(
                self.realizations
            ),
        )