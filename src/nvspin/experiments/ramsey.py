"""Ramsey interferometry simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvspin.control.two_level import (
    rotating_frame_hamiltonian,
)
from nvspin.solvers import propagate_state


FloatArray = NDArray[np.float64]


@dataclass
class RamseyResult:
    """Result of a Ramsey simulation."""

    delays: FloatArray

    # Final populations after optional dephasing
    ground_population: FloatArray
    excited_population: FloatArray

    # Ideal coherent populations before dephasing
    coherent_ground_population: FloatArray
    coherent_excited_population: FloatArray

    # Phenomenological coherence envelope
    coherence_envelope: FloatArray

    @property
    def signal(self) -> FloatArray:
        """Default spin-dependent readout signal."""

        return self.ground_population


class Ramsey:
    """
    Effective two-level Ramsey experiment.

    Sequence
    --------

        pi/2_X - tau - pi/2_X - readout

    Parameters
    ----------
    transition_frequency
        Spin-transition frequency f0 in GHz.

    microwave_frequency
        Microwave carrier frequency in GHz.

    rabi_frequency
        Rabi frequency during the pi/2 pulses in GHz.

    delay_range
        Ramsey free-precession range in ns.

    points
        Number of delay points.

    phase_first
        Phase of the first pi/2 pulse in radians.

    phase_second
        Phase of the second pi/2 pulse in radians.

    T2_star
        Optional phenomenological Ramsey dephasing time
        in ns.

        If None, no phenomenological dephasing is applied.

    decay_exponent
        Stretch exponent p in

            exp[-(tau / T2_star)^p]

        Common choices:

            p = 1  -> exponential decay
            p = 2  -> Gaussian decay
    """

    def __init__(
        self,
        transition_frequency: float,
        microwave_frequency: float,
        rabi_frequency: float,
        delay_range: tuple[float, float] = (
            0.0,
            2000.0,
        ),
        points: int = 1000,
        phase_first: float = 0.0,
        phase_second: float = 0.0,
        T2_star: float | None = None,
        decay_exponent: float = 1.0,
    ) -> None:

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        if transition_frequency < 0:
            raise ValueError(
                "transition_frequency must be non-negative."
            )

        if microwave_frequency < 0:
            raise ValueError(
                "microwave_frequency must be non-negative."
            )

        if rabi_frequency <= 0:
            raise ValueError(
                "rabi_frequency must be positive."
            )

        tau_min, tau_max = delay_range

        if tau_min < 0:
            raise ValueError(
                "Minimum delay must be non-negative."
            )

        if tau_max <= tau_min:
            raise ValueError(
                "delay_range must satisfy tau_max > tau_min."
            )

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        if T2_star is not None:
            if T2_star <= 0:
                raise ValueError(
                    "T2_star must be positive."
                )

        if decay_exponent <= 0:
            raise ValueError(
                "decay_exponent must be positive."
            )

        # -----------------------------------------------------
        # Store experiment parameters
        # -----------------------------------------------------

        self.transition_frequency = float(
            transition_frequency
        )

        self.microwave_frequency = float(
            microwave_frequency
        )

        self.rabi_frequency = float(
            rabi_frequency
        )

        self.delay_range = (
            float(tau_min),
            float(tau_max),
        )

        self.points = int(
            points
        )

        self.phase_first = float(
            phase_first
        )

        self.phase_second = float(
            phase_second
        )

        self.T2_star = (
            None
            if T2_star is None
            else float(T2_star)
        )

        self.decay_exponent = float(
            decay_exponent
        )

    # =========================================================
    # Experiment properties
    # =========================================================

    @property
    def detuning(self) -> float:
        """Return f0 - fMW in GHz."""

        return (
            self.transition_frequency
            - self.microwave_frequency
        )

    @property
    def pi_over_2_duration(self) -> float:
        """Return ideal pi/2 duration in ns."""

        return (
            1.0
            / (
                4.0
                * self.rabi_frequency
            )
        )

    # =========================================================
    # Hamiltonians
    # =========================================================

    def _pulse_hamiltonian(
        self,
        phase: float,
    ):
        """Hamiltonian during a microwave pulse."""

        return rotating_frame_hamiltonian(
            transition_frequency=(
                self.transition_frequency
            ),
            microwave_frequency=(
                self.microwave_frequency
            ),
            rabi_frequency=(
                self.rabi_frequency
            ),
            phase=phase,
        )

    def _free_hamiltonian(self):
        """
        Rotating-frame Hamiltonian during free evolution.

        With the microwave switched off:

            H_free = Delta/2 sigma_z
        """

        return rotating_frame_hamiltonian(
            transition_frequency=(
                self.transition_frequency
            ),
            microwave_frequency=(
                self.microwave_frequency
            ),
            rabi_frequency=0.0,
            phase=0.0,
        )

    # =========================================================
    # Phenomenological dephasing
    # =========================================================

    def _coherence_envelope(
        self,
        delays: FloatArray,
    ) -> FloatArray:
        """
        Return the Ramsey coherence envelope.

        Without phenomenological dephasing:

            C(tau) = 1

        With T2*:

            C(tau)
            =
            exp[-(tau / T2*)^p]
        """

        if self.T2_star is None:

            return np.ones_like(
                delays,
                dtype=float,
            )

        return np.exp(
            -(
                delays
                / self.T2_star
            ) ** self.decay_exponent
        )

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(
        self,
    ) -> RamseyResult:
        """Run the Ramsey sequence."""

        delays = np.linspace(
            self.delay_range[0],
            self.delay_range[1],
            self.points,
        )

        # -----------------------------------------------------
        # Hamiltonians
        # -----------------------------------------------------

        H_first = self._pulse_hamiltonian(
            self.phase_first
        )

        H_second = self._pulse_hamiltonian(
            self.phase_second
        )

        H_free = self._free_hamiltonian()

        # -----------------------------------------------------
        # Initial state
        #
        # |0>
        # -----------------------------------------------------

        initial_state = np.array(
            [
                1.0,
                0.0,
            ],
            dtype=complex,
        )

        # -----------------------------------------------------
        # First pi/2 pulse
        # -----------------------------------------------------

        state_after_first = propagate_state(
            initial_state,
            H_first,
            self.pi_over_2_duration,
        )

        # -----------------------------------------------------
        # Arrays for ideal coherent evolution
        # -----------------------------------------------------

        coherent_ground = np.zeros(
            self.points,
            dtype=float,
        )

        coherent_excited = np.zeros(
            self.points,
            dtype=float,
        )

        # -----------------------------------------------------
        # Ideal Ramsey evolution
        # -----------------------------------------------------

        for index, delay in enumerate(
            delays
        ):

            # Free precession
            state = propagate_state(
                state_after_first,
                H_free,
                delay,
            )

            # Second pi/2 pulse
            state = propagate_state(
                state,
                H_second,
                self.pi_over_2_duration,
            )

            populations = (
                np.abs(state) ** 2
            )

            coherent_ground[index] = (
                populations[0]
            )

            coherent_excited[index] = (
                populations[1]
            )

        # =====================================================
        # T2* phenomenological dephasing
        # =====================================================

        envelope = self._coherence_envelope(
            delays
        )

        # Dephase the Ramsey contrast around 1/2.
        #
        # This is deliberately NOT:
        #
        #     P0 = P0 * envelope
        #
        # because pure dephasing destroys coherence rather
        # than population normalization.
        #
        # At long times:
        #
        #     P0 -> 1/2
        #     P1 -> 1/2

        ground_population = (
            0.5
            +
            (
                coherent_ground
                - 0.5
            )
            * envelope
        )

        excited_population = (
            1.0
            - ground_population
        )

        # =====================================================
        # Return result
        # =====================================================

        return RamseyResult(
            delays=delays,

            ground_population=(
                ground_population
            ),

            excited_population=(
                excited_population
            ),

            coherent_ground_population=(
                coherent_ground
            ),

            coherent_excited_population=(
                coherent_excited
            ),

            coherence_envelope=(
                envelope
            ),
        )