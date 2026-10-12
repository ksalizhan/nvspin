"""Hahn-echo simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvspin.control.two_level import (
    rotating_frame_hamiltonian,
)

from nvspin.noise import (
    DecoherenceModel,
)

from nvspin.solvers import (
    evolve_density_matrix,
    population,
    pure_state_density_matrix,
)


FloatArray = NDArray[np.float64]


@dataclass
class HahnEchoResult:
    """Result of a Hahn-echo simulation."""

    delays: FloatArray
    total_free_evolution_times: FloatArray

    ground_population: FloatArray
    excited_population: FloatArray

    @property
    def signal(self) -> FloatArray:
        """Default spin-dependent readout signal."""

        return self.ground_population


class HahnEcho:
    """
    Effective two-level Hahn-echo experiment.

    Sequence
    --------
        pi/2 - tau - pi - tau - pi/2 - readout

    Parameters
    ----------
    free_detuning
        Static frequency detuning accumulated during free
        evolution, in GHz.

    rabi_frequency
        On-resonance Rabi frequency during microwave pulses,
        in GHz.

    pulse_detuning
        Detuning present specifically during microwave pulses,
        in GHz.

        Default = 0 gives perfectly resonant finite pulses.

    delay_range
        Range of each free-evolution interval tau, in ns.

    points
        Number of tau values.

    phase_first
        Phase of the first pi/2 pulse in radians.

    phase_pi
        Phase of the refocusing pi pulse in radians.

    phase_last
        Phase of the final pi/2 pulse in radians.

    decoherence
        Optional Markovian decoherence model.

        If None, evolution is purely unitary.

        If supplied, density-matrix evolution uses the
        Lindblad master-equation backend.

    Notes
    -----
    free_detuning and pulse_detuning are intentionally
    independent.

    This allows nvspin to distinguish:

        - free-precession phase accumulation
        - microwave pulse errors
        - environmental decoherence
    """

    def __init__(
        self,
        free_detuning: float,
        rabi_frequency: float,
        pulse_detuning: float = 0.0,
        delay_range: tuple[float, float] = (
            0.0,
            2000.0,
        ),
        points: int = 1000,
        phase_first: float = 0.0,
        phase_pi: float = 0.0,
        phase_last: float = 0.0,
        decoherence: DecoherenceModel | None = None,
    ) -> None:

        tau_min, tau_max = delay_range

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        if rabi_frequency <= 0:
            raise ValueError(
                "rabi_frequency must be positive."
            )

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

        # -----------------------------------------------------
        # Store parameters
        # -----------------------------------------------------

        self.free_detuning = float(
            free_detuning
        )

        self.pulse_detuning = float(
            pulse_detuning
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

        self.phase_pi = float(
            phase_pi
        )

        self.phase_last = float(
            phase_last
        )

        self.decoherence = decoherence

    # =========================================================
    # Pulse durations
    # =========================================================

    @property
    def pi_over_2_duration(self) -> float:
        """Ideal resonant pi/2 duration in ns."""

        return (
            1.0
            / (
                4.0
                * self.rabi_frequency
            )
        )

    @property
    def pi_duration(self) -> float:
        """Ideal resonant pi duration in ns."""

        return (
            1.0
            / (
                2.0
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
        """
        Hamiltonian during a microwave pulse.

        A zero reference transition frequency is used so
        pulse_detuning can be specified directly.
        """

        return rotating_frame_hamiltonian(
            transition_frequency=(
                self.pulse_detuning
            ),
            microwave_frequency=0.0,
            rabi_frequency=(
                self.rabi_frequency
            ),
            phase=phase,
        )

    def _free_hamiltonian(self):
        """
        Hamiltonian during free evolution.

            H_free = Delta_free / 2 * sigma_z
        """

        return rotating_frame_hamiltonian(
            transition_frequency=(
                self.free_detuning
            ),
            microwave_frequency=0.0,
            rabi_frequency=0.0,
            phase=0.0,
        )

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(
        self,
    ) -> HahnEchoResult:
        """Run the Hahn-echo sequence."""

        delays = np.linspace(
            self.delay_range[0],
            self.delay_range[1],
            self.points,
        )

        total_times = (
            2.0 * delays
        )

        # -----------------------------------------------------
        # Hamiltonians
        # -----------------------------------------------------

        H_first = self._pulse_hamiltonian(
            self.phase_first
        )

        H_pi = self._pulse_hamiltonian(
            self.phase_pi
        )

        H_last = self._pulse_hamiltonian(
            self.phase_last
        )

        H_free = self._free_hamiltonian()

        # -----------------------------------------------------
        # Initial pure state |0>
        # -----------------------------------------------------

        initial_state = np.array(
            [
                1.0,
                0.0,
            ],
            dtype=complex,
        )

        initial_rho = pure_state_density_matrix(
            initial_state
        )

        # -----------------------------------------------------
        # First pi/2 pulse
        # -----------------------------------------------------

        rho_after_first = evolve_density_matrix(
            initial_rho,
            H_first,
            self.pi_over_2_duration,
            decoherence=self.decoherence,
        )

        ground_population = np.zeros(
            self.points,
            dtype=float,
        )

        excited_population = np.zeros(
            self.points,
            dtype=float,
        )

        # -----------------------------------------------------
        # Sweep tau
        # -----------------------------------------------------

        for index, delay in enumerate(
            delays
        ):

            rho = rho_after_first.copy()

            # -------------------------------------------------
            # First free-precession interval
            # -------------------------------------------------

            rho = evolve_density_matrix(
                rho,
                H_free,
                delay,
                decoherence=self.decoherence,
            )

            # -------------------------------------------------
            # Refocusing pi pulse
            # -------------------------------------------------

            rho = evolve_density_matrix(
                rho,
                H_pi,
                self.pi_duration,
                decoherence=self.decoherence,
            )

            # -------------------------------------------------
            # Second free-precession interval
            # -------------------------------------------------

            rho = evolve_density_matrix(
                rho,
                H_free,
                delay,
                decoherence=self.decoherence,
            )

            # -------------------------------------------------
            # Final pi/2 pulse
            # -------------------------------------------------

            rho = evolve_density_matrix(
                rho,
                H_last,
                self.pi_over_2_duration,
                decoherence=self.decoherence,
            )

            # -------------------------------------------------
            # Readout
            # -------------------------------------------------

            ground_population[index] = (
                population(
                    rho,
                    0,
                )
            )

            excited_population[index] = (
                population(
                    rho,
                    1,
                )
            )

        # -----------------------------------------------------
        # Return result
        # -----------------------------------------------------

        return HahnEchoResult(
            delays=delays,
            total_free_evolution_times=(
                total_times
            ),
            ground_population=(
                ground_population
            ),
            excited_population=(
                excited_population
            ),
        )