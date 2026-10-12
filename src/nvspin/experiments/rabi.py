"""Rabi oscillation simulations."""

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
class RabiResult:
    """Result of a Rabi simulation."""

    durations: FloatArray
    ground_population: FloatArray
    excited_population: FloatArray

    @property
    def signal(self) -> FloatArray:
        """
        Default spin-dependent readout signal.

        For an ideal NV two-level model this is simply the
        population remaining in the initialized state.
        """

        return self.ground_population


class Rabi:
    """
    Effective two-level Rabi experiment.

    Parameters
    ----------
    transition_frequency
        Resonance frequency f0 in GHz.

    microwave_frequency
        Applied microwave frequency in GHz.

        If None, the drive is resonant with the transition.

    rabi_frequency
        On-resonance Rabi frequency in GHz.

    duration_range
        Minimum and maximum microwave-pulse duration in ns.

    points
        Number of pulse-duration points.

    phase
        Microwave phase in radians.
    """

    def __init__(
        self,
        transition_frequency: float,
        rabi_frequency: float,
        microwave_frequency: float | None = None,
        duration_range: tuple[
            float,
            float,
        ] = (
            0.0,
            500.0,
        ),
        points: int = 1000,
        phase: float = 0.0,
    ) -> None:

        if transition_frequency < 0:
            raise ValueError(
                "transition_frequency must be non-negative."
            )

        if rabi_frequency <= 0:
            raise ValueError(
                "rabi_frequency must be positive."
            )

        t_min, t_max = (
            duration_range
        )

        if t_min < 0:
            raise ValueError(
                "Minimum duration must be non-negative."
            )

        if t_max <= t_min:
            raise ValueError(
                "duration_range must satisfy t_max > t_min."
            )

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        if microwave_frequency is None:
            microwave_frequency = (
                transition_frequency
            )

        self.transition_frequency = float(
            transition_frequency
        )

        self.microwave_frequency = float(
            microwave_frequency
        )

        self.rabi_frequency = float(
            rabi_frequency
        )

        self.duration_range = (
            float(t_min),
            float(t_max),
        )

        self.points = int(
            points
        )

        self.phase = float(
            phase
        )

    def simulate(
        self,
    ) -> RabiResult:
        """Simulate Rabi oscillations."""

        durations = np.linspace(
            self.duration_range[0],
            self.duration_range[1],
            self.points,
        )

        H_rot = (
            rotating_frame_hamiltonian(
                transition_frequency=(
                    self.transition_frequency
                ),
                microwave_frequency=(
                    self.microwave_frequency
                ),
                rabi_frequency=(
                    self.rabi_frequency
                ),
                phase=self.phase,
            )
        )

        initial_state = np.array(
            [
                1.0,
                0.0,
            ],
            dtype=complex,
        )

        ground_population = np.zeros(
            self.points,
            dtype=float,
        )

        excited_population = np.zeros(
            self.points,
            dtype=float,
        )

        for index, duration in enumerate(
            durations
        ):

            final_state = propagate_state(
                initial_state,
                H_rot,
                duration,
            )

            populations = (
                np.abs(final_state) ** 2
            )

            ground_population[index] = (
                populations[0]
            )

            excited_population[index] = (
                populations[1]
            )

        return RabiResult(
            durations=durations,
            ground_population=(
                ground_population
            ),
            excited_population=(
                excited_population
            ),
        )