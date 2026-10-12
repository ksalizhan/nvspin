"""Microwave pulse definitions for nvspin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicrowavePulse:
    """
    Rectangular microwave pulse.

    Parameters
    ----------
    duration
        Pulse duration in ns.

    frequency
        Microwave carrier frequency in GHz.

    rabi_frequency
        On-resonance Rabi frequency in GHz (cycles/ns).

        For this convention:

            t_pi = 1 / (2 * rabi_frequency)

            t_pi_over_2 = 1 / (4 * rabi_frequency)

    phase
        Microwave phase in radians.

        phase = 0       -> X pulse
        phase = pi/2    -> Y pulse

    Notes
    -----
    The pulse object describes control parameters only.
    It does not construct a Hamiltonian by itself.
    """

    duration: float
    frequency: float
    rabi_frequency: float
    phase: float = 0.0

    def __post_init__(self) -> None:

        if self.duration < 0:
            raise ValueError(
                "Pulse duration must be non-negative."
            )

        if self.frequency < 0:
            raise ValueError(
                "Microwave frequency must be non-negative."
            )

        if self.rabi_frequency < 0:
            raise ValueError(
                "Rabi frequency must be non-negative."
            )

    @property
    def phase_degrees(self) -> float:
        """Return pulse phase in degrees."""

        return float(
            np.rad2deg(self.phase)
        )


def pi_pulse(
    frequency: float,
    rabi_frequency: float,
    phase: float = 0.0,
) -> MicrowavePulse:
    """
    Construct a resonant pi pulse.

    With rabi_frequency in GHz and duration in ns:

        t_pi = 1 / (2 f_R)
    """

    if rabi_frequency <= 0:
        raise ValueError(
            "rabi_frequency must be positive."
        )

    duration = (
        1.0
        / (2.0 * rabi_frequency)
    )

    return MicrowavePulse(
        duration=duration,
        frequency=frequency,
        rabi_frequency=rabi_frequency,
        phase=phase,
    )


def pi_over_2_pulse(
    frequency: float,
    rabi_frequency: float,
    phase: float = 0.0,
) -> MicrowavePulse:
    """
    Construct a resonant pi/2 pulse.

    With rabi_frequency in GHz and duration in ns:

        t_pi/2 = 1 / (4 f_R)
    """

    if rabi_frequency <= 0:
        raise ValueError(
            "rabi_frequency must be positive."
        )

    duration = (
        1.0
        / (4.0 * rabi_frequency)
    )

    return MicrowavePulse(
        duration=duration,
        frequency=frequency,
        rabi_frequency=rabi_frequency,
        phase=phase,
    )