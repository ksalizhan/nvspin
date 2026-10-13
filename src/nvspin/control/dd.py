"""Dynamical-decoupling pulse-sequence definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DDPulse:
    """One ideal DD pi pulse."""

    phase: float

    @property
    def axis(self) -> str:
        """Return conventional pulse-axis label."""

        phase = np.mod(
            self.phase,
            2.0 * np.pi,
        )

        if np.isclose(phase, 0.0):
            return "X"

        if np.isclose(
            phase,
            np.pi / 2.0,
        ):
            return "Y"

        return f"{np.rad2deg(phase):.1f} deg"


@dataclass(frozen=True)
class DDSequence:
    """Description of a dynamical-decoupling sequence."""

    name: str
    pulses: tuple[DDPulse, ...]

    @property
    def number_of_pulses(self) -> int:
        return len(self.pulses)

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(
            pulse.axis
            for pulse in self.pulses
        )


def pdd(
    n_pulses: int,
) -> DDSequence:
    """
    Periodic dynamical decoupling.

    Uses repeated X pi pulses:

        X X X X ...
    """

    if n_pulses < 1:
        raise ValueError(
            "n_pulses must be at least 1."
        )

    pulses = tuple(
        DDPulse(
            phase=0.0
        )
        for _ in range(n_pulses)
    )

    return DDSequence(
        name=f"PDD-{n_pulses}",
        pulses=pulses,
    )


def cpmg(
    n_pulses: int,
) -> DDSequence:
    """
    CPMG refocusing train.

    In this convention the refocusing pulses are Y pulses:

        Y Y Y Y ...

    The preparation/readout pi/2 pulses are not included in
    this object.
    """

    if n_pulses < 1:
        raise ValueError(
            "n_pulses must be at least 1."
        )

    pulses = tuple(
        DDPulse(
            phase=np.pi / 2.0
        )
        for _ in range(n_pulses)
    )

    return DDSequence(
        name=f"CPMG-{n_pulses}",
        pulses=pulses,
    )


def xy8(
    blocks: int = 1,
) -> DDSequence:
    """
    XY8 dynamical-decoupling sequence.

    One block is:

        X Y X Y Y X Y X
    """

    if blocks < 1:
        raise ValueError(
            "blocks must be at least 1."
        )

    X = 0.0
    Y = np.pi / 2.0

    block = (
        X,
        Y,
        X,
        Y,
        Y,
        X,
        Y,
        X,
    )

    phases = (
        block * blocks
    )

    pulses = tuple(
        DDPulse(
            phase=phase
        )
        for phase in phases
    )

    return DDSequence(
        name=f"XY8-{blocks}",
        pulses=pulses,
    )

def free_induction() -> DDSequence:
    """
    Free-induction evolution with no refocusing pulses.
    """

    return DDSequence(
        name="Free induction",
        pulses=(),
    )

def free_induction() -> DDSequence:
    """
    Free-induction evolution with no refocusing pulses.

    This is the zero-pulse reference used for measuring
    dephasing without dynamical decoupling.
    """

    return DDSequence(
        name="Free induction",
        pulses=(),
    )