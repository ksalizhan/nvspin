"""Pulse-sequence definitions for nvspin."""

from __future__ import annotations

from dataclasses import dataclass, field

from .pulse import MicrowavePulse


@dataclass(frozen=True)
class FreeEvolution:
    """
    Free-evolution interval.

    Parameters
    ----------
    duration
        Evolution time in ns.
    """

    duration: float

    def __post_init__(self) -> None:

        if self.duration < 0:
            raise ValueError(
                "Free-evolution duration must be non-negative."
            )


SequenceElement = MicrowavePulse | FreeEvolution


@dataclass
class PulseSequence:
    """Ordered collection of pulses and free evolution."""

    elements: list[SequenceElement] = field(
        default_factory=list
    )

    def add_pulse(
        self,
        pulse: MicrowavePulse,
    ) -> None:
        """Append a microwave pulse."""

        self.elements.append(
            pulse
        )

    def add_free_evolution(
        self,
        duration: float,
    ) -> None:
        """Append a free-evolution interval."""

        self.elements.append(
            FreeEvolution(
                duration=duration
            )
        )

    @property
    def total_duration(self) -> float:
        """Return total sequence duration in ns."""

        return float(
            sum(
                element.duration
                for element
                in self.elements
            )
        )

    def clear(self) -> None:
        """Remove all sequence elements."""

        self.elements.clear()