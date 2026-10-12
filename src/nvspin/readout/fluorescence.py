"""Fluorescence readout models for NV centers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optical_cycle import (
    NVOpticalCycle,
)


@dataclass
class FluorescenceReadout:
    """
    Photon-rate model based on the NV optical cycle.

    Parameters
    ----------
    optical_cycle
        NV optical rate-equation model.

    collection_efficiency
        Overall photon collection/detection efficiency.

        This scales photon count rate but does not by itself
        change ideal contrast.

    background_rate
        Spin-independent detected background rate in 1/ns.

        Background reduces observed contrast.
    """

    optical_cycle: NVOpticalCycle

    collection_efficiency: float = 1.0
    background_rate: float = 0.0

    def __post_init__(self) -> None:

        if not (
            0.0
            <= self.collection_efficiency
            <= 1.0
        ):
            raise ValueError(
                "collection_efficiency must lie between 0 and 1."
            )

        if self.background_rate < 0:
            raise ValueError(
                "background_rate must be non-negative."
            )

    def photon_rate(
        self,
        populations,
    ) -> float:
        """
        Return detected fluorescence rate in 1/ns.

        Radiative photons originate from:

            e0 -> g0
            e1 -> g1
        """

        p = np.asarray(
            populations,
            dtype=float,
        )

        if p.shape != (5,):
            raise ValueError(
                "populations must have shape (5,)."
            )

        rates = (
            self.optical_cycle.rates
        )

        emitted_rate = (
            rates.radiative_ms0
            * p[2]
            +
            rates.radiative_ms1
            * p[3]
        )

        return float(
            self.collection_efficiency
            * emitted_rate
            +
            self.background_rate
        )

    def integrated_photons(
        self,
        initial_populations,
        readout_time: float,
        steps: int = 1000,
    ) -> float:
        """
        Integrate detected fluorescence during a laser
        readout window.

        This is particularly useful for pulsed ODMR,
        Rabi, Ramsey, Hahn, and DD readout.
        """

        if readout_time <= 0:
            raise ValueError(
                "readout_time must be positive."
            )

        if steps < 2:
            raise ValueError(
                "steps must be at least 2."
            )

        times = np.linspace(
            0.0,
            readout_time,
            steps,
        )

        photon_rates = np.zeros(
            steps,
            dtype=float,
        )

        for index, time in enumerate(
            times
        ):

            populations = (
                self.optical_cycle.propagate(
                    initial_populations,
                    time,
                )
            )

            photon_rates[index] = (
                self.photon_rate(
                    populations
                )
            )

        return float(
            np.trapezoid(
                photon_rates,
                times,
            )
        )

    @staticmethod
    def contrast(
        bright_signal: float,
        dark_signal: float,
    ) -> float:
        """
        Calculate ODMR/readout contrast.

            C = (bright - dark) / bright
        """

        if bright_signal <= 0:
            raise ValueError(
                "bright_signal must be positive."
            )

        return float(
            (
                bright_signal
                - dark_signal
            )
            / bright_signal
        )