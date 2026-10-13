"""Classical stochastic noise models for nvspin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck frequency noise.

    Parameters
    ----------
    sigma
        Stationary standard deviation of the frequency
        fluctuations in GHz.

    correlation_time
        Noise correlation time in ns.

    Notes
    -----
    The process obeys

        dX = -(X/tau_c) dt
             + sqrt(2 sigma^2 / tau_c) dW

    and has stationary correlation

        <X(t) X(0)>
        = sigma^2 exp(-|t| / tau_c).

    The returned values therefore have units of GHz and
    can be used as qubit detunings.
    """

    sigma: float
    correlation_time: float

    def __post_init__(self) -> None:

        if self.sigma < 0:
            raise ValueError(
                "sigma must be non-negative."
            )

        if self.correlation_time <= 0:
            raise ValueError(
                "correlation_time must be positive."
            )

    def generate(
        self,
        duration: float,
        dt: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Generate one stationary OU noise realization.

        Returns
        -------
        times
            Time samples in ns.

        values
            Frequency fluctuations in GHz.
        """

        if duration <= 0:
            raise ValueError(
                "duration must be positive."
            )

        if dt <= 0:
            raise ValueError(
                "dt must be positive."
            )

        if rng is None:
            rng = np.random.default_rng()

        number_of_steps = (
            int(
                np.ceil(
                    duration / dt
                )
            )
            + 1
        )

        times = (
            np.arange(
                number_of_steps,
                dtype=float,
            )
            * dt
        )

        values = np.zeros(
            number_of_steps,
            dtype=float,
        )

        # Draw initial value from the stationary distribution.
        values[0] = rng.normal(
            loc=0.0,
            scale=self.sigma,
        )

        # Exact discrete-time OU update.
        decay = np.exp(
            -dt
            / self.correlation_time
        )

        innovation_sigma = (
            self.sigma
            * np.sqrt(
                1.0
                - decay**2
            )
        )

        for index in range(
            1,
            number_of_steps,
        ):

            values[index] = (
                decay
                * values[index - 1]
                +
                innovation_sigma
                * rng.normal()
            )

        return (
            times,
            values,
        )