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

    def coherence_under_dd(
        self,
        pulse_times: np.ndarray,
        total_time: float,
    ) -> float:
        """
        Exact ensemble coherence for OU longitudinal noise
        under ideal instantaneous DD pulses.

        The switching function y(t) changes sign at each
        pulse time.

        For zero-mean Gaussian noise,

            L = exp[-Var(phi) / 2]

        where

            phi = 2*pi integral y(t) delta_f(t) dt.

        The OU correlation function is

            C(t - t')
                = sigma^2
                  exp(-|t - t'| / correlation_time).
        """

        if total_time < 0:
            raise ValueError(
                "total_time must be non-negative."
            )

        if total_time == 0:
            return 1.0

        pulse_times = np.asarray(
            pulse_times,
            dtype=float,
        )

        if np.any(
            pulse_times <= 0.0
        ):
            raise ValueError(
                "pulse_times must be positive."
            )

        if np.any(
            pulse_times >= total_time
        ):
            raise ValueError(
                "pulse_times must lie inside total_time."
            )

        if np.any(
            np.diff(pulse_times) <= 0.0
        ):
            raise ValueError(
                "pulse_times must be strictly increasing."
            )

        boundaries = np.concatenate(
            (
                [0.0],
                pulse_times,
                [total_time],
            )
        )

        tau_c = (
            self.correlation_time
        )

        # q(t) =
        #
        # integral_0^t
        # y(s) exp[-(t-s)/tau_c] ds

        q = 0.0

        # Accumulates:
        #
        # integral_0^T y(t) q(t) dt

        integral = 0.0

        # Before the first pi pulse:
        #
        # y = +1

        y = 1.0

        for index in range(
            len(boundaries) - 1
        ):

            delta = (
                boundaries[index + 1]
                - boundaries[index]
            )

            decay = np.exp(
                -delta / tau_c
            )

            # Exact integral of y*q(t)
            # across this interval.

            integral += (
                y
                * q
                * tau_c
                * (
                    1.0 - decay
                )
                +
                tau_c
                * delta
                -
                tau_c**2
                * (
                    1.0 - decay
                )
            )

            # Exact update of q.

            q = (
                q
                * decay
                +
                y
                * tau_c
                * (
                    1.0 - decay
                )
            )

            # Ideal pi pulse reverses the
            # longitudinal coupling sign.

            y *= -1.0

        # Full symmetric double integral:
        #
        # Var(phi)
        #
        # = (2*pi)^2 sigma^2
        #
        #   * integral_0^T integral_0^T
        #
        #       y(t)y(t')
        #       exp(-|t-t'|/tau_c)
        #
        #       dt dt'

        phase_variance = (
            (2.0 * np.pi) ** 2
            * self.sigma**2
            * 2.0
            * integral
        )

        # Numerical roundoff can occasionally produce
        # a tiny negative value.

        phase_variance = max(
            float(phase_variance),
            0.0,
        )

        return float(
            np.exp(
                -0.5
                * phase_variance
            )
        )