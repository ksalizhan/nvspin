"""Demonstrate correlated Ornstein-Uhlenbeck frequency noise."""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


def main():

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.001,             # 1 MHz
        correlation_time=1000.0, # 1 us
    )

    times, values = noise.generate(
        duration=10000.0,
        dt=10.0,
        rng=np.random.default_rng(7),
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        times,
        values * 1000.0,
        color="black",
        linewidth=1.0,
    )

    plt.axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
    )

    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        r"$\delta f(t)$ (MHz)"
    )

    plt.title(
        "Correlated Frequency Noise\n"
        "Ornstein-Uhlenbeck Process"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()