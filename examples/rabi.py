"""Ideal resonant Rabi oscillation demonstration."""

import matplotlib.pyplot as plt

from nvspin.experiments import Rabi


def main():

    # 10 MHz Rabi frequency
    f_rabi = 0.010

    experiment = Rabi(
        transition_frequency=2.870,
        microwave_frequency=2.870,
        rabi_frequency=f_rabi,
        duration_range=(
            0.0,
            500.0,
        ),
        points=1000,
    )

    result = experiment.simulate()

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        result.durations,
        result.ground_population,
        color="black",
        linewidth=1.5,
        label=r"$P_0$",
    )

    plt.plot(
        result.durations,
        result.excited_population,
        color="red",
        linewidth=1.2,
        linestyle="--",
        label=r"$P_1$",
    )

    # pi/2 and pi calibration points

    plt.axvline(
        25.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label=r"$\pi/2$",
    )

    plt.axvline(
        50.0,
        color="blue",
        linestyle=":",
        linewidth=1.0,
        label=r"$\pi$",
    )

    plt.xlabel(
        "Microwave pulse duration (ns)"
    )

    plt.ylabel(
        "Population"
    )

    plt.title(
        "Ideal Resonant Rabi Oscillations\n"
        r"$f_R=10$ MHz"
    )

    plt.ylim(
        -0.03,
        1.03,
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()