"""Ideal Ramsey interferometry demonstration."""

import matplotlib.pyplot as plt

from nvspin.experiments import Ramsey


def main():

    # ---------------------------------------------------------
    # Frequencies
    # ---------------------------------------------------------

    transition_frequency = 2.870

    # 5 MHz below resonance
    microwave_frequency = 2.865

    # 10 MHz Rabi frequency
    rabi_frequency = 0.010

    # ---------------------------------------------------------
    # Ramsey experiment
    # ---------------------------------------------------------

    experiment = Ramsey(
        transition_frequency=transition_frequency,
        microwave_frequency=microwave_frequency,
        rabi_frequency=rabi_frequency,
        delay_range=(
            0.0,
            2000.0,
        ),
        points=2000,
    )

    result = experiment.simulate()

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        result.delays,
        result.ground_population,
        color="black",
        linewidth=1.4,
        label=r"$P_0$",
    )

    plt.plot(
        result.delays,
        result.excited_population,
        color="red",
        linestyle="--",
        linewidth=1.1,
        label=r"$P_1$",
    )

    plt.xlabel(
        r"Free-precession time $\tau$ (ns)"
    )

    plt.ylabel(
        "Population"
    )

    plt.title(
        "Ramsey Interferometry\n"
        r"$\Delta=5$ MHz, "
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