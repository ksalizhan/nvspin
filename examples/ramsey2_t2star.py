"""Ramsey interferometry with phenomenological T2* dephasing."""

import matplotlib.pyplot as plt

from nvspin.experiments import Ramsey


def main():

    # ---------------------------------------------------------
    # Spin-control parameters
    # ---------------------------------------------------------

    transition_frequency = 2.870   # GHz
    microwave_frequency = 2.865    # GHz
    rabi_frequency = 0.010         # GHz = 10 MHz

    # ---------------------------------------------------------
    # Dephasing parameters
    # ---------------------------------------------------------

    T2_star = 1000.0       # ns = 1 us
    #decay_exponent = 2.0   # Gaussian envelope
    decay_exponent = 1.0   # Exponential envelope

    # ---------------------------------------------------------
    # Ramsey experiment
    # ---------------------------------------------------------

    experiment = Ramsey(
        transition_frequency=transition_frequency,
        microwave_frequency=microwave_frequency,
        rabi_frequency=rabi_frequency,
        delay_range=(
            0.0,
            5000.0,
        ),
        points=3000,
        T2_star=T2_star,
        decay_exponent=decay_exponent,
    )

    result = experiment.simulate()

    # ---------------------------------------------------------
    # Envelope
    # ---------------------------------------------------------

    upper_envelope = (
        0.5
        + 0.5
        * result.coherence_envelope
    )

    lower_envelope = (
        0.5
        - 0.5
        * result.coherence_envelope
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    # Ideal coherent reference
    plt.plot(
        result.delays,
        result.coherent_ground_population,
        color="gray",
        linewidth=1.0,
        alpha=0.4,
        label="Ideal Ramsey",
    )

    # Ramsey with T2*
    plt.plot(
        result.delays,
        result.ground_population,
        color="black",
        linewidth=1.4,
        label=r"Ramsey with $T_2^*$",
    )

    # Envelope guides
    plt.plot(
        result.delays,
        upper_envelope,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="Dephasing envelope",
    )

    plt.plot(
        result.delays,
        lower_envelope,
        color="red",
        linestyle="--",
        linewidth=1.0,
    )

    # T2* marker
    plt.axvline(
        T2_star,
        color="blue",
        linestyle=":",
        linewidth=1.2,
        label=r"$T_2^*$",
    )

    plt.xlabel(
        r"Free-precession time $\tau$ (ns)"
    )

    plt.ylabel(
        r"$P_0$"
    )

    plt.title(
        "Ramsey Interferometry with Dephasing\n"
        r"$\Delta=5$ MHz, "
        r"$f_R=10$ MHz, "
        r"$T_2^*=1~\mu$s, "
        r"$p=1$"
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