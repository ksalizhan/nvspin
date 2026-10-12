"""
Hahn echo with Lindblad relaxation and pure dephasing.

This example compares:

    1. Ideal coherent Hahn echo
    2. Hahn echo with Markovian decoherence

Decoherence channels:
    - T1 relaxation
    - T_phi pure dephasing

The effective transverse coherence time satisfies:

    1/T2 = 1/(2*T1) + 1/T_phi
"""

import matplotlib.pyplot as plt

from nvspin.experiments import HahnEcho
from nvspin.noise import DecoherenceModel


def main():

    # ========================================================
    # Decoherence model
    # ========================================================

    noise = DecoherenceModel(
        T1=10000.0,       # ns = 10 us
        T_phi=2000.0,     # ns = 2 us
    )

    print()
    print("Decoherence model")
    print("-----------------")

    print(
        f"T1    = {noise.T1:.1f} ns"
    )

    print(
        f"T_phi = {noise.T_phi:.1f} ns"
    )

    print(
        f"T2    = {noise.T2:.1f} ns"
    )

    # ========================================================
    # Ideal Hahn echo
    # ========================================================

    ideal = HahnEcho(
        free_detuning=0.005,      # 5 MHz
        pulse_detuning=0.0,
        rabi_frequency=0.010,     # 10 MHz
        delay_range=(
            0.0,
            5000.0,
        ),
        points=1000,
        decoherence=None,
    ).simulate()

    # ========================================================
    # Hahn echo with Lindblad decoherence
    # ========================================================

    decohering = HahnEcho(
        free_detuning=0.005,
        pulse_detuning=0.0,
        rabi_frequency=0.010,
        delay_range=(
            0.0,
            5000.0,
        ),
        points=1000,
        decoherence=noise,
    ).simulate()

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ideal.total_free_evolution_times,
        ideal.ground_population,
        color="gray",
        linewidth=1.2,
        label="Ideal Hahn",
    )

    plt.plot(
        decohering.total_free_evolution_times,
        decohering.ground_population,
        color="black",
        linewidth=1.5,
        label="Hahn + Lindblad decoherence",
    )

    # --------------------------------------------------------
    # T2 reference
    # --------------------------------------------------------

    plt.axvline(
        noise.T2,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=r"$T_2$",
    )

    # ========================================================
    # Labels
    # ========================================================

    plt.xlabel(
        r"Total free-evolution time $2\tau$ (ns)"
    )

    plt.ylabel(
        r"$P_0$"
    )

    plt.title(
        "Hahn Echo with Markovian Decoherence\n"
        r"$T_1=10~\mu$s, "
        r"$T_\phi=2~\mu$s"
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