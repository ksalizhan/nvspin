"""
Hahn echo with finite-duration microwave pulses and pulse detuning.

This example separates two different effects:

    free_detuning
        Static detuning accumulated during free evolution.

    pulse_detuning
        Detuning present during the microwave pulses.

The purpose is to demonstrate pulse-error-induced structure
that remains even though an ideal Hahn echo would refocus
the static free-evolution detuning.

Compare with:

    hahn1_ideal.py
        Instantaneous ideal pulses.

    hahn2_finite_resonant.py
        Finite-duration but perfectly resonant pulses.
"""

import matplotlib.pyplot as plt

from nvspin.experiments import HahnEcho


def main():

    # ========================================================
    # Parameters
    # ========================================================

    # Static detuning during free evolution
    free_detuning = 0.005       # GHz = 5 MHz

    # Detuning during microwave pulses
    pulse_detuning = 0.005      # GHz = 5 MHz

    # On-resonance Rabi frequency
    rabi_frequency = 0.010      # GHz = 10 MHz

    # ========================================================
    # Hahn experiment
    # ========================================================

    experiment = HahnEcho(
        free_detuning=free_detuning,
        pulse_detuning=pulse_detuning,
        rabi_frequency=rabi_frequency,
        delay_range=(
            0.0,
            2000.0,
        ),
        points=2000,
    )

    result = experiment.simulate()

    # ========================================================
    # Print experiment information
    # ========================================================

    print()
    print("Hahn echo with pulse detuning")
    print("-----------------------------")

    print(
        f"Free detuning: "
        f"{free_detuning * 1000:.3f} MHz"
    )

    print(
        f"Pulse detuning: "
        f"{pulse_detuning * 1000:.3f} MHz"
    )

    print(
        f"Rabi frequency: "
        f"{rabi_frequency * 1000:.3f} MHz"
    )

    print(
        f"pi/2 duration: "
        f"{experiment.pi_over_2_duration:.3f} ns"
    )

    print(
        f"pi duration: "
        f"{experiment.pi_duration:.3f} ns"
    )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        result.total_free_evolution_times,
        result.ground_population,
        color="black",
        linewidth=1.5,
        label="Hahn echo with pulse detuning",
    )

    # Perfect Hahn reference
    plt.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="Perfect refocusing",
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
        "Hahn Echo with Pulse Detuning\n"
        r"$\Delta_{\rm free}=5$ MHz, "
        r"$\Delta_{\rm pulse}=5$ MHz, "
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