"""
Hahn echo with finite-duration, perfectly resonant pulses.

The system accumulates static detuning during free
precession, but the microwave pulses themselves are
perfectly resonant.

This demonstrates that finite pulse duration alone does
not destroy ideal Hahn refocusing.
"""

import matplotlib.pyplot as plt

from nvspin.experiments import HahnEcho


def main():

    # 5 MHz static detuning during free evolution
    free_detuning = 0.005

    # Perfectly resonant microwave pulses
    pulse_detuning = 0.0

    # 10 MHz Rabi frequency
    rabi_frequency = 0.010

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

    print()
    print("Finite resonant Hahn echo")
    print("-------------------------")

    print(
        f"Free detuning: "
        f"{free_detuning * 1000:.3f} MHz"
    )

    print(
        f"Pulse detuning: "
        f"{pulse_detuning * 1000:.3f} MHz"
    )

    print(
        f"pi/2 duration: "
        f"{experiment.pi_over_2_duration:.3f} ns"
    )

    print(
        f"pi duration: "
        f"{experiment.pi_duration:.3f} ns"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        result.total_free_evolution_times,
        result.ground_population,
        color="black",
        linewidth=1.6,
        label="Finite resonant pulses",
    )

    plt.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="Perfect refocusing",
    )

    plt.xlabel(
        r"Total free-evolution time $2\tau$ (ns)"
    )

    plt.ylabel(
        r"$P_0$"
    )

    plt.title(
        "Hahn Echo with Finite Resonant Pulses\n"
        r"$\Delta_{\rm free}=5$ MHz, "
        r"$\Delta_{\rm pulse}=0$, "
        r"$f_R=10$ MHz"
    )

    # Zoom tightly around unity so even small errors
    # would be visible.
    plt.ylim(
        0.98,
        1.002,
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()