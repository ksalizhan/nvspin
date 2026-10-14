"""
Single-13C DD spectroscopy with an unresolved OU bath.

The resolved 13C is treated as an explicit quantum spin.

The unresolved background environment is represented by
classical correlated longitudinal frequency noise.

The two contributions remain physically distinct:

    explicit 13C
        -> coherent resonance structure

    OU bath
        -> smooth DD coherence envelope
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.experiments import (
    SingleCarbon13,
    SingleCarbonNoisyDDSpectroscopy,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


# ============================================================
# Parameters
# ============================================================

F_LARMOR = 0.0004282

A_PARALLEL = 0.000040
A_PERPENDICULAR = 0.000030

NUMBER_OF_PULSES = 32

SPACING_RANGE = (
    1000.0,
    1230.0,
)

POINTS = 250


# OU background bath

NOISE_SIGMA = 0.00010

NOISE_CORRELATION_TIME = (
    1000.0
)

# REALIZATIONS = 200
# DT = 10.0

# SEED = 12345


def predicted_resonance():

    return (
        1.0
        / (
            2.0 * F_LARMOR
            + A_PARALLEL
        )
    )


def main():

    carbon = SingleCarbon13(
        larmor_frequency=F_LARMOR,
        A_parallel=A_PARALLEL,
        A_perpendicular=A_PERPENDICULAR,
    )

    noise = OrnsteinUhlenbeckNoise(
        sigma=NOISE_SIGMA,
        correlation_time=(
            NOISE_CORRELATION_TIME
        ),
    )

    experiment = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=(
                NUMBER_OF_PULSES
            ),
            spacing_range=(
                SPACING_RANGE
            ),
            noise=noise,
            points=POINTS,
        )
    )

    result = experiment.simulate()

    resonance = (
        predicted_resonance()
    )

    # ========================================================
    # Print information
    # ========================================================

    print()
    print(
        "Single-13C spectroscopy with OU bath"
    )

    print(
        "------------------------------------"
    )

    print(
        f"N = {NUMBER_OF_PULSES}"
    )

    print(
        f"13C f_L = "
        f"{F_LARMOR * 1000:.4f} MHz"
    )

    print(
        f"A_parallel = "
        f"{A_PARALLEL * 1e6:.1f} kHz"
    )

    print(
        f"A_perpendicular = "
        f"{A_PERPENDICULAR * 1e6:.1f} kHz"
    )

    print(
        f"OU sigma = "
        f"{NOISE_SIGMA * 1000:.3f} MHz"
    )

    print(
        f"OU correlation time = "
        f"{NOISE_CORRELATION_TIME / 1000:.3f} us"
    )

    print(
        f"Predicted resonance = "
        f"{resonance / 1000:.6f} us"
    )

    # ========================================================
    # Figure 1
    #
    # Separate coherence contributions
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        result.pulse_spacings
        / 1000.0,
        result.carbon_coherence,
        linewidth=1.5,
        label=r"Explicit $^{13}$C",
    )

    plt.plot(
        result.pulse_spacings
        / 1000.0,
        result.bath_coherence,
        linewidth=1.5,
        label="OU bath",
    )

    plt.plot(
        result.pulse_spacings
        / 1000.0,
        result.total_coherence,
        color="black",
        linewidth=2.0,
        label="Combined coherence",
    )

    plt.axvline(
        resonance / 1000.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=(
            r"$1/(2f_L+A_\parallel)$"
        ),
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        "Electron coherence"
    )

    plt.title(
        r"Resolved $^{13}$C + "
        "Correlated Background Bath"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$"
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 2
    #
    # Experimental-style P_x signal
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        result.pulse_spacings
        / 1000.0,
        result.preservation_probability,
        color="black",
        linewidth=1.7,
    )

    plt.axvline(
        resonance / 1000.0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="Predicted 13C resonance",
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        r"Single-$^{13}$C DD Spectroscopy "
        "with OU Background"
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