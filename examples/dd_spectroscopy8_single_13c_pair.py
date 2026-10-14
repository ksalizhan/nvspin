"""
DD spectroscopy of one interacting 13C-13C nuclear pair.

The pair is represented using the high-field pseudospin
model

    H0 = X I_x
    H1 = X I_x + Z I_z

where

    X = nuclear-nuclear dipolar flip-flop coupling
    Z = NV-induced hyperfine-field gradient.

This is fundamentally different from the isolated-carbon
model because X represents a physical interaction between
two nuclear spins.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.physics import (
    Carbon13Pair,
    subsystem_coherence,
)


# ============================================================
# Pair parameters
# ============================================================
#
# Representative published parameters for Abobeih Pair 1:
#
#     X / 2pi ~ 0.2367 kHz
#     Z / 2pi ~ 7.890 kHz
#
# nvspin stores ordinary frequencies in GHz.
#
#     1 kHz = 1e-6 GHz
#
# Therefore:
#
#     X = 0.2367 kHz
#     Z = 7.890 kHz
#
# become:
#
#     0.2367e-6 GHz
#     7.890e-6 GHz
#
# ============================================================

PAIR = Carbon13Pair(
    label="Pair 1",
    X=0.2367e-6,
    Z=7.890e-6,
)


# ============================================================
# DD parameters
# ============================================================

NUMBER_OF_PULSES = 16

SPACING_MIN = 1000.0
SPACING_MAX = 300000.0

POINTS = 5000


def main():

    pulse_spacings = np.linspace(
        SPACING_MIN,
        SPACING_MAX,
        POINTS,
    )

    coherence = np.zeros(
        POINTS,
        dtype=complex,
    )

    for index, spacing in enumerate(
        pulse_spacings
    ):

        coherence[index] = (
            subsystem_coherence(
                subsystem=PAIR,
                number_of_pulses=(
                    NUMBER_OF_PULSES
                ),
                pulse_spacing=(
                    float(spacing)
                ),
            )
        )

    preservation_probability = (
        0.5
        * (
            1.0
            + np.real(
                coherence
            )
        )
    )

    print()
    print(
        "Single 13C-13C pair DD spectroscopy"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Pair: {PAIR.label}"
    )

    print(
        f"X = "
        f"{PAIR.X * 1e6:.4f} kHz"
    )

    print(
        f"Z = "
        f"{PAIR.Z * 1e6:.4f} kHz"
    )

    print(
        f"f0 = "
        f"{PAIR.frequency_0 * 1e6:.4f} kHz"
    )

    print(
        f"f1 = "
        f"{PAIR.frequency_1 * 1e6:.4f} kHz"
    )

    # ========================================================
    # Find strongest response
    # ========================================================

    minimum_index = np.argmin(
        preservation_probability
    )

    strongest_spacing = (
        pulse_spacings[
            minimum_index
        ]
    )

    minimum_probability = (
        preservation_probability[
            minimum_index
        ]
    )

    print(
        f"Strongest simulated response: "
        f"{strongest_spacing / 1000:.3f} us"
    )

    print(
        f"Minimum Px: "
        f"{minimum_probability:.5f}"
    )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        pulse_spacings
        / 1000.0,
        preservation_probability,
        color="black",
        linewidth=1.2,
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        r"DD Spectroscopy of One "
        r"$^{13}$C--$^{13}$C Pair"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$, "
        rf"$X/2\pi\rightarrow "
        f"{PAIR.X * 1e6:.4f}"
        r"\,\mathrm{kHz}$, "
        rf"$Z/2\pi\rightarrow "
        f"{PAIR.Z * 1e6:.3f}"
        r"\,\mathrm{kHz}$"
    )

    plt.ylim(
        -0.03,
        1.03,
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()