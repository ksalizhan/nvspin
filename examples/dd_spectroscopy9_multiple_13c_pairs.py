"""
DD spectroscopy of six interacting 13C-13C nuclear pairs.

Each pair is represented using the high-field pseudospin
model

    H0 = X I_x
    H1 = X I_x + Z I_z

where

    X = nuclear-nuclear dipolar flip-flop coupling
    Z = NV-induced hyperfine-field gradient.

The parameters are based on Supplementary Table 4 of

    Abobeih et al.,
    Nature Communications 9, 2552 (2018).

This example treats the six resolved pairs as independent
subsystems. Interactions between different pairs are not
included.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.physics import (
    Carbon13Pair,
    IndependentNuclearEnvironment,
)


# ============================================================
# Published 13C-13C pair parameters
# ============================================================
#
# nvspin stores ordinary frequencies in GHz.
#
#     1 kHz = 1e-6 GHz
#
# X values below use the pair coupling values reported /
# assigned in Supplementary Table 4.
#
# Z is the hyperfine-field-gradient parameter inferred from
# the measured conditional pseudospin frequencies.
# ============================================================

PAIRS = (

    Carbon13Pair(
        label="Pair 1",
        X=0.2367e-6,
        Z=7.890e-6,
    ),

    Carbon13Pair(
        label="Pair 2",
        X=0.2367e-6,
        Z=6.582e-6,
    ),

    Carbon13Pair(
        label="Pair 3",
        X=0.0761e-6,
        Z=4.420e-6,
    ),

    Carbon13Pair(
        label="Pair 4",
        X=2.061e-6,
        Z=0.230e-6,
    ),

    Carbon13Pair(
        label="Pair 5",
        X=0.1868e-6,
        Z=2.801e-6,
    ),

    Carbon13Pair(
        label="Pair 6",
        X=0.1334e-6,
        Z=1.826e-6,
    ),
)


# ============================================================
# DD scan
# ============================================================
#
# Use one common N initially so that the six pair responses
# can be compared under exactly the same control sequence.
#
# The experimental table used different N values for
# characterizing different pairs, so this first plot is not
# intended as a point-by-point reproduction of that table.
# ============================================================

NUMBER_OF_PULSES = 16

SPACING_MIN = 1000.0
SPACING_MAX = 600000.0

POINTS = 6000


def main():

    environment = (
        IndependentNuclearEnvironment(
            subsystems=PAIRS
        )
    )

    pulse_spacings = np.linspace(
        SPACING_MIN,
        SPACING_MAX,
        POINTS,
    )

    individual = np.zeros(
        (
            len(PAIRS),
            POINTS,
        ),
        dtype=complex,
    )

    total = np.zeros(
        POINTS,
        dtype=complex,
    )

    # ========================================================
    # Simulate
    # ========================================================

    for index, spacing in enumerate(
        pulse_spacings
    ):

        contributions = (
            environment.individual_coherences(
                number_of_pulses=(
                    NUMBER_OF_PULSES
                ),
                pulse_spacing=(
                    float(spacing)
                ),
            )
        )

        for pair_index, value in enumerate(
            contributions
        ):

            individual[
                pair_index,
                index,
            ] = value

        total[index] = np.prod(
            contributions
        )

    # ========================================================
    # Convert coherence to Px
    # ========================================================

    individual_px = (
        0.5
        * (
            1.0
            + np.real(
                individual
            )
        )
    )

    total_px = (
        0.5
        * (
            1.0
            + np.real(
                total
            )
        )
    )

    # ========================================================
    # Print pair information
    # ========================================================

    print()
    print(
        "Six-pair 13C-13C DD spectroscopy"
    )

    print(
        "--------------------------------"
    )

    print(
        "Pair      "
        "X (kHz)      "
        "Z (kHz)      "
        "f0 (kHz)      "
        "f1 (kHz)"
    )

    for pair in PAIRS:

        print(
            f"{pair.label:8s}  "
            f"{pair.X * 1e6:8.4f}    "
            f"{pair.Z * 1e6:8.4f}    "
            f"{pair.frequency_0 * 1e6:8.4f}    "
            f"{pair.frequency_1 * 1e6:8.4f}"
        )

    # ========================================================
    # Print strongest feature for each pair
    # ========================================================

    print()
    print(
        "Strongest response in scan"
    )

    print(
        "--------------------------"
    )

    print(
        "Pair      spacing (us)      minimum Px"
    )

    for pair_index, pair in enumerate(
        PAIRS
    ):

        minimum_index = np.argmin(
            individual_px[
                pair_index
            ]
        )

        minimum_spacing = (
            pulse_spacings[
                minimum_index
            ]
        )

        minimum_probability = (
            individual_px[
                pair_index,
                minimum_index,
            ]
        )

        print(
            f"{pair.label:8s}  "
            f"{minimum_spacing / 1000:12.3f}    "
            f"{minimum_probability:10.5f}"
        )

    # ========================================================
    # Figure 1
    #
    # Individual pair responses
    # ========================================================

    plt.figure(
        figsize=(12, 8)
    )

    for pair_index, pair in enumerate(
        PAIRS
    ):

        plt.plot(
            pulse_spacings
            / 1000.0,
            individual_px[
                pair_index
            ],
            linewidth=1.1,
            label=pair.label,
        )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Individual contribution $P_x$"
    )

    plt.title(
        r"Individual $^{13}$C--$^{13}$C "
        "Pair DD Responses"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$"
    )

    plt.ylim(
        -0.03,
        1.03,
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 2
    #
    # Combined six-pair response
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        pulse_spacings
        / 1000.0,
        total_px,
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
        r"DD Spectroscopy of Six "
        r"$^{13}$C--$^{13}$C Pairs"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$"
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