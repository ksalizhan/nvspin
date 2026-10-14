"""
Structured nuclear environment inspired by Abobeih et al.

Environment
-----------

    7 isolated 13C nuclear spins
    +
    6 interacting 13C-13C pseudospin pairs

The isolated nuclei use

    H0 = f_L I_z

    H1 = (f_L + A_parallel) I_z
         + A_perpendicular I_x

The interacting pairs use

    H0 = X I_x

    H1 = X I_x + Z I_z

All resolved subsystems are currently assumed independent
of one another, so the total NV coherence factorizes:

    L_total
        = product_k L_carbon,k
          product_j L_pair,j

This model therefore contains explicit nuclear-nuclear
interaction WITHIN each pair, but not interactions between
different pairs or between pairs and isolated nuclei.

No OU bath, finite pulse duration, T1 relaxation, pulse
errors, optical readout model, or unresolved microscopic
13C bath is included in this example.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.physics import (
    Carbon13Pair,
    Carbon13Spin,
    IndependentNuclearEnvironment,
)


# ============================================================
# Seven isolated 13C spins
# ============================================================

CARBONS = (

    Carbon13Spin(
        label="C1",
        larmor_frequency=0.000431994,
        A_parallel=-0.0000364,
        A_perpendicular=0.000025,
    ),

    Carbon13Spin(
        label="C2",
        larmor_frequency=0.000431874,
        A_parallel=0.0000206,
        A_perpendicular=0.000043,
    ),

    Carbon13Spin(
        label="C3",
        larmor_frequency=0.000431891,
        A_parallel=-0.0000114,
        A_perpendicular=0.000059,
    ),

    Carbon13Spin(
        label="C4",
        larmor_frequency=0.000431947,
        A_parallel=0.0000081,
        A_perpendicular=0.000021,
    ),

    Carbon13Spin(
        label="C5",
        larmor_frequency=0.000431934,
        A_parallel=0.0000244,
        A_perpendicular=0.000026,
    ),

    Carbon13Spin(
        label="C6",
        larmor_frequency=0.000431960,
        A_parallel=-0.0000487,
        A_perpendicular=0.000012,
    ),

    Carbon13Spin(
        label="C7",
        larmor_frequency=0.000431950,
        A_parallel=0.0000145,
        A_perpendicular=0.000011,
    ),
)


# ============================================================
# Six interacting 13C-13C pairs
# ============================================================
#
# X and Z are stored as ordinary frequencies in GHz.
#
# Values quoted as X/(2*pi), Z/(2*pi) in kHz in the
# angular-frequency convention correspond numerically to
# these ordinary-frequency values in nvspin.
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
# DD parameters
# ============================================================
#
# Use one common ideal DD sequence for every subsystem.
#
# This is a controlled theoretical comparison. It is NOT
# yet a point-by-point reconstruction of the experimental
# sequence used to characterize each individual pair.
# ============================================================

NUMBER_OF_PULSES = 32

SPACING_MIN = 500.0
SPACING_MAX = 600000.0

POINTS = 8000


def main():

    # ========================================================
    # Separate environments
    # ========================================================

    carbon_environment = (
        IndependentNuclearEnvironment(
            subsystems=CARBONS
        )
    )

    pair_environment = (
        IndependentNuclearEnvironment(
            subsystems=PAIRS
        )
    )

    # Complete structured environment.
    #
    # This remains a product of independent resolved
    # subsystems.

    full_environment = (
        IndependentNuclearEnvironment(
            subsystems=(
                *CARBONS,
                *PAIRS,
            )
        )
    )

    pulse_spacings = np.linspace(
        SPACING_MIN,
        SPACING_MAX,
        POINTS,
    )

    # ========================================================
    # Allocate results
    # ========================================================

    carbon_coherence = np.zeros(
        POINTS,
        dtype=complex,
    )

    pair_coherence = np.zeros(
        POINTS,
        dtype=complex,
    )

    total_coherence = np.zeros(
        POINTS,
        dtype=complex,
    )

    individual_carbons = np.zeros(
        (
            len(CARBONS),
            POINTS,
        ),
        dtype=complex,
    )

    individual_pairs = np.zeros(
        (
            len(PAIRS),
            POINTS,
        ),
        dtype=complex,
    )

    # ========================================================
    # Simulate
    # ========================================================

    for index, spacing in enumerate(
        pulse_spacings
    ):

        spacing = float(
            spacing
        )

        # ----------------------------------------------------
        # Individual isolated nuclei
        # ----------------------------------------------------

        carbon_values = (
            carbon_environment.individual_coherences(
                number_of_pulses=(
                    NUMBER_OF_PULSES
                ),
                pulse_spacing=spacing,
            )
        )

        for carbon_index, value in enumerate(
            carbon_values
        ):

            individual_carbons[
                carbon_index,
                index,
            ] = value

        carbon_coherence[index] = np.prod(
            carbon_values
        )

        # ----------------------------------------------------
        # Individual interacting pairs
        # ----------------------------------------------------

        pair_values = (
            pair_environment.individual_coherences(
                number_of_pulses=(
                    NUMBER_OF_PULSES
                ),
                pulse_spacing=spacing,
            )
        )

        for pair_index, value in enumerate(
            pair_values
        ):

            individual_pairs[
                pair_index,
                index,
            ] = value

        pair_coherence[index] = np.prod(
            pair_values
        )

        # ----------------------------------------------------
        # Complete resolved environment
        # ----------------------------------------------------

        total_coherence[index] = (
            full_environment.coherence(
                number_of_pulses=(
                    NUMBER_OF_PULSES
                ),
                pulse_spacing=spacing,
            )
        )

    # ========================================================
    # Convert to preservation probabilities
    # ========================================================

    carbon_px = (
        0.5
        * (
            1.0
            + np.real(
                carbon_coherence
            )
        )
    )

    pair_px = (
        0.5
        * (
            1.0
            + np.real(
                pair_coherence
            )
        )
    )

    total_px = (
        0.5
        * (
            1.0
            + np.real(
                total_coherence
            )
        )
    )

    individual_carbon_px = (
        0.5
        * (
            1.0
            + np.real(
                individual_carbons
            )
        )
    )

    individual_pair_px = (
        0.5
        * (
            1.0
            + np.real(
                individual_pairs
            )
        )
    )

    # ========================================================
    # Numerical factorization check
    # ========================================================

    expected_total = (
        carbon_coherence
        * pair_coherence
    )

    maximum_factorization_error = np.max(
        np.abs(
            total_coherence
            - expected_total
        )
    )

    print()
    print(
        "Structured Abobeih-like nuclear environment"
    )

    print(
        "-------------------------------------------"
    )

    print(
        f"Isolated carbons: {len(CARBONS)}"
    )

    print(
        f"Interacting pairs: {len(PAIRS)}"
    )

    print(
        f"Resolved subsystems: "
        f"{len(CARBONS) + len(PAIRS)}"
    )

    print(
        f"DD pulses: {NUMBER_OF_PULSES}"
    )

    print(
        f"Maximum pulse spacing: "
        f"{SPACING_MAX / 1000:.1f} us"
    )

    print(
        f"Maximum total evolution time: "
        f"{NUMBER_OF_PULSES * SPACING_MAX / 1e6:.3f} ms"
    )

    print(
        f"Maximum factorization error: "
        f"{maximum_factorization_error:.3e}"
    )

    # ========================================================
    # Print strongest individual pair features
    # ========================================================

    print()
    print(
        "Strongest pair features"
    )

    print(
        "-----------------------"
    )

    print(
        "Pair      spacing (us)      minimum Px"
    )

    for pair_index, pair in enumerate(
        PAIRS
    ):

        minimum_index = np.argmin(
            individual_pair_px[
                pair_index
            ]
        )

        print(
            f"{pair.label:8s}  "
            f"{pulse_spacings[minimum_index] / 1000:12.3f}    "
            f"{individual_pair_px[pair_index, minimum_index]:10.5f}"
        )

    # ========================================================
    # Figure 1
    #
    # Seven isolated-carbon contributions
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    for carbon_index, carbon in enumerate(
        CARBONS
    ):

        plt.plot(
            pulse_spacings
            / 1000.0,
            individual_carbon_px[
                carbon_index
            ],
            linewidth=0.9,
            label=carbon.label,
        )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Individual contribution $P_x$"
    )

    plt.title(
        r"Seven Isolated $^{13}$C Responses"
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
    # Six pair contributions
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    for pair_index, pair in enumerate(
        PAIRS
    ):

        plt.plot(
            pulse_spacings
            / 1000.0,
            individual_pair_px[
                pair_index
            ],
            linewidth=1.0,
            label=pair.label,
        )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Individual contribution $P_x$"
    )

    plt.title(
        r"Six Interacting "
        r"$^{13}$C--$^{13}$C Pair Responses"
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
    # Figure 3
    #
    # Compare the two environment sectors
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        pulse_spacings
        / 1000.0,
        carbon_px,
        linewidth=1.1,
        label=(
            r"7 isolated $^{13}$C"
        ),
    )

    plt.plot(
        pulse_spacings
        / 1000.0,
        pair_px,
        linewidth=1.1,
        label=(
            r"6 $^{13}$C--$^{13}$C pairs"
        ),
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        "Resolved Nuclear-Environment Sectors"
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

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 4
    #
    # Complete structured environment
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        pulse_spacings
        / 1000.0,
        total_px,
        color="black",
        linewidth=1.1,
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        "Structured Nuclear Environment"
        "\n"
        r"7 isolated $^{13}$C + "
        r"6 interacting $^{13}$C--$^{13}$C pairs"
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