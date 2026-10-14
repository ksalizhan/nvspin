"""
DD spectroscopy of seven independent 13C nuclear spins.

Parameters are based on Supplementary Table 2 of:

    Abobeih et al.,
    Nature Communications 9, 2552 (2018).

This first implementation treats the seven nuclei as
independent spin-1/2 subsystems.

Nuclear-nuclear interactions and 13C-13C pairs are not yet
included.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.physics import (
    Carbon13Spin,
    IndependentNuclearEnvironment,
)


# ============================================================
# Published isolated-carbon parameters
# ============================================================
#
# Frequencies:
#
#     GHz
#
# Hyperfine values from kHz -> GHz:
#
#     1 kHz = 1e-6 GHz
#
# Note:
# These values reproduce the parameter table, but a fully
# quantitative reproduction must preserve the exact electron
# manifold/sign convention used for each nucleus.
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
# DD scan
# ============================================================

NUMBER_OF_PULSES = 32

# Start with a broad scan.
#
# Our pulse_spacing means center-to-center separation between
# adjacent pi pulses.

SPACING_MIN = 500.0
SPACING_MAX = 10000.0 # 10 us

POINTS = 3000


def main():

    environment = (
        IndependentNuclearEnvironment(
            subsystems=CARBONS
        )
    )

    pulse_spacings = np.linspace(
        SPACING_MIN,
        SPACING_MAX,
        POINTS,
    )

    individual = np.zeros(
        (
            len(CARBONS),
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

        for carbon_index, value in enumerate(
            contributions
        ):

            individual[
                carbon_index,
                index,
            ] = value

        total[index] = np.prod(
            contributions
        )

    # ========================================================
    # Convert NV coherence to preservation probability
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
    # Print approximate fundamental resonances
    # ========================================================

    print()
    print(
        "Seven-carbon DD spectroscopy"
    )

    print(
        "----------------------------"
    )

    print(
        "Carbon    f0 (kHz)    "
        "A_parallel (kHz)    "
        "A_perp (kHz)    "
        "approx spacing (us)"
    )

    for carbon in CARBONS:

        approximate_spacing = (
            1.0
            / (
                2.0
                * carbon.larmor_frequency
                + carbon.A_parallel
            )
        )

        print(
            f"{carbon.label:6s}    "
            f"{carbon.larmor_frequency * 1e6:8.3f}    "
            f"{carbon.A_parallel * 1e6:16.1f}    "
            f"{carbon.A_perpendicular * 1e6:12.1f}    "
            f"{approximate_spacing / 1000:17.6f}"
        )

    # ========================================================
    # Figure 1:
    #
    # Individual nuclear contributions
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )

    for carbon_index, carbon in enumerate(
        CARBONS
    ):

        plt.plot(
            pulse_spacings
            / 1000.0,
            individual_px[
                carbon_index
            ],
            linewidth=1.0,
            label=carbon.label,
        )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Individual contribution $P_x$"
    )

    plt.title(
        "Individual Isolated "
        r"$^{13}$C DD Responses"
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
    # Figure 2:
    #
    # Combined seven-carbon spectrum
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        pulse_spacings
        / 1000.0,
        total_px,
        color="black",
        linewidth=1.3,
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        "DD Spectroscopy of Seven "
        r"Independent $^{13}$C Spins"
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