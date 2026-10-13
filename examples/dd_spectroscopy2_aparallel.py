"""
Effect of longitudinal hyperfine coupling on DD spectroscopy.

We vary A_parallel while holding:

    f_L
    A_perpendicular
    N

fixed.

The high-field resonance prediction in the pulse-center
spacing convention used by nvspin is approximately

    spacing_k =
        (2k - 1) / (2 f_L + A_parallel)

for k = 1, 2, 3, ...

For the fundamental resonance:

    spacing_1 =
        1 / (2 f_L + A_parallel)

The goal is to demonstrate that A_parallel shifts the
resonance position.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.experiments import (
    SingleCarbon13,
    SingleCarbonDDSpectroscopy,
)


# ============================================================
# Parameters
# ============================================================

F_LARMOR = 0.0004282

A_PERPENDICULAR = 0.000030

A_PARALLEL_VALUES = [
    0.000000,
    0.000020,
    0.000040,
    0.000060,
]

NUMBER_OF_PULSES = 32

SPACING_RANGE = (
    950.0,
    1300.0,
)

POINTS = 1200


# ============================================================
# Analytical prediction
# ============================================================

def predicted_resonance(
    f_larmor,
    A_parallel,
    order=1,
):
    """
    High-field resonance prediction.

    Returns center-to-center pulse spacing in ns.

    Frequencies are in GHz.
    """

    return (
        (2 * order - 1)
        / (
            2.0 * f_larmor
            + A_parallel
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    plt.figure(
        figsize=(10, 7)
    )

    print()
    print(
        "A_parallel DD spectroscopy sweep"
    )
    print(
        "--------------------------------"
    )

    print(
        "A_parallel (kHz)   "
        "predicted (us)   "
        "simulated (us)   "
        "difference (ns)"
    )

    for A_parallel in (
        A_PARALLEL_VALUES
    ):

        carbon = SingleCarbon13(
            larmor_frequency=F_LARMOR,
            A_parallel=A_parallel,
            A_perpendicular=(
                A_PERPENDICULAR
            ),
        )

        experiment = (
            SingleCarbonDDSpectroscopy(
                carbon=carbon,
                number_of_pulses=(
                    NUMBER_OF_PULSES
                ),
                spacing_range=(
                    SPACING_RANGE
                ),
                points=POINTS,
            )
        )

        result = experiment.simulate()

        # ----------------------------------------------------
        # Numerical minimum
        # ----------------------------------------------------

        minimum_index = np.argmin(
            result.preservation_probability
        )

        simulated_resonance = (
            result.pulse_spacings[
                minimum_index
            ]
        )

        # ----------------------------------------------------
        # Analytical high-field prediction
        # ----------------------------------------------------

        predicted = predicted_resonance(
            F_LARMOR,
            A_parallel,
        )

        difference = (
            simulated_resonance
            - predicted
        )

        print(
            f"{A_parallel * 1e6:16.1f}   "
            f"{predicted / 1000:14.6f}   "
            f"{simulated_resonance / 1000:14.6f}   "
            f"{difference:14.3f}"
        )

        # ----------------------------------------------------
        # Plot numerical spectrum
        # ----------------------------------------------------

        plt.plot(
            result.pulse_spacings
            / 1000.0,
            result.preservation_probability,
            linewidth=1.5,
            label=(
                rf"$A_\parallel="
                f"{A_parallel * 1e6:.0f}"
                r"\,\mathrm{kHz}$"
            ),
        )

        # Analytical resonance marker
        plt.axvline(
            predicted / 1000.0,
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )

    # Bare Larmor prediction
    bare_resonance = (
        1.0
        / (
            2.0
            * F_LARMOR
        )
    )

    plt.axvline(
        bare_resonance / 1000.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=r"Bare $1/(2f_L)$",
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        r"Effect of $A_\parallel$ on "
        r"$^{13}$C DD Resonance"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$, "
        rf"$A_\perp="
        f"{A_PERPENDICULAR * 1e6:.0f}"
        r"\,\mathrm{kHz}$"
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