"""
Effect of transverse hyperfine coupling on DD spectroscopy.

We vary A_perpendicular while holding:

    f_L
    A_parallel
    N

fixed.

The goal is to demonstrate that A_perpendicular primarily
controls the strength of the conditional nuclear dynamics,
while the resonance center remains approximately fixed.
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

A_PARALLEL = 0.000040

A_PERPENDICULAR_VALUES = [
    0.000005,
    0.000010,
    0.000020,
    0.000030,
    0.000040,
]

NUMBER_OF_PULSES = 32

SPACING_RANGE = (
    1000.0,
    1230.0,
)

POINTS = 1200


# ============================================================
# Analytical resonance
# ============================================================

def predicted_resonance():
    """
    High-field fundamental resonance prediction.

    Center-to-center pulse spacing in ns.
    """

    return (
        1.0
        / (
            2.0 * F_LARMOR
            + A_PARALLEL
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    prediction = (
        predicted_resonance()
    )

    plt.figure(
        figsize=(10, 7)
    )

    print()
    print(
        "A_perpendicular DD spectroscopy sweep"
    )
    print(
        "-------------------------------------"
    )

    print(
        "A_perp (kHz)   "
        "dip position (us)   "
        "minimum Px   "
        "contrast"
    )

    for A_perpendicular in (
        A_PERPENDICULAR_VALUES
    ):

        carbon = SingleCarbon13(
            larmor_frequency=F_LARMOR,
            A_parallel=A_PARALLEL,
            A_perpendicular=(
                A_perpendicular
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
        # Strongest dip
        # ----------------------------------------------------

        minimum_index = np.argmin(
            result.preservation_probability
        )

        dip_position = (
            result.pulse_spacings[
                minimum_index
            ]
        )

        minimum_probability = (
            result.preservation_probability[
                minimum_index
            ]
        )

        contrast = (
            1.0
            - minimum_probability
        )

        print(
            f"{A_perpendicular * 1e6:12.1f}   "
            f"{dip_position / 1000:17.6f}   "
            f"{minimum_probability:10.5f}   "
            f"{contrast:8.5f}"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.plot(
            result.pulse_spacings
            / 1000.0,
            result.preservation_probability,
            linewidth=1.5,
            label=(
                rf"$A_\perp="
                f"{A_perpendicular * 1e6:.0f}"
                r"\,\mathrm{kHz}$"
            ),
        )

    # --------------------------------------------------------
    # Predicted resonance center
    # --------------------------------------------------------

    plt.axvline(
        prediction / 1000.0,
        color="black",
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
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        r"Effect of $A_\perp$ on "
        r"$^{13}$C DD Resonance"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$, "
        rf"$A_\parallel="
        f"{A_PARALLEL * 1e6:.0f}"
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