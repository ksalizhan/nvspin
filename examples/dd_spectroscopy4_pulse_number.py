"""
Pulse-number dependence of single-13C DD spectroscopy.

This example demonstrates two related effects:

1. The DD resonance versus pulse-center spacing for several
   numbers of refocusing pulses N.

2. The electron preservation probability P_x at resonance
   as a function of N.

For weak conditional rotations, increasing N amplifies the
nuclear-spin signal approximately quadratically.

For sufficiently large N, however, the conditional nuclear
rotation becomes coherent and the signal oscillates with N.

No OU bath, T1 relaxation, finite pulse duration, or pulse
errors are included.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.experiments import (
    SingleCarbon13,
    SingleCarbonDDSpectroscopy,
)


# ============================================================
# Nuclear parameters
# ============================================================

F_LARMOR = 0.0004282       # GHz = 0.4282 MHz

A_PARALLEL = 0.000040      # GHz = 40 kHz

A_PERPENDICULAR = 0.000030 # GHz = 30 kHz


# ============================================================
# Spectroscopy parameters
# ============================================================

SPECTRUM_PULSE_NUMBERS = [
    8,
    16,
    32,
    56,
    64,
    96,
]

SPACING_RANGE = (
    1.04e3,
    1.19e3,
)

SPECTRUM_POINTS = 1200


# ============================================================
# N sweep
# ============================================================

N_MIN = 2
N_MAX = 120
N_STEP = 2


# ============================================================
# Analytical resonance prediction
# ============================================================

def predicted_resonance():
    """
    High-field fundamental resonance prediction.

    nvspin uses center-to-center pi-pulse spacing:

        spacing_res ~ 1 / (2 f_L + A_parallel)

    Frequencies are in GHz, so the result is in ns.
    """

    return (
        1.0
        / (
            2.0 * F_LARMOR
            + A_PARALLEL
        )
    )


# ============================================================
# Build carbon
# ============================================================

def build_carbon():

    return SingleCarbon13(
        larmor_frequency=F_LARMOR,
        A_parallel=A_PARALLEL,
        A_perpendicular=A_PERPENDICULAR,
    )


# ============================================================
# Spectroscopy curves
# ============================================================

def simulate_spectrum(
    carbon,
    number_of_pulses,
):

    experiment = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=(
                number_of_pulses
            ),
            spacing_range=(
                SPACING_RANGE
            ),
            points=SPECTRUM_POINTS,
        )
    )

    return experiment.simulate()


# ============================================================
# P_x at one spacing
# ============================================================

def preservation_at_spacing(
    carbon,
    number_of_pulses,
    pulse_spacing,
):
    """
    Calculate P_x directly at one pulse spacing.
    """

    # The public experiment currently scans a range, so use
    # the existing coherence_at_spacing method directly.

    experiment = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=(
                number_of_pulses
            ),
            spacing_range=(
                0.999 * pulse_spacing,
                1.001 * pulse_spacing,
            ),
            points=2,
        )
    )

    coherence = (
        experiment.coherence_at_spacing(
            pulse_spacing
        )
    )

    return (
        0.5
        * (
            1.0
            + coherence
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    carbon = build_carbon()

    resonance = (
        predicted_resonance()
    )

    print()
    print(
        "Single-13C pulse-number dependence"
    )

    print(
        "----------------------------------"
    )

    print(
        f"f_L = "
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
        f"Predicted fundamental resonance = "
        f"{resonance / 1000:.6f} us"
    )

    # ========================================================
    # Figure 1
    #
    # Spectroscopy versus pulse spacing for selected N
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    print()
    print(
        "Spectroscopy minima"
    )

    print(
        "-------------------"
    )

    print(
        "N      dip position (us)      minimum Px"
    )

    for number_of_pulses in (
        SPECTRUM_PULSE_NUMBERS
    ):

        result = simulate_spectrum(
            carbon,
            number_of_pulses,
        )

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

        print(
            f"{number_of_pulses:3d}    "
            f"{dip_position / 1000:17.6f}    "
            f"{minimum_probability:10.5f}"
        )

        plt.plot(
            result.pulse_spacings
            / 1000.0,
            result.preservation_probability,
            linewidth=1.5,
            label=(
                f"N = {number_of_pulses}"
            ),
        )

    plt.axvline(
        resonance / 1000.0,
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
        r"Pulse-Number Dependence of "
        r"Single-$^{13}$C DD Spectroscopy"
        "\n"
        rf"$A_\parallel="
        f"{A_PARALLEL * 1e6:.0f}"
        r"\,\mathrm{kHz}$, "
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

    # ========================================================
    # Figure 2
    #
    # P_x at analytical resonance versus N
    # ========================================================

    pulse_numbers = np.arange(
        N_MIN,
        N_MAX + 1,
        N_STEP,
        dtype=int,
    )

    probabilities = np.zeros(
        len(pulse_numbers),
        dtype=float,
    )

    for index, number_of_pulses in enumerate(
        pulse_numbers
    ):

        probabilities[index] = (
            preservation_at_spacing(
                carbon,
                int(
                    number_of_pulses
                ),
                resonance,
            )
        )

    # --------------------------------------------------------
    # Locate extrema
    # --------------------------------------------------------

    minimum_index = np.argmin(
        probabilities
    )

    minimum_N = (
        pulse_numbers[
            minimum_index
        ]
    )

    minimum_Px = (
        probabilities[
            minimum_index
        ]
    )

    print()
    print(
        "Pulse-number sweep at resonance"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Lowest P_x in scanned range: "
        f"N = {minimum_N}, "
        f"P_x = {minimum_Px:.5f}"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        pulse_numbers,
        probabilities,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="black",
    )

    plt.scatter(
        [minimum_N],
        [minimum_Px],
        s=60,
        zorder=5,
        label=(
            f"Minimum: N={minimum_N}"
        ),
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"$P_x$ at predicted resonance"
    )

    plt.title(
        r"Coherent Nuclear Response versus "
        r"DD Pulse Number"
        "\n"
        rf"$\tau_\mathrm{{res}}="
        f"{resonance / 1000:.4f}"
        r"\,\mu\mathrm{s}$"
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
    # Figure 3
    #
    # Resonance contrast versus N
    # ========================================================

    contrast = (
        1.0
        - probabilities
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        pulse_numbers,
        contrast,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="black",
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"Resonance contrast $1-P_x$"
    )

    plt.title(
        r"Single-$^{13}$C Signal Amplification "
        r"versus Pulse Number"
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