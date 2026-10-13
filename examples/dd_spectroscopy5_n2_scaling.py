"""
Verify weak-signal N^2 amplification in single-13C
dynamical-decoupling spectroscopy.

At resonance, the conditional nuclear response contains

    sin^2(N phi / 2).

For sufficiently small N phi,

    sin^2(N phi / 2)
        ~ N^2 phi^2 / 4,

so the spectroscopy contrast should satisfy

    1 - P_x ~ N^2.

This scaling is expected only in the weak-response regime.
At larger N the exact coherent oscillation must replace the
quadratic approximation.
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
# Pulse-number sweep
# ============================================================

# Use even N to remain consistent with the symmetric
# conditional-evolution units used in the spectroscopy model.

PULSE_NUMBERS = np.arange(
    2,
    22,
    2,
    dtype=int,
)


# ============================================================
# Resonance
# ============================================================

def predicted_resonance():
    """
    High-field fundamental resonance in nvspin's
    center-to-center pulse-spacing convention.

        spacing_res ~ 1 / (2 f_L + A_parallel)
    """

    return (
        1.0
        / (
            2.0 * F_LARMOR
            + A_PARALLEL
        )
    )


# ============================================================
# Calculate P_x directly
# ============================================================

def preservation_probability(
    carbon,
    number_of_pulses,
    pulse_spacing,
):
    """Calculate P_x at one pulse spacing."""

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

    carbon = SingleCarbon13(
        larmor_frequency=F_LARMOR,
        A_parallel=A_PARALLEL,
        A_perpendicular=A_PERPENDICULAR,
    )

    resonance = (
        predicted_resonance()
    )

    probabilities = np.zeros(
        len(PULSE_NUMBERS),
        dtype=float,
    )

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------

    for index, number_of_pulses in enumerate(
        PULSE_NUMBERS
    ):

        probabilities[index] = (
            preservation_probability(
                carbon,
                int(number_of_pulses),
                resonance,
            )
        )

    contrast = (
        1.0
        - probabilities
    )

    n_squared = (
        PULSE_NUMBERS.astype(float) ** 2
    )

    normalized_contrast = (
        contrast
        / n_squared
    )

    # ========================================================
    # Fit the weak-signal contrast to
    #
    #     contrast = slope * N^2
    #
    # with zero intercept.
    # ========================================================

    slope = (
        np.dot(
            n_squared,
            contrast,
        )
        / np.dot(
            n_squared,
            n_squared,
        )
    )

    fitted_contrast = (
        slope
        * n_squared
    )

    # --------------------------------------------------------
    # Relative deviation from the N^2 model
    # --------------------------------------------------------

    relative_error = np.zeros_like(
        contrast
    )

    nonzero = (
        fitted_contrast > 0
    )

    relative_error[
        nonzero
    ] = (
        contrast[nonzero]
        - fitted_contrast[nonzero]
    ) / fitted_contrast[nonzero]

    # ========================================================
    # Print results
    # ========================================================

    print()
    print(
        "Weak-signal N^2 scaling test"
    )

    print(
        "----------------------------"
    )

    print(
        f"Predicted resonance: "
        f"{resonance / 1000:.6f} us"
    )

    print()

    print(
        "N      Px          "
        "1-Px        "
        "(1-Px)/N^2"
    )

    for (
        number_of_pulses,
        probability,
        signal,
        normalized,
    ) in zip(
        PULSE_NUMBERS,
        probabilities,
        contrast,
        normalized_contrast,
    ):

        print(
            f"{number_of_pulses:2d}   "
            f"{probability:10.7f}   "
            f"{signal:10.7f}   "
            f"{normalized:12.8e}"
        )

    print()

    print(
        "Best zero-intercept quadratic coefficient:"
    )

    print(
        f"1 - Px ~ "
        f"{slope:.8e} N^2"
    )

    # ========================================================
    # Figure 1
    #
    # Contrast versus N
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        PULSE_NUMBERS,
        contrast,
        marker="o",
        linewidth=1.5,
        label="Exact simulation",
    )

    plt.plot(
        PULSE_NUMBERS,
        fitted_contrast,
        linestyle="--",
        linewidth=1.5,
        label=r"Fit: $C \propto N^2$",
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"Resonance contrast $1-P_x$"
    )

    plt.title(
        r"Weak-Signal $N^2$ Amplification"
        "\n"
        r"Single-$^{13}$C DD Spectroscopy"
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
    # Contrast versus N^2
    #
    # True quadratic scaling should become a straight line.
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        n_squared,
        contrast,
        marker="o",
        linewidth=1.5,
        label="Exact simulation",
    )

    plt.plot(
        n_squared,
        fitted_contrast,
        linestyle="--",
        linewidth=1.5,
        label="Linear in $N^2$",
    )

    plt.xlabel(
        r"$N^2$"
    )

    plt.ylabel(
        r"Resonance contrast $1-P_x$"
    )

    plt.title(
        r"Test of $1-P_x \propto N^2$"
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
    # Normalized contrast.
    #
    # A plateau means N^2 scaling.
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        PULSE_NUMBERS,
        normalized_contrast,
        marker="o",
        linewidth=1.5,
    )

    plt.axhline(
        slope,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Quadratic-fit coefficient",
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"$(1-P_x)/N^2$"
    )

    plt.title(
        r"Normalized Weak-$^{13}$C Signal"
        "\n"
        r"Plateau $\Rightarrow N^2$ scaling"
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
    # Deviation from quadratic approximation
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        PULSE_NUMBERS,
        100.0 * relative_error,
        marker="o",
        linewidth=1.5,
    )

    plt.axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"Deviation from $N^2$ fit (%)"
    )

    plt.title(
        "Breakdown of the Weak-Signal Approximation"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()