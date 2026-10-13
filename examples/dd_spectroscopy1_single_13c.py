"""
Ideal DD spectroscopy of one weakly coupled 13C spin.

The simulation uses an effective NV two-level system and
one explicit spin-1/2 carbon-13 nucleus.

No OU bath, T1, optical readout, or finite-pulse errors are
included.

The scan variable is the center-to-center separation between
consecutive electron pi pulses.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.experiments import (
    SingleCarbon13,
    SingleCarbonDDSpectroscopy,
)


def main():

    # ========================================================
    # Magnetic field / nuclear parameters
    # ========================================================

    # Approximately 40 mT.
    #
    # 13C Larmor frequency:
    #
    #     gamma_C / 2pi ~ 10.705 MHz/T
    #
    # giving:
    #
    #     f_L ~ 0.4282 MHz

    f_larmor = 0.0004282

    # Weak hyperfine coupling.
    #
    # 40 kHz parallel
    # 30 kHz transverse

    A_parallel = 0.000040
    A_perpendicular = 0.000030

    carbon = SingleCarbon13(
        larmor_frequency=f_larmor,
        A_parallel=A_parallel,
        A_perpendicular=A_perpendicular,
    )

    # ========================================================
    # Approximate fundamental resonance
    # ========================================================

    # Ignoring the hyperfine correction:
    #
    # pulse_spacing ~ 1 / (2 f_L)

    approximate_resonance = (
        1.0
        / (
            2.0
            * f_larmor
        )
    )

    print()
    print("Single-13C DD spectroscopy")
    print("--------------------------")

    print(
        f"13C Larmor frequency: "
        f"{f_larmor * 1000:.4f} MHz"
    )

    print(
        f"A_parallel: "
        f"{A_parallel * 1e6:.1f} kHz"
    )

    print(
        f"A_perpendicular: "
        f"{A_perpendicular * 1e6:.1f} kHz"
    )

    print(
        f"Approximate fundamental "
        f"pulse spacing: "
        f"{approximate_resonance:.1f} ns"
    )

    # ========================================================
    # Spectroscopy
    # ========================================================

    experiment = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=32,
            spacing_range=(
                800.0,
                1500.0,
            ),
            points=1200,
        )
    )

    result = experiment.simulate()

    # ========================================================
    # Find strongest dip
    # ========================================================

    minimum_index = np.argmin(
        result.preservation_probability
    )

    measured_resonance = (
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
        f"Strongest simulated dip: "
        f"{measured_resonance:.2f} ns"
    )

    print(
        f"Minimum P_x: "
        f"{minimum_probability:.4f}"
    )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        result.pulse_spacings
        / 1000.0,
        result.preservation_probability,
        color="black",
        linewidth=1.5,
        label=r"$P_x$",
    )

    plt.axvline(
        approximate_resonance
        / 1000.0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=(
            r"$1/(2f_L)$"
        ),
    )

    plt.axvline(
        measured_resonance
        / 1000.0,
        color="blue",
        linestyle=":",
        linewidth=1.0,
        label="Simulated dip",
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        "Ideal DD Spectroscopy of One "
        r"Weakly Coupled $^{13}$C"
        "\n"
        r"$N=32$ ideal refocusing pulses"
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