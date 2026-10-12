"""NV fluorescence contrast from a five-level optical rate model."""

import numpy as np
import matplotlib.pyplot as plt

from nvspin.readout import (
    FluorescenceReadout,
    NVOpticalCycle,
    NVOpticalRates,
)


def main():

    # ========================================================
    # Optical rates
    #
    # Initial illustrative values in 1/ns.
    #
    # These are NOT being claimed as universal NV constants.
    # They are parameters of the optical model and should
    # ultimately be fitted/chosen from appropriate literature
    # or experimental calibration.
    # ========================================================

    rates = NVOpticalRates(
        excitation=0.05,

        radiative_ms0=0.077,
        radiative_ms1=0.077,

        isc_ms0=0.005,
        isc_ms1=0.050,

        singlet_to_ms0=0.004,
        singlet_to_ms1=0.001,
    )

    cycle = NVOpticalCycle(
        rates
    )

    readout = FluorescenceReadout(
        optical_cycle=cycle,
        collection_efficiency=1.0,
        background_rate=0.0,
    )

    # ========================================================
    # Initial spin states
    # ========================================================

    bright_initial = np.array(
        [
            1.0,   # ground ms=0
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    dark_initial = np.array(
        [
            0.0,
            1.0,   # ground ms=+/-1
            0.0,
            0.0,
            0.0,
        ]
    )

    # ========================================================
    # Sweep readout-window duration
    # ========================================================

    readout_times = np.linspace(
        5.0,
        2000.0,
        300,
    )

    bright_photons = np.zeros_like(
        readout_times
    )

    dark_photons = np.zeros_like(
        readout_times
    )

    contrasts = np.zeros_like(
        readout_times
    )

    for index, readout_time in enumerate(
        readout_times
    ):

        bright = readout.integrated_photons(
            bright_initial,
            readout_time=readout_time,
            steps=500,
        )

        dark = readout.integrated_photons(
            dark_initial,
            readout_time=readout_time,
            steps=500,
        )

        bright_photons[index] = bright
        dark_photons[index] = dark

        contrasts[index] = (
            readout.contrast(
                bright,
                dark,
            )
        )

    # ========================================================
    # Report maximum contrast
    # ========================================================

    maximum_index = np.argmax(
        contrasts
    )

    print()
    print("Optical readout")
    print("---------------")

    print(
        f"Maximum contrast: "
        f"{100 * contrasts[maximum_index]:.2f} %"
    )

    print(
        f"Readout time at maximum: "
        f"{readout_times[maximum_index]:.1f} ns"
    )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        readout_times,
        100.0 * contrasts,
        color="black",
        linewidth=1.5,
    )

    plt.xlabel(
        "Readout-window duration (ns)"
    )

    plt.ylabel(
        "Spin-readout contrast (%)"
    )

    plt.title(
        "NV Optical Readout Contrast\n"
        "Five-Level Rate-Equation Model"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()