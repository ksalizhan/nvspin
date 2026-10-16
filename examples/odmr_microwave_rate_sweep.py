"""
Microwave-rate dependence of rate-equation CW ODMR.

Purpose
-------
Test that ODMR contrast emerges dynamically from competition
between microwave population mixing and the optical cycle.

For each maximum microwave mixing rate W_max, the simulation
calculates

    H_spin
        -> microwave transitions
        -> W(f)
        -> optical steady state
        -> fluorescence F(f)

and extracts

    C = 1 - min(F).

Expected behavior
-----------------
W_max -> 0:
    ODMR contrast -> 0.

Increasing W_max:
    ODMR contrast increases.

Large W_max:
    contrast approaches saturation.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin import NVCenter
from nvspin.experiments import ODMR

from nvspin.readout import (
    NVOpticalCycle,
    NVOpticalRates,
)


# ============================================================
# Optical model
# ============================================================

def build_optical_cycle():

    rates = NVOpticalRates(
        excitation=0.05,

        radiative_ms0=0.077,
        radiative_ms1=0.077,

        isc_ms0=0.005,
        isc_ms1=0.050,

        singlet_to_ms0=0.004,
        singlet_to_ms1=0.001,
    )

    return NVOpticalCycle(
        rates
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # NV system
    # ========================================================

    nv = NVCenter(
        D=2.870,
        E=0.0,
        B=[
            0.0,
            0.0,
            0.005,
        ],
    )

    optical_cycle = (
        build_optical_cycle()
    )

    # ========================================================
    # Microwave mixing rates
    # ========================================================
    #
    # Units: 1/ns
    #
    # Include W=0 explicitly as the no-microwave control.
    # ========================================================

    microwave_rates = np.array(
        [
            0.0,
            1.0e-4,
            3.0e-4,
            1.0e-3,
            3.0e-3,
            1.0e-2,
            3.0e-2,
            1.0e-1,
        ]
    )

    contrasts = []

    spectra = []

    # ========================================================
    # Sweep microwave rate
    # ========================================================

    for microwave_rate in microwave_rates:

        experiment = ODMR(
            nv,

            optical_cycle=(
                optical_cycle
            ),

            microwave_rate=(
                microwave_rate
            ),

            frequency_range=(
                2.65,
                3.10,
            ),

            points=6000,

            linewidth=0.001,

            microwave_direction=(
                1.0,
                0.0,
                0.0,
            ),
        )

        result = (
            experiment.simulate()
        )

        minimum = float(
            np.min(
                result.fluorescence
            )
        )

        contrast = (
            1.0
            - minimum
        )

        contrasts.append(
            contrast
        )

        spectra.append(
            (
                microwave_rate,
                result,
            )
        )

        print(
            f"W_max = "
            f"{microwave_rate:.6g} 1/ns"
            f"    "
            f"F_min = "
            f"{minimum:.6f}"
            f"    "
            f"contrast = "
            f"{100.0 * contrast:.3f} %"
        )

    contrasts = np.asarray(
        contrasts
    )

    # ========================================================
    # Plot 1:
    # ODMR spectra for selected microwave rates
    # ========================================================

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    # Plot a subset so the figure remains readable.
    selected_indices = [
        0,
        2,
        4,
        5,
        6,
        7,
    ]

    for index in selected_indices:

        microwave_rate, result = (
            spectra[index]
        )

        plt.plot(
            result.frequencies,
            result.fluorescence,
            linewidth=1.2,
            label=(
                rf"$W_{{\max}}="
                rf"{microwave_rate:g}"
                rf"\ \mathrm{{ns^{{-1}}}}$"
            ),
        )

    plt.xlabel(
        "Microwave frequency (GHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        "Microwave-Rate Dependence of CW ODMR"
        "\n"
        r"$B=5$ mT, "
        r"$\mathbf{B}\parallel$ NV"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Plot 2:
    # Emergent contrast versus microwave rate
    # ========================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    # W=0 cannot appear on a logarithmic x-axis.
    nonzero = (
        microwave_rates > 0.0
    )

    plt.semilogx(
        microwave_rates[nonzero],
        100.0 * contrasts[nonzero],
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        r"Maximum microwave mixing rate "
        r"$W_{\max}$ (ns$^{-1}$)"
    )

    plt.ylabel(
        "ODMR contrast (%)"
    )

    plt.title(
        "Emergent ODMR Contrast"
        "\n"
        "Five-Level Rate-Equation Readout"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()