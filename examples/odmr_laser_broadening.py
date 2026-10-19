"""
Laser-induced broadening diagnostic for seven-variable CW ODMR.

We hold fixed:

    B = 5 mT, parallel to NV
    T2 = 300 ns
    reference Rabi frequency = 5 MHz

and vary only the optical excitation rate k_ex.

The seven-variable model predicts

    gamma_2 = 1/T2 + k_ex

and therefore, in the weak-microwave limit,

    Gamma_weak = gamma_2 / pi.

The purpose is to determine whether the disappearance of the
14N hyperfine triplet is caused by optical broadening.
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
# Fixed physical parameters
# ============================================================

T2 = 300.0

RABI_FREQUENCY = 0.005  # GHz = 5 MHz

EXCITATION_RATES = [
    0.050,
    0.010,
    0.003,
    0.001,
]


# ============================================================
# Optical cycle
# ============================================================


def build_optical_cycle(
    excitation_rate,
):

    rates = NVOpticalRates(
        excitation=excitation_rate,

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

    # --------------------------------------------------------
    # NV + intrinsic 14N
    # --------------------------------------------------------

    nv = NVCenter(
        D=2.870,
        E=0.0,
        B=[
            0.0,
            0.0,
            0.005,
        ],
    )

    results = []

    # --------------------------------------------------------
    # Sweep laser excitation rate
    # --------------------------------------------------------

    print()
    print(
        "Laser-broadening diagnostic"
    )
    print(
        "==========================="
    )

    print(
        f"T2 = {T2:.1f} ns"
    )

    print(
        f"Reference Rabi frequency = "
        f"{1000.0 * RABI_FREQUENCY:.3f} MHz"
    )

    print()

    for k_ex in EXCITATION_RATES:

        optical_cycle = (
            build_optical_cycle(
                k_ex
            )
        )

        experiment = ODMR(
            nv,

            optical_cycle=(
                optical_cycle
            ),

            rabi_frequency=(
                RABI_FREQUENCY
            ),

            T2=T2,

            frequency_range=(
                2.65,
                3.10,
            ),

            points=8000,

            microwave_direction=(
                1.0,
                0.0,
                0.0,
            ),
        )

        result = (
            experiment.simulate()
        )

        theoretical_width = (
            experiment.weak_drive_fwhm
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

        results.append(
            (
                k_ex,
                experiment,
                result,
            )
        )

        print(
            f"k_ex = "
            f"{k_ex:7.4f} ns^-1"
            f"    "
            f"T2_eff = "
            f"{experiment.effective_T2:8.3f} ns"
            f"    "
            f"Gamma_weak = "
            f"{1000.0 * theoretical_width:8.3f} MHz"
            f"    "
            f"contrast = "
            f"{100.0 * contrast:7.3f} %"
        )

    # --------------------------------------------------------
    # Print actual Hamiltonian transitions
    # --------------------------------------------------------

    transitions = (
        results[0][2]
        .strongest_transitions(
            12
        )
    )

    print()
    print(
        "Strongest Hamiltonian transitions"
    )
    print(
        "================================="
    )

    for transition in transitions:

        print(
            f"{transition.frequency:.9f} GHz"
            f"    "
            f"strength = "
            f"{transition.strength:.6f}"
            f"    "
            f"weight = "
            f"{transition.population_weight:.6f}"
        )

    # --------------------------------------------------------
    # Plot full ODMR spectra
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for (
        k_ex,
        experiment,
        result,
    ) in results:

        plt.plot(
            result.frequencies,
            result.fluorescence,
            linewidth=1.4,
            label=(
                rf"$k_{{ex}}="
                rf"{k_ex:g}"
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
        "Laser-Induced Broadening of CW ODMR"
        "\n"
        r"$B=5$ mT, "
        r"$T_2=300$ ns, "
        r"$\Omega/2\pi=5$ MHz"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Zoom around lower-frequency hyperfine group
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for (
        k_ex,
        experiment,
        result,
    ) in results:

        plt.plot(
            result.frequencies,
            result.fluorescence,
            linewidth=1.4,
            label=(
                rf"$k_{{ex}}="
                rf"{k_ex:g}"
                rf"\ \mathrm{{ns^{{-1}}}}$"
            ),
        )

    # Mark Hamiltonian transition frequencies.
    #
    # Only show transitions in the lower ODMR branch.
    for transition in transitions:

        if (
            2.70
            < transition.frequency
            < 2.76
        ):

            plt.axvline(
                transition.frequency,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

    plt.xlim(
        2.715,
        2.745,
    )

    plt.xlabel(
        "Microwave frequency (GHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        r"Resolving the $^{14}$N Hyperfine Structure"
        "\n"
        "Laser Excitation-Rate Sweep"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Predicted weak-drive linewidth versus laser rate
    # --------------------------------------------------------

    excitation_array = np.asarray(
        EXCITATION_RATES,
        dtype=float,
    )

    linewidths = np.asarray(
        [
            experiment.weak_drive_fwhm
            for (
                _,
                experiment,
                _,
            )
            in results
        ]
    )

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        excitation_array,
        1000.0 * linewidths,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        r"Excitation rate "
        r"$k_{\rm ex}$ (ns$^{-1}$)"
    )

    plt.ylabel(
        "Weak-drive homogeneous FWHM (MHz)"
    )

    plt.title(
        "Optical Contribution to ODMR Linewidth"
        "\n"
        r"$\Gamma_{\rm weak}"
        r"=(1/T_2+k_{\rm ex})/\pi$"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()