"""
Microwave-power broadening of CW ODMR.

Diagnostic experiment for the seven-variable optical-Bloch
ODMR model.

We hold fixed:

    B = 5 mT, parallel to NV
    T2 = 300 ns
    k_ex = 0.001 ns^-1

and sweep the reference Rabi frequency:

    Omega / (2*pi)
        =
        0.1, 0.3, 0.5, 1, 2, 5 MHz.

Purpose
-------
Determine whether the 14N hyperfine triplet becomes resolved
at weak microwave drive and progressively merges as microwave
power increases.

This separates microwave power broadening from the
laser-induced broadening investigated separately.
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
# Fixed parameters
# ============================================================

T2 = 300.0

K_EX = 0.001

RABI_FREQUENCIES = np.array(
    [
        0.0001,  # 0.1 MHz
        0.0003,  # 0.3 MHz
        0.0005,  # 0.5 MHz
        0.0010,  # 1.0 MHz
        0.0020,  # 2.0 MHz
        0.0050,  # 5.0 MHz
    ]
)


# ============================================================
# Optical cycle
# ============================================================


def build_optical_cycle():

    rates = NVOpticalRates(
        excitation=K_EX,

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
    # NV + intrinsic 14N
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

    results = []

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "Microwave-power broadening diagnostic"
    )
    print(
        "====================================="
    )

    print(
        f"k_ex = {K_EX:.6f} ns^-1"
    )

    print(
        f"T2   = {T2:.3f} ns"
    )

    print()

    # ========================================================
    # Rabi-frequency sweep
    # ========================================================

    for rabi_frequency in RABI_FREQUENCIES:

        experiment = ODMR(
            nv,

            optical_cycle=optical_cycle,

            rabi_frequency=float(
                rabi_frequency
            ),

            T2=T2,

            # Focus calculation around lower-frequency
            # ms=0 -> ms=-1 branch.
            frequency_range=(
                2.715,
                2.745,
            ),

            # Fine sampling is important because we are
            # explicitly looking for MHz-scale structure.
            points=6000,

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

        results.append(
            (
                rabi_frequency,
                experiment,
                result,
            )
        )

        print(
            f"Omega/2pi = "
            f"{1000.0 * rabi_frequency:6.3f} MHz"
            f"    "
            f"T2_eff = "
            f"{experiment.effective_T2:8.3f} ns"
            f"    "
            f"Gamma_weak = "
            f"{1000.0 * experiment.weak_drive_fwhm:6.3f} MHz"
            f"    "
            f"contrast = "
            f"{100.0 * contrast:7.3f} %"
        )

    # ========================================================
    # Exact Hamiltonian transition frequencies
    # ========================================================

    transitions = (
        results[0][2]
        .strongest_transitions(
            20
        )
    )

    lower_transitions = [
        transition
        for transition in transitions
        if (
            2.715
            < transition.frequency
            < 2.745
        )
    ]

    lower_transitions = sorted(
        lower_transitions,
        key=lambda transition:
        transition.frequency,
    )

    print()
    print(
        "Lower-branch Hamiltonian transitions"
    )
    print(
        "===================================="
    )

    for transition in lower_transitions:

        print(
            f"{transition.frequency:.9f} GHz"
            f"    "
            f"strength = "
            f"{transition.strength:.6f}"
            f"    "
            f"weight = "
            f"{transition.population_weight:.6f}"
        )

    # ========================================================
    # Hyperfine splittings
    # ========================================================

    if len(
        lower_transitions
    ) >= 2:

        print()
        print(
            "Adjacent transition separations"
        )
        print(
            "==============================="
        )

        for (
            left,
            right,
        ) in zip(
            lower_transitions[:-1],
            lower_transitions[1:],
        ):

            splitting_ghz = (
                right.frequency
                - left.frequency
            )

            splitting_mhz = (
                1000.0
                * splitting_ghz
            )

            print(
                f"{splitting_mhz:.6f} MHz"
            )

    # ========================================================
    # Figure 1:
    # All Rabi frequencies
    # ========================================================

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for (
        rabi_frequency,
        experiment,
        result,
    ) in results:

        plt.plot(
            result.frequencies,
            result.fluorescence,
            linewidth=1.4,
            label=(
                rf"$\Omega/2\pi="
                rf"{1000.0 * rabi_frequency:g}$ MHz"
            ),
        )

    # Exact Hamiltonian transition positions
    for transition in lower_transitions:

        plt.axvline(
            transition.frequency,
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
        )

    plt.xlabel(
        "Microwave frequency (GHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        r"Microwave Power Broadening of "
        r"$^{14}$N Hyperfine ODMR"
        "\n"
        r"$B=5$ mT, "
        r"$k_{\rm ex}=0.001$ ns$^{-1}$, "
        r"$T_2=300$ ns"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 2:
    # Weak-drive spectra only
    # ========================================================

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for (
        rabi_frequency,
        experiment,
        result,
    ) in results:

        if (
            rabi_frequency
            > 0.0005
        ):
            continue

        plt.plot(
            result.frequencies,
            result.fluorescence,
            linewidth=1.5,
            label=(
                rf"$\Omega/2\pi="
                rf"{1000.0 * rabi_frequency:g}$ MHz"
            ),
        )

    for transition in lower_transitions:

        plt.axvline(
            transition.frequency,
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
        )

    plt.xlabel(
        "Microwave frequency (GHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        r"Weak-Drive $^{14}$N Hyperfine Resolution"
        "\n"
        r"$\Omega/2\pi \leq 0.5$ MHz"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 3:
    # Contrast versus Rabi frequency
    # ========================================================

    rabi_mhz = []

    contrasts = []

    for (
        rabi_frequency,
        experiment,
        result,
    ) in results:

        rabi_mhz.append(
            1000.0
            * rabi_frequency
        )

        contrasts.append(
            100.0
            * (
                1.0
                - np.min(
                    result.fluorescence
                )
            )
        )

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.semilogx(
        rabi_mhz,
        contrasts,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        r"Reference Rabi frequency "
        r"$\Omega/2\pi$ (MHz)"
    )

    plt.ylabel(
        "ODMR contrast (%)"
    )

    plt.title(
        "ODMR Saturation versus Microwave Drive"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()