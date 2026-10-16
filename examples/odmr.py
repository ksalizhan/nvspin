"""
CW-ODMR simulation for an NV- center.

This version uses:

    - exact spin-Hamiltonian eigenstates
    - exact microwave transition matrix elements
    - a physical reference Rabi frequency
    - homogeneous transverse coherence time T2
    - Bloch-derived microwave population-transfer rates
    - five-level optical rate-equation fluorescence

There is NO:

    - fixed ODMR contrast
    - independently prescribed Lorentzian linewidth
    - independently prescribed microwave population-mixing rate

Instead,

    Omega + T2 + detuning
        -> microwave transition rate
        -> optical steady state
        -> fluorescence
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin import NVCenter
from nvspin.experiments import ODMR

from nvspin.readout import (
    NVOpticalCycle,
    NVOpticalRates,
)


def main():

    # =========================================================
    # NV system
    # =========================================================

    nv = NVCenter(
        D=2.870,
        E=0.0,
        B=[
            0.0,
            0.0,
            0.005,      # 5 mT
        ],
    )

    # =========================================================
    # Optical-cycle rates
    # =========================================================
    #
    # Units: 1/ns
    # =========================================================

    rates = NVOpticalRates(
        excitation=0.05,

        radiative_ms0=0.077,
        radiative_ms1=0.077,

        isc_ms0=0.005,
        isc_ms1=0.050,

        singlet_to_ms0=0.004,
        singlet_to_ms1=0.001,
    )

    optical_cycle = NVOpticalCycle(
        rates
    )

    # =========================================================
    # Microwave / coherence parameters
    # =========================================================
    #
    # rabi_frequency:
    #
    #     ordinary frequency in GHz
    #
    #     0.005 GHz = 5 MHz
    #
    # T2:
    #
    #     homogeneous transverse coherence time in ns
    #
    # For the current Bloch-derived response:
    #
    #     Gamma_FWHM = 1 / (pi T2)
    #
    # so T2 = 300 ns gives approximately
    #
    #     Gamma_FWHM = 1.061 MHz
    #
    # in the weak-drive limit.
    # =========================================================

    rabi_frequency = 0.005
    T2 = 300.0

    # =========================================================
    # ODMR experiment
    # =========================================================

    experiment = ODMR(
        nv,

        optical_cycle=optical_cycle,

        rabi_frequency=rabi_frequency,

        T2=T2,

        frequency_range=(
            2.65,
            3.10,
        ),

        points=6000,

        microwave_direction=(
            1.0,
            0.0,
            0.0,
        ),
    )

    # =========================================================
    # Derived weak-drive linewidth
    # =========================================================

    print()
    print(
        "Microwave / coherence parameters"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Reference Rabi frequency = "
        f"{1000.0 * rabi_frequency:.3f} MHz"
    )

    print(
        f"T2                       = "
        f"{T2:.3f} ns"
    )

    print(
        f"Weak-drive FWHM          = "
        f"{1000.0 * experiment.weak_drive_fwhm:.3f} MHz"
    )

    # =========================================================
    # Run simulation
    # =========================================================

    result = experiment.simulate()

    # =========================================================
    # Strongest transitions
    # =========================================================

    print()
    print(
        "Strongest ODMR transitions"
    )
    print(
        "--------------------------"
    )

    for transition in (
        result.strongest_transitions(
            12
        )
    ):

        print(
            f"{transition.frequency:.6f} GHz"
            f"   strength = "
            f"{transition.strength:.6f}"
        )

    # =========================================================
    # Emergent ODMR contrast
    # =========================================================

    fluorescence_off = float(
        np.max(
            result.fluorescence
        )
    )

    fluorescence_on = float(
        np.min(
            result.fluorescence
        )
    )

    contrast = (
        (
            fluorescence_off
            - fluorescence_on
        )
        / fluorescence_off
    )

    print()
    print(
        "Rate-equation ODMR result"
    )
    print(
        "-------------------------"
    )

    print(
        f"Off-resonant fluorescence = "
        f"{fluorescence_off:.6f}"
    )

    print(
        f"Minimum fluorescence      = "
        f"{fluorescence_on:.6f}"
    )

    print(
        f"Predicted ODMR contrast   = "
        f"{100.0 * contrast:.3f} %"
    )

    # =========================================================
    # Plot
    # =========================================================

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    plt.plot(
        result.frequencies,
        result.fluorescence,
        color="black",
        linewidth=1.4,
    )

    plt.xlabel(
        "Microwave frequency (GHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        r"NV$^-$ CW-ODMR"
        "\n"
        r"$B=5$ mT, "
        r"$\mathbf{B}\parallel$ NV, "
        r"$\Omega/2\pi=5$ MHz"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()