"""
Rate-equation CW-ODMR simulation for an NV- center.

The NV axis and static magnetic field are aligned.

The spin spectrum is generated from:
    - exact Hamiltonian eigenstates
    - microwave transition matrix elements
    - Lorentzian microwave linewidths

The ODMR fluorescence contrast is NOT imposed
phenomenologically.

Instead, at every microwave frequency the code:
    1. calculates the microwave spectral response,
    2. converts it into a microwave population-mixing rate,
    3. solves the five-level optical steady-state rate equations,
    4. calculates fluorescence from the radiative excited-state
       populations.
"""

import matplotlib.pyplot as plt

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
    #
    # These rates remain constant during each steady-state
    # calculation.
    #
    # The populations change dynamically / with steady state.
    #
    # The microwave mixing rate changes across the ODMR
    # frequency sweep.
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
    # ODMR experiment
    # =========================================================
    #
    # There is deliberately NO contrast parameter.
    #
    # The ODMR contrast emerges from the five-level
    # optical rate equations.
    #
    # microwave_rate sets the maximum effective
    # microwave-induced population mixing rate W_max.
    #
    # Units: 1/ns
    # =========================================================

    experiment = ODMR(
        nv,

        optical_cycle=optical_cycle,

        microwave_rate=0.01,

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

    # =========================================================
    # Run simulation
    # =========================================================

    result = experiment.simulate()

    # =========================================================
    # Print strongest transitions
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
    # Extract rate-equation ODMR contrast
    # =========================================================

    fluorescence_off = (
        result.fluorescence.max()
    )

    fluorescence_on = (
        result.fluorescence.min()
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
        r"Five-level rate-equation readout, "
        r"$B=5$ mT, $\mathbf{B}\parallel$ NV"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()