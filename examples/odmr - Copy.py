"""
CW-ODMR simulation for an NV- center with intrinsic 14N.

The NV axis and static magnetic field are aligned.

The spectrum is generated from:
    - exact Hamiltonian eigenstates
    - microwave transition matrix elements
    - phenomenological Lorentzian linewidths
"""

import matplotlib.pyplot as plt

from nvspin import NVCenter
from nvspin.experiments import ODMR


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
    # ODMR experiment
    # =========================================================

    experiment = ODMR(
        nv,
        frequency_range=(
            2.65,
            3.10,
        ),
        points=6000,
        linewidth=0.001,
        contrast=0.03,
        microwave_direction=(
            1,
            0,
            0,
        ),
    )

    result = experiment.simulate()

    # =========================================================
    # Print strongest transitions
    # =========================================================

    print()
    print("Strongest ODMR transitions")
    print("--------------------------")

    for transition in result.strongest_transitions(
        12
    ):

        print(
            f"{transition.frequency:.6f} GHz"
            f"   strength = "
            f"{transition.strength:.6f}"
        )

    # =========================================================
    # Plot
    # =========================================================

    plt.figure(
        figsize=(9, 6)
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
        r"NV$^-$ + $^{14}$N CW-ODMR"
        "\n"
        r"$B=5$ mT, $B\parallel$ NV"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()