"""
NV- + 14N ground-state Zeeman / hyperfine energy levels.

Includes:
    - NV electronic spin S = 1
    - intrinsic 14N nuclear spin I = 1
    - zero-field splitting
    - electron Zeeman interaction
    - nuclear Zeeman interaction
    - 14N hyperfine interaction
    - 14N quadrupole interaction

Magnetic-field angles:
    theta = 0, 5, 15, 30, 45, 90 degrees

Magnetic-field range:
    0 to 200 mT

The magnetic field lies in the x-z plane:

    Bx = B sin(theta)
    By = 0
    Bz = B cos(theta)
"""

import numpy as np
import matplotlib.pyplot as plt

from nvspin import NVCenter
from nvspin.experiments import EnergyLevels


def main():

    # ---------------------------------------------------------
    # NV- with intrinsic 14N
    #
    # Nitrogen14 is the default nitrogen isotope in NVCenter.
    # ---------------------------------------------------------

    nv = NVCenter(
        D=2.870,
        E=0.0,
    )

    # ---------------------------------------------------------
    # Magnetic-field orientations
    # ---------------------------------------------------------

    theta_values = [
        0,
        5,
        15,
        30,
        45,
        90,
    ]

    colors = {
        0: "black",
        5: "tab:blue",
        15: "tab:green",
        30: "tab:orange",
        45: "tab:red",
        90: "tab:purple",
    }

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    # ---------------------------------------------------------
    # Sweep field orientations
    # ---------------------------------------------------------

    for theta_deg in theta_values:

        theta = np.deg2rad(
            theta_deg
        )

        direction = (
            np.sin(theta),
            0.0,
            np.cos(theta),
        )

        experiment = EnergyLevels(
            nv,
            B_min=0.0,
            B_max=0.200,
            points=1200,
            direction=direction,
        )

        result = experiment.simulate()

        B_mT = (
            result.magnetic_fields
            * 1e3
        )

        # Actual eigenenergies.
        #
        # NV S=1 -> dimension 3
        # 14N I=1 -> dimension 3
        #
        # Total dimension = 9.

        energies = result.energies

        color = colors[
            theta_deg
        ]

        # -----------------------------------------------------
        # Plot all 9 eigenenergy branches
        # -----------------------------------------------------

        for state_index in range(
            energies.shape[1]
        ):

            plt.plot(
                B_mT,
                energies[:, state_index],
                color=color,
                linewidth=1.0,
                alpha=0.85,
            )

        # -----------------------------------------------------
        # One legend entry per field orientation
        # -----------------------------------------------------

        plt.plot(
            [],
            [],
            color=color,
            linewidth=1.8,
            label=rf"$\theta={theta_deg}^\circ$",
        )

    # ---------------------------------------------------------
    # Bare NV zero-field splitting
    # ---------------------------------------------------------

    plt.axhline(
        nv.D,
        color="gray",
        linestyle=":",
        linewidth=1.0,
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    plt.xlabel(
        "Magnetic field (mT)"
    )

    plt.ylabel(
        "Eigenenergy (GHz)"
    )

    plt.title(
        r"NV$^-$ + $^{14}$N Ground-State "
        r"Zeeman / Hyperfine Structure"
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend(
        title="Field angle"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()