"""
Pure NV- ground-state Zeeman energy levels.

Plots all three electronic energy levels for different
magnetic-field misalignment angles:

    theta = 0, 5, 15, 30, 45, 90 degrees

The magnetic field lies in the x-z plane:

    Bx = B sin(theta)
    By = 0
    Bz = B cos(theta)

Field range:
    0 to 200 mT

No nitrogen, 13C, strain, or other interactions are included.
"""

import numpy as np
import matplotlib.pyplot as plt

from nvspin import NVCenter
from nvspin.experiments import EnergyLevels


def main():

    # ---------------------------------------------------------
    # Pure NV electronic spin
    # ---------------------------------------------------------

    nv = NVCenter(
        D=2.870,
        E=0.0,
        nitrogen=None,
    )

    # ---------------------------------------------------------
    # Field misalignment angles
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
        figsize=(9, 7)
    )

    # ---------------------------------------------------------
    # Sweep orientations
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
            B_max=0.200,      # 200 mT
            points=1000,
            direction=direction,
        )

        result = experiment.simulate()

        B_mT = (
            result.magnetic_fields
            * 1e3
        )

        # IMPORTANT:
        #
        # Use the actual eigenenergies here,
        # NOT result.transition_energies.

        energies = result.energies

        color = colors[
            theta_deg
        ]

        # -----------------------------------------------------
        # Plot all three electronic eigenenergies
        # -----------------------------------------------------

        for state_index in range(3):

            plt.plot(
                B_mT,
                energies[:, state_index],
                color=color,
                linewidth=1.4,
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
    # Zero-field splitting reference
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
        r"NV$^-$ Ground-State Zeeman Energy Levels"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        title="Field angle"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()