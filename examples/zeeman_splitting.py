"""
Pure NV- electronic Zeeman splitting for different
magnetic-field misalignment angles.

The magnetic field lies in the x-z plane:

    Bx = B sin(theta)
    By = 0
    Bz = B cos(theta)

Angles:
    0, 5, 15, 30, 45, 90 degrees

No nitrogen, carbon-13, strain, or other interactions
are included.

For theta = 0 degrees:

    f_- = D - gamma_e B
    f_+ = D + gamma_e B

For theta != 0, the transverse magnetic-field component
mixes the NV ms states, so the transition frequencies are
obtained by diagonalizing the full Hamiltonian.
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
    # Magnetic-field misalignment angles
    # ---------------------------------------------------------

    theta_values = [
        0,
        5,
        15,
        30,
        45,
        90,
    ]

    # One color per angle.
    #
    # Both Zeeman branches belonging to the same angle use
    # the same color.
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
        figsize=(9, 6.5)
    )

    # ---------------------------------------------------------
    # Sweep field orientation
    # ---------------------------------------------------------

    for theta_deg in theta_values:

        theta = np.deg2rad(
            theta_deg
        )

        # Magnetic field lies in x-z plane:
        #
        # Bx = B sin(theta)
        # Bz = B cos(theta)
        #
        # EnergyLevels normalizes this direction vector.

        direction = (
            np.sin(theta),
            0.0,
            np.cos(theta),
        )

        experiment = EnergyLevels(
            nv,
            B_min=0.0,
            B_max=0.020,
            points=300,
            direction=direction,
        )

        result = experiment.simulate()

        transitions = (
            result.transition_energies
        )

        B_mT = (
            result.magnetic_fields
            * 1e3
        )

        color = colors[
            theta_deg
        ]

        # -----------------------------------------------------
        # Lower branch
        # -----------------------------------------------------

        plt.plot(
            B_mT,
            transitions[:, 1],
            color=color,
            linewidth=1.5,
        )

        # -----------------------------------------------------
        # Upper branch
        # -----------------------------------------------------

        plt.plot(
            B_mT,
            transitions[:, 2],
            color=color,
            linewidth=1.5,
        )

        # -----------------------------------------------------
        # One legend entry per angle
        # -----------------------------------------------------

        plt.plot(
            [],
            [],
            color=color,
            linewidth=1.8,
            label=rf"$\theta={theta_deg}^\circ$",
        )

    # ---------------------------------------------------------
    # Zero-field splitting
    # ---------------------------------------------------------

    plt.axhline(
        nv.D,
        color="gray",
        linewidth=1.0,
        linestyle=":",
        label=r"$D$",
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    plt.xlabel(
        "Magnetic field (mT)"
    )

    plt.ylabel(
        "Spin-transition frequency (GHz)"
    )

    plt.title(
        r"NV$^-$ Ground-State Zeeman Splitting "
        r"vs Magnetic-Field Misalignment"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()