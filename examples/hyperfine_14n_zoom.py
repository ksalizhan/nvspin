"""
NV- + 14N hyperfine structure.

Zoomed view of the ms ~ +/-1 electronic manifolds.

Includes:
    - NV electronic spin S = 1
    - intrinsic 14N nuclear spin I = 1
    - zero-field splitting
    - electron Zeeman interaction
    - nuclear Zeeman interaction
    - 14N hyperfine interaction
    - 14N quadrupole interaction

Magnetic-field orientations:
    theta = 0, 45, 90 degrees

Color convention:
    theta = 0 deg   -> black
    theta = 45 deg  -> red
    theta = 90 deg  -> blue

Magnetic-field range:
    0 to 20 mT

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

    # =========================================================
    # NV- + intrinsic 14N
    # =========================================================

    nv = NVCenter(
        D=2.870,
        E=0.0,
    )

    # =========================================================
    # Magnetic-field orientations
    # =========================================================

    theta_values = [
        0,
        45,
        90,
    ]

    colors = {
        0: "black",
        45: "red",
        90: "blue",
    }

    # =========================================================
    # Figure
    # =========================================================

    plt.figure(
        figsize=(10, 7)
    )

    # =========================================================
    # Sweep field orientations
    # =========================================================

    for theta_deg in theta_values:

        theta = np.deg2rad(
            theta_deg
        )

        # -----------------------------------------------------
        # Magnetic-field direction
        #
        # Field lies in the x-z plane:
        #
        # Bx = B sin(theta)
        # By = 0
        # Bz = B cos(theta)
        # -----------------------------------------------------

        direction = (
            np.sin(theta),
            0.0,
            np.cos(theta),
        )

        # -----------------------------------------------------
        # Energy-level calculation
        # -----------------------------------------------------

        experiment = EnergyLevels(
            nv,
            B_min=0.0,
            B_max=0.020,      # 20 mT
            points=1000,
            direction=direction,
        )

        result = experiment.simulate()

        # Convert Tesla -> mT
        B_mT = (
            result.magnetic_fields
            * 1e3
        )

        # Actual eigenenergies in GHz
        energies = result.energies

        color = colors[
            theta_deg
        ]

        # =====================================================
        # Select ms ~ +/-1 manifolds
        # =====================================================
        #
        # Total Hilbert-space dimension:
        #
        # NV electron:
        #     S = 1  -> 3 states
        #
        # 14N:
        #     I = 1  -> 3 states
        #
        # Total:
        #
        #     3 x 3 = 9 states
        #
        # At these relatively low magnetic fields, the lowest
        # three eigenstates are predominantly associated with
        # the ms ~ 0 manifold.
        #
        # The upper six states are predominantly associated
        # with the ms ~ -1 and ms ~ +1 manifolds.
        #
        # NOTE:
        # This energy-ordering selection is currently a
        # visualization shortcut. Later nvspin will classify
        # states using their electronic spin character.
        # =====================================================

        upper_manifolds = energies[:, 3:]

        # -----------------------------------------------------
        # Plot all six hyperfine branches
        # -----------------------------------------------------

        for state_index in range(
            upper_manifolds.shape[1]
        ):

            plt.plot(
                B_mT,
                upper_manifolds[:, state_index],
                color=color,
                linewidth=1.2,
                alpha=0.90,
            )

        # -----------------------------------------------------
        # Single legend entry for each field orientation
        # -----------------------------------------------------

        plt.plot(
            [],
            [],
            color=color,
            linewidth=1.8,
            label=rf"$\theta={theta_deg}^\circ$",
        )

    # =========================================================
    # Bare NV zero-field splitting
    # =========================================================

    plt.axhline(
        nv.D,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label=r"$D$",
    )

    # =========================================================
    # Plot formatting
    # =========================================================

    plt.xlabel(
        "Magnetic field (mT)",
        fontsize=12,
    )

    plt.ylabel(
        "Eigenenergy (GHz)",
        fontsize=12,
    )

    plt.title(
        r"NV$^-$ + $^{14}$N Hyperfine Structure "
        r"($m_s \approx \pm1$)",
        fontsize=14,
    )

    plt.xlim(
        0,
        20,
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend(
        title="Field orientation"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()