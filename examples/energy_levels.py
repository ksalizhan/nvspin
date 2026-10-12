"""Example: NV- ground-state energy levels."""

import matplotlib.pyplot as plt

from nvspin import NVCenter
from nvspin.experiments import EnergyLevels


def main():

    nv = NVCenter(
        D=2.870,
        E=0.0,
        nitrogen=None,
    )

    directions = {
        "B parallel to NV": (0, 0, 1),
        "45 degrees": (1, 0, 1),
        "B perpendicular to NV": (1, 0, 0),
    }

    plt.figure(figsize=(9, 6))

    for label, direction in directions.items():

        experiment = EnergyLevels(
            nv,
            B_min=0.0,
            B_max=0.010,
            points=300,
            direction=direction,
        )

        result = experiment.simulate()

        transitions = result.transition_energies

        # Skip column 0 because it is the reference ground state.
        for index in range(1, transitions.shape[1]):

            plt.plot(
                result.magnetic_fields * 1e3,
                transitions[:, index],
                label=label if index == 1 else None,
            )

    plt.xlabel("Magnetic field (mT)")
    plt.ylabel("Energy relative to ground state (GHz)")
    plt.title("NV- Ground-State Energy Levels")

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()