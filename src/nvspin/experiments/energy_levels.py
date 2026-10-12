"""Energy-level calculations for NV-center spin systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvspin.physics.hamiltonian import eigensystem
from nvspin.physics.nv_center import NVCenter


FloatArray = NDArray[np.float64]


@dataclass
class EnergyLevelResult:
    """Result of an energy-level calculation."""

    magnetic_fields: FloatArray
    energies: FloatArray

    @property
    def transition_energies(self) -> FloatArray:
        """
        Energies relative to the lowest-energy state
        at each magnetic-field point.
        """

        return self.energies - self.energies[:, [0]]


class EnergyLevels:
    """
    Calculate NV-system energy levels versus magnetic field.

    The field direction is kept fixed while its magnitude
    is swept.
    """

    def __init__(
        self,
        system: NVCenter,
        B_min: float = 0.0,
        B_max: float = 0.010,
        points: int = 200,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        if B_max < B_min:
            raise ValueError(
                "B_max must be greater than or equal to B_min."
            )

        direction_array = np.asarray(
            direction,
            dtype=float,
        )

        if direction_array.shape != (3,):
            raise ValueError(
                "direction must contain three components."
            )

        norm = np.linalg.norm(direction_array)

        if np.isclose(norm, 0.0):
            raise ValueError(
                "direction must be non-zero."
            )

        self.system = system
        self.B_min = float(B_min)
        self.B_max = float(B_max)
        self.points = int(points)

        self.direction = (
            direction_array / norm
        )

    def simulate(self) -> EnergyLevelResult:
        """Run the magnetic-field sweep."""

        magnetic_fields = np.linspace(
            self.B_min,
            self.B_max,
            self.points,
        )

        original_B = self.system.B.copy()

        all_energies = []

        try:
            for B_magnitude in magnetic_fields:

                self.system.B = (
                    B_magnitude
                    * self.direction
                )

                energies, _ = eigensystem(
                    self.system
                )

                all_energies.append(
                    energies
                )

        finally:
            # Do not unexpectedly modify the user's NV object.
            self.system.B = original_B

        return EnergyLevelResult(
            magnetic_fields=magnetic_fields,
            energies=np.asarray(
                all_energies,
                dtype=float,
            ),
        )