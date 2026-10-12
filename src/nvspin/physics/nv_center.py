"""NV-center spin-system model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .nuclei import (
    Carbon13,
    Nitrogen14,
    Nitrogen15,
    NuclearSpin,
)


Vector3 = NDArray[np.float64]


@dataclass
class NVCenter:
    """
    Ground-state NV- spin system.

    Parameters
    ----------
    D
        Zero-field splitting in GHz.

    E
        Transverse zero-field splitting / strain parameter
        in GHz.

    B
        Magnetic-field vector [Bx, By, Bz] in Tesla.

    nitrogen
        Intrinsic nitrogen isotope:
        "14N", "15N", or None.
    """

    D: float = 2.870
    E: float = 0.0

    B: Vector3 = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=float,
        )
    )

    nitrogen: NuclearSpin | None = field(
        default_factory=Nitrogen14
    )

    carbons: list[Carbon13] = field(
        default_factory=list
    )

    gamma_e: float = 28.025

    def __post_init__(self) -> None:
        self.B = np.asarray(
            self.B,
            dtype=float,
        )

        if self.B.shape != (3,):
            raise ValueError(
                "Magnetic field B must contain [Bx, By, Bz]."
            )

        if self.D <= 0:
            raise ValueError(
                "Zero-field splitting D must be positive."
            )

    @property
    def nuclei(self) -> list[NuclearSpin]:
        """Return all nuclei in Hilbert-space order."""

        result: list[NuclearSpin] = []

        if self.nitrogen is not None:
            result.append(self.nitrogen)

        result.extend(self.carbons)

        return result

    @property
    def dimension(self) -> int:
        """
        Return total Hilbert-space dimension.

        NV electronic spin contributes dimension 3.
        """

        dimension = 3

        for nucleus in self.nuclei:
            dimension *= nucleus.dimension

        return dimension

    @property
    def magnetic_field_magnitude(self) -> float:
        """Return |B| in Tesla."""

        return float(
            np.linalg.norm(self.B)
        )

    def set_magnetic_field(
        self,
        Bx: float,
        By: float,
        Bz: float,
    ) -> None:
        """Set magnetic field in Tesla."""

        self.B = np.array(
            [Bx, By, Bz],
            dtype=float,
        )

    def set_nitrogen(
        self,
        isotope: str | None,
    ) -> NuclearSpin | None:
        """
        Set the intrinsic nitrogen isotope.

        Parameters
        ----------
        isotope
            "14N", "15N", or None.
        """

        if isotope is None:
            self.nitrogen = None

        elif isotope == "14N":
            self.nitrogen = Nitrogen14()

        elif isotope == "15N":
            self.nitrogen = Nitrogen15()

        else:
            raise ValueError(
                "nitrogen isotope must be '14N', '15N', or None."
            )

        return self.nitrogen

    def add_carbon13(
        self,
        A_parallel: float = 0.0,
        A_perp: float = 0.0,
        *,
        A=None,
        label: str | None = None,
    ) -> Carbon13:
        """
        Add a carbon-13 nuclear spin.

        Hyperfine parameters are in GHz.
        """

        if label is None:
            label = f"13C-{len(self.carbons) + 1}"

        carbon = Carbon13(
            A_parallel=A_parallel,
            A_perp=A_perp,
            A=A,
            label=label,
        )

        self.carbons.append(carbon)

        return carbon

    def remove_carbon13(
        self,
        carbon: Carbon13,
    ) -> None:
        """Remove a carbon-13 spin."""

        self.carbons.remove(carbon)

    def clear_carbons(self) -> None:
        """Remove all carbon-13 spins."""

        self.carbons.clear()

    def summary(self) -> str:
        """Return a short human-readable system description."""

        nitrogen = (
            self.nitrogen.label
            if self.nitrogen is not None
            else "none"
        )

        return (
            "NVCenter("
            f"D={self.D:.6f} GHz, "
            f"E={self.E:.6f} GHz, "
            f"B={self.B.tolist()} T, "
            f"nitrogen={nitrogen}, "
            f"13C={len(self.carbons)}, "
            f"dimension={self.dimension}"
            ")"
        )