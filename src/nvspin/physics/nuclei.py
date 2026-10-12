"""Nuclear-spin models for NV-center simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


HyperfineTensor = NDArray[np.float64]


@dataclass
class NuclearSpin:
    """
    Base class describing a nuclear spin coupled to an NV center.

    All frequency-like quantities are stored in GHz.

    Parameters
    ----------
    spin
        Nuclear spin quantum number I.

    gamma
        Nuclear gyromagnetic ratio in GHz/T.

    A
        3x3 hyperfine tensor in GHz.

    quadrupole
        Axial quadrupole coupling in GHz.

    label
        Human-readable identifier.
    """

    spin: float
    gamma: float
    A: HyperfineTensor
    quadrupole: float = 0.0
    label: str = "nucleus"

    def __post_init__(self) -> None:
        self.A = np.asarray(
            self.A,
            dtype=float,
        )

        if self.A.shape != (3, 3):
            raise ValueError(
                "Hyperfine tensor A must have shape (3, 3)."
            )

        if self.spin <= 0:
            raise ValueError(
                "Nuclear spin must be positive."
            )

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension 2I + 1."""

        return int(round(2 * self.spin + 1))

    @property
    def A_parallel(self) -> float:
        """Return Azz in GHz."""

        return float(self.A[2, 2])

    @property
    def A_perp(self) -> float:
        """
        Return the mean transverse diagonal coupling.

        For an axial tensor this is simply Axx = Ayy.
        """

        return float(
            0.5 * (
                self.A[0, 0]
                + self.A[1, 1]
            )
        )

    def set_axial_hyperfine(
        self,
        A_parallel: float,
        A_perp: float,
    ) -> None:
        """
        Replace the hyperfine tensor with an axial tensor.

        Parameters are in GHz.
        """

        self.A = np.diag(
            [
                A_perp,
                A_perp,
                A_parallel,
            ]
        ).astype(float)


def axial_hyperfine_tensor(
    A_parallel: float,
    A_perp: float,
) -> HyperfineTensor:
    """Construct an axial hyperfine tensor in GHz."""

    return np.diag(
        [
            A_perp,
            A_perp,
            A_parallel,
        ]
    ).astype(float)


class Carbon13(NuclearSpin):
    """Carbon-13 nuclear spin."""

    def __init__(
        self,
        A_parallel: float = 0.0,
        A_perp: float = 0.0,
        *,
        A: HyperfineTensor | None = None,
        label: str = "13C",
    ) -> None:

        if A is None:
            A = axial_hyperfine_tensor(
                A_parallel,
                A_perp,
            )

        super().__init__(
            spin=0.5,
            gamma=10.705e-3,
            A=A,
            quadrupole=0.0,
            label=label,
        )


class Nitrogen14(NuclearSpin):
    """Intrinsic nitrogen-14 nuclear spin."""

    def __init__(
        self,
        A_parallel: float = -2.14e-3,
        A_perp: float = -2.70e-3,
        quadrupole: float = -4.95e-3,
        *,
        A: HyperfineTensor | None = None,
        label: str = "14N",
    ) -> None:

        if A is None:
            A = axial_hyperfine_tensor(
                A_parallel,
                A_perp,
            )

        super().__init__(
            spin=1.0,
            gamma=3.077e-3,
            A=A,
            quadrupole=quadrupole,
            label=label,
        )


class Nitrogen15(NuclearSpin):
    """Intrinsic nitrogen-15 nuclear spin."""

    def __init__(
        self,
        A_parallel: float = 3.03e-3,
        A_perp: float = 3.65e-3,
        *,
        A: HyperfineTensor | None = None,
        label: str = "15N",
    ) -> None:

        if A is None:
            A = axial_hyperfine_tensor(
                A_parallel,
                A_perp,
            )

        super().__init__(
            spin=0.5,
            gamma=-4.316e-3,
            A=A,
            quadrupole=0.0,
            label=label,
        )