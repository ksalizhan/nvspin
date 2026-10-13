"""Dynamical-decoupling spectroscopy of a single nuclear spin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# Pauli operators for the nuclear spin.
#
# We use I_i = sigma_i / 2 so frequencies in GHz can be
# propagated with exp(-i 2*pi H t), with t in ns.

SIGMA_X = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ],
    dtype=complex,
)

SIGMA_Z = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0],
    ],
    dtype=complex,
)

I_X = 0.5 * SIGMA_X
I_Z = 0.5 * SIGMA_Z

IDENTITY = np.eye(
    2,
    dtype=complex,
)


@dataclass(frozen=True)
class SingleCarbon13:
    """
    Effective weakly coupled carbon-13 spin.

    Parameters
    ----------
    larmor_frequency
        Bare 13C Larmor frequency in GHz.

    A_parallel
        Longitudinal hyperfine coupling in GHz.

    A_perpendicular
        Transverse hyperfine coupling in GHz.
    """

    larmor_frequency: float

    A_parallel: float

    A_perpendicular: float


@dataclass
class DDSpectroscopyResult:
    """Result of a single-13C DD spectroscopy scan."""

    pulse_spacings: FloatArray

    coherence: FloatArray

    preservation_probability: FloatArray


def unitary(
    hamiltonian: ComplexArray,
    duration: float,
) -> ComplexArray:
    """
    Exact unitary evolution for a 2x2 Hermitian Hamiltonian.

    Frequencies are in GHz and duration is in ns.
    """

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            hamiltonian
        )
    )

    phases = np.exp(
        -1j
        * 2.0
        * np.pi
        * eigenvalues
        * duration
    )

    return (
        eigenvectors
        @ np.diag(phases)
        @ eigenvectors.conj().T
    )


class SingleCarbonDDSpectroscopy:
    """
    Ideal DD spectroscopy for one weakly coupled 13C.

    The NV electron is represented by the two states

        |0> = |ms = 0>
        |1> = |ms = -1>

    and the nuclear Hamiltonian is conditional on the
    electron state.

    Pulse-spacing convention
    ------------------------
    pulse_spacing is the center-to-center separation between
    consecutive ideal electron pi pulses.

    For N pulses, the free-evolution timing is

        spacing/2
        pi
        spacing
        pi
        ...
        pi
        spacing/2

    so the total ideal free-evolution time is

        T = N * pulse_spacing.

    Notes
    -----
    Electron pi pulses are ideal and instantaneous.

    The nuclear spin is initially unpolarized:

        rho_n = I / 2.
    """

    def __init__(
        self,
        carbon: SingleCarbon13,
        number_of_pulses: int,
        spacing_range: tuple[float, float],
        points: int = 1000,
    ) -> None:

        if number_of_pulses < 1:
            raise ValueError(
                "number_of_pulses must be at least 1."
            )

        spacing_min, spacing_max = (
            spacing_range
        )

        if spacing_min <= 0:
            raise ValueError(
                "Minimum pulse spacing must be positive."
            )

        if spacing_max <= spacing_min:
            raise ValueError(
                "spacing_range must satisfy max > min."
            )

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        self.carbon = carbon

        self.number_of_pulses = int(
            number_of_pulses
        )

        self.spacing_range = (
            float(spacing_min),
            float(spacing_max),
        )

        self.points = int(
            points
        )

    @property
    def H0(self) -> ComplexArray:
        """
        Nuclear Hamiltonian for electron |0>.

            H0 = f_L I_z
        """

        return (
            self.carbon.larmor_frequency
            * I_Z
        )

    @property
    def H1(self) -> ComplexArray:
        """
        Nuclear Hamiltonian for electron |1>.

            H1 =
                (f_L + A_parallel) I_z
                + A_perpendicular I_x
        """

        return (
            (
                self.carbon.larmor_frequency
                + self.carbon.A_parallel
            )
            * I_Z
            +
            self.carbon.A_perpendicular
            * I_X
        )

    def _conditional_propagators(
        self,
        pulse_spacing: float,
    ) -> tuple[
        ComplexArray,
        ComplexArray,
    ]:
        """
        Construct conditional nuclear propagators.

        Branch 0 begins with the electron in |0>.
        Branch 1 begins with the electron in |1>.

        Every ideal electron pi pulse swaps which nuclear
        Hamiltonian acts during the next interval.
        """

        half_spacing = (
            pulse_spacing / 2.0
        )

        # Electron-state branch labels.
        branch_0 = 0
        branch_1 = 1

        U0 = IDENTITY.copy()
        U1 = IDENTITY.copy()

        # ----------------------------------------------------
        # Initial half interval
        # ----------------------------------------------------

        U0 = (
            unitary(
                self.H0,
                half_spacing,
            )
            @ U0
        )

        U1 = (
            unitary(
                self.H1,
                half_spacing,
            )
            @ U1
        )

        # ----------------------------------------------------
        # Pulses and subsequent intervals
        # ----------------------------------------------------

        for pulse_index in range(
            self.number_of_pulses
        ):

            # Electron pi pulse swaps |0> <-> |1>.
            branch_0 = (
                1 - branch_0
            )

            branch_1 = (
                1 - branch_1
            )

            # Final interval is half spacing.
            if (
                pulse_index
                == self.number_of_pulses - 1
            ):
                duration = (
                    half_spacing
                )
            else:
                duration = (
                    pulse_spacing
                )

            H_branch_0 = (
                self.H0
                if branch_0 == 0
                else self.H1
            )

            H_branch_1 = (
                self.H0
                if branch_1 == 0
                else self.H1
            )

            U0 = (
                unitary(
                    H_branch_0,
                    duration,
                )
                @ U0
            )

            U1 = (
                unitary(
                    H_branch_1,
                    duration,
                )
                @ U1
            )

        return (
            U0,
            U1,
        )

    def coherence_at_spacing(
        self,
        pulse_spacing: float,
    ) -> float:
        """
        Return electron coherence for one pulse spacing.

        For an initially unpolarized spin-1/2 nucleus,

            L = (1/2) Tr(U0 U1^dagger).

        We return the real component relevant to the chosen
        electron readout basis.
        """

        U0, U1 = (
            self._conditional_propagators(
                pulse_spacing
            )
        )

        overlap = (
            0.5
            * np.trace(
                U0
                @ U1.conj().T
            )
        )

        return float(
            np.real(
                overlap
            )
        )

    def simulate(
        self,
    ) -> DDSpectroscopyResult:
        """Scan pulse spacing."""

        pulse_spacings = np.linspace(
            self.spacing_range[0],
            self.spacing_range[1],
            self.points,
        )

        coherence = np.zeros(
            self.points,
            dtype=float,
        )

        for index, spacing in enumerate(
            pulse_spacings
        ):

            coherence[index] = (
                self.coherence_at_spacing(
                    float(spacing)
                )
            )

        # If the electron is prepared and measured along
        # the same transverse axis:
        #
        #     P_x = (1 + L) / 2

        preservation_probability = (
            0.5
            * (
                1.0
                + coherence
            )
        )

        return DDSpectroscopyResult(
            pulse_spacings=(
                pulse_spacings
            ),
            coherence=(
                coherence
            ),
            preservation_probability=(
                preservation_probability
            ),
        )