"""Conditional nuclear environments coupled to an NV electron."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]


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


def unitary(
    hamiltonian: ComplexArray,
    duration: float,
) -> ComplexArray:
    """
    Exact unitary propagation.

    Hamiltonian frequencies are in GHz and duration is in ns.

    Evolution therefore uses

        U = exp(-i 2*pi H t).
    """

    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian
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


class ConditionalNuclearSubsystem(Protocol):
    """
    Interface for an environment subsystem whose Hamiltonian
    depends on the NV electron state.
    """

    @property
    def H0(self) -> ComplexArray:
        ...

    @property
    def H1(self) -> ComplexArray:
        ...

    @property
    def initial_density_matrix(
        self,
    ) -> ComplexArray:
        ...


@dataclass(frozen=True)
class Carbon13Spin:
    """
    One weakly coupled spin-1/2 carbon-13 nucleus.

    Parameters
    ----------
    larmor_frequency
        Nuclear precession frequency for electron |0>,
        in GHz.

    A_parallel
        Longitudinal hyperfine coupling in GHz.

    A_perpendicular
        Transverse hyperfine coupling in GHz.

    label
        Optional human-readable identifier.
    """

    larmor_frequency: float
    A_parallel: float
    A_perpendicular: float

    label: str = "13C"

    @property
    def H0(self) -> ComplexArray:
        """
        Nuclear Hamiltonian conditional on NV |0>.

            H0 = f_L I_z
        """

        return (
            self.larmor_frequency
            * I_Z
        )

    @property
    def H1(self) -> ComplexArray:
        """
        Nuclear Hamiltonian conditional on NV |1>.

            H1 =
                (f_L + A_parallel) I_z
                + A_perpendicular I_x
        """

        return (
            (
                self.larmor_frequency
                + self.A_parallel
            )
            * I_Z
            +
            self.A_perpendicular
            * I_X
        )

    @property
    def initial_density_matrix(
        self,
    ) -> ComplexArray:
        """
        Initially unpolarized nuclear spin.

            rho = I / 2
        """

        return (
            0.5
            * np.eye(
                2,
                dtype=complex,
            )
        )
    
@dataclass(frozen=True)
class Carbon13Pair:
    """
    Effective pseudospin model for an interacting 13C-13C pair.

    In the high-field secular approximation, the two
    anti-parallel nuclear states

        |up down>
        |down up>

    form an effective spin-1/2 pseudospin.

    Conditional Hamiltonians
    ------------------------

        H0 = X I_x

        H1 = X I_x + Z I_z

    Parameters
    ----------
    X
        Nuclear-nuclear dipolar flip-flop coupling in GHz.

    Z
        Hyperfine-field gradient experienced when the
        NV electron is in |1>, in GHz.

    label
        Optional human-readable identifier.

    Notes
    -----
    This is an effective two-dimensional pseudospin model.

    It does not explicitly propagate the full four-dimensional
    Hilbert space

        |up up>,
        |up down>,
        |down up>,
        |down down>.

    The approximation assumes the high-field regime in which
    the parallel states are energetically separated from the
    flip-flop subspace.
    """

    X: float
    Z: float

    label: str = "13C-13C pair"

    @property
    def H0(self) -> ComplexArray:
        """
        Pair pseudospin Hamiltonian for NV |0>.

            H0 = X I_x
        """

        return (
            self.X
            * I_X
        )

    @property
    def H1(self) -> ComplexArray:
        """
        Pair pseudospin Hamiltonian for NV |1>.

            H1 = X I_x + Z I_z
        """

        return (
            self.X
            * I_X
            +
            self.Z
            * I_Z
        )

    @property
    def initial_density_matrix(
        self,
    ) -> ComplexArray:
        """
        Initially unpolarized pair pseudospin.

            rho = I / 2
        """

        return (
            0.5
            * np.eye(
                2,
                dtype=complex,
            )
        )

    @property
    def frequency_0(self) -> float:
        """
        Pseudospin precession frequency for NV |0>.

            f0 = |X|
        """

        return abs(
            self.X
        )

    @property
    def frequency_1(self) -> float:
        """
        Pseudospin precession frequency for NV |1>.

            f1 = sqrt(X^2 + Z^2)
        """

        return float(
            np.sqrt(
                self.X**2
                + self.Z**2
            )
        )

def conditional_propagators(
    subsystem: ConditionalNuclearSubsystem,
    number_of_pulses: int,
    pulse_spacing: float,
) -> tuple[
    ComplexArray,
    ComplexArray,
]:
    """
    Conditional environment propagators under ideal DD.

    pulse_spacing is the center-to-center separation between
    consecutive ideal electron pi pulses.

    Timing:

        spacing/2
        pi
        spacing
        pi
        ...
        pi
        spacing/2

    Electron pi pulses swap which conditional Hamiltonian
    acts on each electron branch.
    """

    if number_of_pulses < 1:
        raise ValueError(
            "number_of_pulses must be at least 1."
        )

    if pulse_spacing <= 0:
        raise ValueError(
            "pulse_spacing must be positive."
        )

    half_spacing = (
        pulse_spacing / 2.0
    )

    dimension = (
        subsystem.H0.shape[0]
    )

    U0 = np.eye(
        dimension,
        dtype=complex,
    )

    U1 = np.eye(
        dimension,
        dtype=complex,
    )

    branch_0 = 0
    branch_1 = 1

    # Initial half interval.
    U0 = (
        unitary(
            subsystem.H0,
            half_spacing,
        )
        @ U0
    )

    U1 = (
        unitary(
            subsystem.H1,
            half_spacing,
        )
        @ U1
    )

    for pulse_index in range(
        number_of_pulses
    ):

        # Ideal electron pi pulse:
        #
        # |0> <-> |1>

        branch_0 = (
            1 - branch_0
        )

        branch_1 = (
            1 - branch_1
        )

        if (
            pulse_index
            == number_of_pulses - 1
        ):
            duration = (
                half_spacing
            )
        else:
            duration = (
                pulse_spacing
            )

        H_branch_0 = (
            subsystem.H0
            if branch_0 == 0
            else subsystem.H1
        )

        H_branch_1 = (
            subsystem.H0
            if branch_1 == 0
            else subsystem.H1
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


def subsystem_coherence(
    subsystem: ConditionalNuclearSubsystem,
    number_of_pulses: int,
    pulse_spacing: float,
) -> complex:
    """
    Return the complex NV coherence contributed by one
    conditional nuclear subsystem.

    For environment density matrix rho_E,

        L =
            Tr(
                U0 rho_E U1^dagger
            ).
    """

    U0, U1 = (
        conditional_propagators(
            subsystem=subsystem,
            number_of_pulses=(
                number_of_pulses
            ),
            pulse_spacing=(
                pulse_spacing
            ),
        )
    )

    rho = (
        subsystem.initial_density_matrix
    )

    return complex(
        np.trace(
            U0
            @ rho
            @ U1.conj().T
        )
    )


@dataclass(frozen=True)
class IndependentNuclearEnvironment:
    """
    Collection of independent nuclear subsystems.

    If the initial environment state factorizes and the
    subsystems do not interact with each other, their
    contributions to NV coherence factorize:

        L_total = product_j L_j.
    """

    subsystems: tuple[
        ConditionalNuclearSubsystem,
        ...,
    ]

    def coherence(
        self,
        number_of_pulses: int,
        pulse_spacing: float,
    ) -> complex:
        """Return total complex NV coherence."""

        total = 1.0 + 0.0j

        for subsystem in (
            self.subsystems
        ):

            total *= (
                subsystem_coherence(
                    subsystem=subsystem,
                    number_of_pulses=(
                        number_of_pulses
                    ),
                    pulse_spacing=(
                        pulse_spacing
                    ),
                )
            )

        return total

    def individual_coherences(
        self,
        number_of_pulses: int,
        pulse_spacing: float,
    ) -> tuple[complex, ...]:
        """Return each subsystem contribution separately."""

        return tuple(
            subsystem_coherence(
                subsystem=subsystem,
                number_of_pulses=(
                    number_of_pulses
                ),
                pulse_spacing=(
                    pulse_spacing
                ),
            )
            for subsystem in self.subsystems
        )