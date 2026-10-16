"""Continuous-wave ODMR simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvspin.physics.hamiltonian import (
    embedded_operator,
    eigensystem,
)
from nvspin.physics.nv_center import NVCenter
from nvspin.physics.operators import spin_operators

from nvspin.readout import NVOpticalCycle


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ============================================================
# Result data structures
# ============================================================


@dataclass
class ODMRTransition:
    """
    One microwave-driven transition.

    Parameters
    ----------
    initial_state
        Eigenstate index of the initial state.

    final_state
        Eigenstate index of the final state.

    frequency
        Transition frequency in GHz.

    strength
        Microwave transition strength

            |<f| n_MW . S |i>|^2.

    population_weight
        Initial-state spectral weight.

        For initial_manifold="ms0", this is determined by
        the ms=0 character of the initial eigenstate.
    """

    initial_state: int
    final_state: int
    frequency: float
    strength: float
    population_weight: float


@dataclass
class ODMRResult:
    """Result of a CW-ODMR simulation."""

    frequencies: FloatArray
    fluorescence: FloatArray
    transitions: list[ODMRTransition]

    def strongest_transitions(
        self,
        number: int = 10,
    ) -> list[ODMRTransition]:
        """
        Return transitions ordered by spectral weight.
        """

        return sorted(
            self.transitions,
            key=lambda transition:
            transition.strength
            * transition.population_weight,
            reverse=True,
        )[:number]


# ============================================================
# ODMR experiment
# ============================================================


class ODMR:
    """
    Continuous-wave ODMR simulator using a five-level
    optical rate-equation model and a Bloch-derived
    microwave transition rate.

    The spin Hamiltonian is diagonalized exactly.

    For each microwave-accessible transition,

        f_ij = E_j - E_i,

    and

        M_ij
        =
        |<j| n_MW . S |i>|^2

    are calculated from the complete spin Hamiltonian.

    A reference Rabi frequency is supplied by the user.
    The transition-specific Rabi frequency is determined
    from the microwave matrix element.

    The frequency-dependent microwave-induced population
    mixing rate is

        W_ij(f)
        =
        (2*pi*Omega_ij)^2 T2
        ----------------------------------
        2 [1 + (2*pi*Delta_ij*T2)^2]

    where

        Delta_ij = f - f_ij.

    Frequencies are in GHz and times are in ns, so

        GHz * ns = dimensionless.

    The total effective microwave population-transfer rate is

        W(f)
        =
        sum_ij
        p_i^spec W_ij(f).

    This rate is then coupled to the five-level optical
    population model.

    At every microwave frequency the code solves

        K(f) p_ss(f) = 0

    subject to

        sum(p_ss) = 1.

    The optical populations are ordered as

        [
            g_ms0,
            g_ms1,
            e_ms0,
            e_ms1,
            singlet,
        ].

    Fluorescence is

        F(f)
        =
        k_r0 p_e0(f)
        +
        k_r1 p_e1(f).

    No phenomenological ODMR contrast is supplied.

    No independent phenomenological ODMR linewidth is
    supplied.

    No independent microwave population-mixing rate is
    supplied.

    Instead, the weak-drive homogeneous microwave response
    is determined by

        rabi_frequency
        T2
        transition frequency
        microwave matrix element.

    Parameters
    ----------
    system
        NVCenter spin system.

    optical_cycle
        NVOpticalCycle containing the optical transition
        rates.

        The optical rates are expected to contain

            excitation
            radiative_ms0
            radiative_ms1
            isc_ms0
            isc_ms1
            singlet_to_ms0
            singlet_to_ms1

        and are assumed to have units of inverse ns.

    rabi_frequency
        Reference resonant Rabi frequency in GHz.

        This is defined for a reference allowed electronic
        transition whose microwave matrix-element strength
        is 1/2.

        For a transition with strength M_ij,

            Omega_ij
            =
            rabi_frequency
            * sqrt(M_ij / 0.5).

        Thus a pure spin-1 transition driven transverse to
        the NV axis has Omega_ij approximately equal to the
        requested rabi_frequency.

    T2
        Homogeneous transverse coherence time in ns.

        In the weak-drive limit, the corresponding
        homogeneous Lorentzian FWHM is

            Gamma
            =
            1 / (pi T2)

        in ordinary-frequency units.

    frequency_range
        Microwave-frequency range in GHz.

    points
        Number of microwave-frequency points.

    microwave_direction
        Microwave magnetic-field direction in the NV frame.

    strength_cutoff
        Ignore transitions whose microwave matrix-element
        strength is below this value.

    initial_manifold
        "ms0" or "all".

        "ms0" weights transitions according to the ms=0
        character of the initial eigenstate.

        "all" assigns equal spectral weight to all
        eigenstates.

    Notes
    -----
    This model is more physical than prescribing an
    independent linewidth and microwave mixing rate.

    However, it is still an effective population model.

    The microwave coherence has been adiabatically eliminated
    to obtain W_ij(f).

    A future full optical-Bloch or Lindblad model would retain
    the microwave coherence explicitly.
    """

    # For a pure spin-1 system with transverse drive,
    #
    # |<0|Sx|+1>|^2
    # =
    # |<0|Sx|-1>|^2
    # =
    # 1/2.
    #
    # We therefore define the user-supplied rabi_frequency
    # relative to this transition strength.
    REFERENCE_STRENGTH = 0.5

    def __init__(
        self,
        system: NVCenter,
        optical_cycle: NVOpticalCycle,
        rabi_frequency: float,
        T2: float,
        frequency_range: tuple[float, float] = (
            2.5,
            3.2,
        ),
        points: int = 4000,
        microwave_direction: tuple[
            float,
            float,
            float,
        ] = (
            1.0,
            0.0,
            0.0,
        ),
        strength_cutoff: float = 1e-8,
        initial_manifold: str = "ms0",
    ) -> None:

        f_min, f_max = frequency_range

        if f_max <= f_min:
            raise ValueError(
                "frequency_range must satisfy "
                "f_max > f_min."
            )

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        if rabi_frequency < 0.0:
            raise ValueError(
                "rabi_frequency must be non-negative."
            )

        if T2 <= 0.0:
            raise ValueError(
                "T2 must be positive."
            )

        direction = np.asarray(
            microwave_direction,
            dtype=float,
        )

        if direction.shape != (3,):
            raise ValueError(
                "microwave_direction must contain "
                "three components."
            )

        norm = np.linalg.norm(
            direction
        )

        if np.isclose(
            norm,
            0.0,
        ):
            raise ValueError(
                "microwave_direction must be non-zero."
            )

        if initial_manifold not in {
            "ms0",
            "all",
        }:
            raise ValueError(
                "initial_manifold must be "
                "'ms0' or 'all'."
            )

        if optical_cycle is None:
            raise ValueError(
                "optical_cycle must be provided."
            )

        self.system = system

        self.optical_cycle = (
            optical_cycle
        )

        self.rabi_frequency = float(
            rabi_frequency
        )

        self.T2 = float(
            T2
        )

        self.frequency_range = (
            float(f_min),
            float(f_max),
        )

        self.points = int(
            points
        )

        self.microwave_direction = (
            direction / norm
        )

        self.strength_cutoff = float(
            strength_cutoff
        )

        self.initial_manifold = (
            initial_manifold
        )

    # =========================================================
    # Hilbert-space information
    # =========================================================

    def _dimensions(
        self,
    ) -> list[int]:

        return [
            3,
            *[
                nucleus.dimension
                for nucleus
                in self.system.nuclei
            ],
        ]

    # =========================================================
    # Electronic operators in full Hilbert space
    # =========================================================

    def _electron_operators(
        self,
    ) -> tuple[
        ComplexArray,
        ComplexArray,
        ComplexArray,
    ]:

        sx0, sy0, sz0 = (
            spin_operators(1.0)
        )

        dimensions = (
            self._dimensions()
        )

        sx = embedded_operator(
            sx0,
            0,
            dimensions,
        )

        sy = embedded_operator(
            sy0,
            0,
            dimensions,
        )

        sz = embedded_operator(
            sz0,
            0,
            dimensions,
        )

        return (
            sx,
            sy,
            sz,
        )

    # =========================================================
    # Microwave drive operator
    # =========================================================

    def microwave_operator(
        self,
    ) -> ComplexArray:
        """
        Return the dimensionless microwave coupling operator

            n_MW . S

        in the complete electron+nuclear Hilbert space.
        """

        sx, sy, sz = (
            self._electron_operators()
        )

        nx, ny, nz = (
            self.microwave_direction
        )

        return (
            nx * sx
            + ny * sy
            + nz * sz
        )

    # =========================================================
    # ms = 0 projector
    # =========================================================

    def ms0_projector(
        self,
    ) -> ComplexArray:
        """
        Return the electronic ms=0 projector embedded in the
        complete electron+nuclear Hilbert space.
        """

        p0_electron = np.diag(
            [
                0.0,
                1.0,
                0.0,
            ]
        ).astype(
            complex
        )

        return embedded_operator(
            p0_electron,
            0,
            self._dimensions(),
        )

    # =========================================================
    # Initial spectral populations
    # =========================================================

    def initial_populations(
        self,
        states: ComplexArray,
    ) -> FloatArray:
        """
        Construct spectral transition weights.

        These weights are used to determine how strongly each
        Hamiltonian transition contributes to the effective
        microwave population-transfer rate.

        The actual optical steady-state populations are solved
        separately from the five-level rate equations.
        """

        dimension = (
            states.shape[1]
        )

        if (
            self.initial_manifold
            == "all"
        ):

            return np.full(
                dimension,
                1.0 / dimension,
                dtype=float,
            )

        p0 = (
            self.ms0_projector()
        )

        ms0_character = np.real(
            np.diag(
                states.conj().T
                @ p0
                @ states
            )
        )

        ms0_character = np.clip(
            ms0_character,
            0.0,
            None,
        )

        total = np.sum(
            ms0_character
        )

        if np.isclose(
            total,
            0.0,
        ):
            raise RuntimeError(
                "Could not identify the "
                "ms=0 manifold."
            )

        return (
            ms0_character
            / total
        )

    # =========================================================
    # Transition calculation
    # =========================================================

    def calculate_transitions(
        self,
    ) -> list[ODMRTransition]:
        """
        Calculate microwave-accessible transitions from the
        complete spin Hamiltonian.
        """

        energies, states = (
            eigensystem(
                self.system
            )
        )

        drive = (
            self.microwave_operator()
        )

        populations = (
            self.initial_populations(
                states
            )
        )

        transitions: list[
            ODMRTransition
        ] = []

        dimension = len(
            energies
        )

        for initial in range(
            dimension
        ):

            for final in range(
                initial + 1,
                dimension,
            ):

                frequency = float(
                    energies[final]
                    - energies[initial]
                )

                matrix_element = (
                    states[:, final].conj()
                    @ drive
                    @ states[:, initial]
                )

                strength = float(
                    np.abs(
                        matrix_element
                    ) ** 2
                )

                if (
                    strength
                    < self.strength_cutoff
                ):
                    continue

                population_weight = float(
                    populations[initial]
                )

                if (
                    population_weight
                    <= 0.0
                ):
                    continue

                transitions.append(
                    ODMRTransition(
                        initial_state=initial,
                        final_state=final,
                        frequency=frequency,
                        strength=strength,
                        population_weight=(
                            population_weight
                        ),
                    )
                )

        return transitions

    # =========================================================
    # Transition-specific Rabi frequency
    # =========================================================

    def _transition_rabi_frequency(
        self,
        transition: ODMRTransition,
    ) -> float:
        """
        Return the Rabi frequency for one transition in GHz.

        The user-supplied rabi_frequency corresponds to an
        allowed pure electronic spin-1 transverse transition
        with matrix-element strength 1/2.

        Therefore

            Omega_ij
            =
            Omega_ref
            sqrt(M_ij / (1/2)).
        """

        if transition.strength <= 0.0:
            return 0.0

        return float(
            self.rabi_frequency
            * np.sqrt(
                transition.strength
                / self.REFERENCE_STRENGTH
            )
        )

    # =========================================================
    # Bloch-derived microwave transition rate
    # =========================================================

    def _transition_mixing_rate(
        self,
        frequency: float,
        transition: ODMRTransition,
    ) -> float:
        """
        Return the effective microwave population-transfer rate
        for one transition.

        Frequencies are stored in GHz = 1/ns.

        The angular Rabi frequency is

            Omega_w = 2*pi*Omega,

        and angular detuning is

            Delta_w = 2*pi*(f - f_ij).

        After adiabatic elimination of the transverse
        microwave coherence,

            W_ij
            =
            Omega_w^2 T2
            -----------------------------
            2 [1 + Delta_w^2 T2^2].

        The result has units of inverse ns.
        """

        omega = (
            self._transition_rabi_frequency(
                transition
            )
        )

        if omega == 0.0:
            return 0.0

        detuning = (
            float(frequency)
            - transition.frequency
        )

        omega_angular = (
            2.0
            * np.pi
            * omega
        )

        detuning_angular = (
            2.0
            * np.pi
            * detuning
        )

        denominator = (
            2.0
            * (
                1.0
                +
                (
                    detuning_angular
                    * self.T2
                ) ** 2
            )
        )

        return float(
            omega_angular**2
            * self.T2
            / denominator
        )

    # =========================================================
    # Total microwave mixing rate
    # =========================================================

    def _microwave_mixing_rate(
        self,
        frequency: float,
        transitions: list[
            ODMRTransition
        ],
    ) -> float:
        """
        Return the total effective microwave-induced
        ground-state population-transfer rate W(f).

        Importantly, the transition spectrum is NOT normalized
        to its maximum.

        Absolute Hamiltonian transition strengths therefore
        remain in the model.
        """

        total_rate = 0.0

        for transition in transitions:

            transition_rate = (
                self._transition_mixing_rate(
                    frequency,
                    transition,
                )
            )

            total_rate += (
                transition.population_weight
                * transition_rate
            )

        return float(
            total_rate
        )

    # =========================================================
    # Five-level optical rate matrix
    # =========================================================

    def _rate_matrix(
        self,
        microwave_mixing: float,
    ) -> FloatArray:
        """
        Construct the five-level optical rate matrix.

        Population ordering:

            0 : g_ms0
            1 : g_ms1
            2 : e_ms0
            3 : e_ms1
            4 : singlet

        Evolution convention:

            dp/dt = K p.

        Therefore every column of K must sum to zero.
        """

        rates = (
            self.optical_cycle.rates
        )

        k_ex = float(
            rates.excitation
        )

        k_r0 = float(
            rates.radiative_ms0
        )

        k_r1 = float(
            rates.radiative_ms1
        )

        k_isc0 = float(
            rates.isc_ms0
        )

        k_isc1 = float(
            rates.isc_ms1
        )

        k_s0 = float(
            rates.singlet_to_ms0
        )

        k_s1 = float(
            rates.singlet_to_ms1
        )

        w = float(
            microwave_mixing
        )

        return np.array(
            [
                [
                    -(k_ex + w),
                    w,
                    k_r0,
                    0.0,
                    k_s0,
                ],
                [
                    w,
                    -(k_ex + w),
                    0.0,
                    k_r1,
                    k_s1,
                ],
                [
                    k_ex,
                    0.0,
                    -(k_r0 + k_isc0),
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    k_ex,
                    0.0,
                    -(k_r1 + k_isc1),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    k_isc0,
                    k_isc1,
                    -(k_s0 + k_s1),
                ],
            ],
            dtype=float,
        )

    # =========================================================
    # Steady-state solution
    # =========================================================

    @staticmethod
    def _steady_state(
        rate_matrix: FloatArray,
    ) -> FloatArray:
        """
        Solve

            K p_ss = 0

        subject to

            sum(p_ss) = 1.
        """

        matrix = np.array(
            rate_matrix,
            dtype=float,
            copy=True,
        )

        rhs = np.zeros(
            5,
            dtype=float,
        )

        # The steady-state system is singular because
        # probability is conserved.
        #
        # Replace one redundant equation with the
        # normalization condition.
        matrix[-1, :] = 1.0
        rhs[-1] = 1.0

        populations = (
            np.linalg.solve(
                matrix,
                rhs,
            )
        )

        populations[
            np.abs(
                populations
            ) < 1e-14
        ] = 0.0

        if np.any(
            populations
            < -1e-10
        ):
            raise RuntimeError(
                "Optical steady state "
                "contains negative populations."
            )

        populations = np.clip(
            populations,
            0.0,
            None,
        )

        total = float(
            np.sum(
                populations
            )
        )

        if total <= 0.0:
            raise RuntimeError(
                "Optical steady state has "
                "zero total population."
            )

        populations = (
            populations
            / total
        )

        return populations

    # =========================================================
    # Fluorescence
    # =========================================================

    def _emission_rate(
        self,
        populations: FloatArray,
    ) -> float:
        """
        Calculate the emitted radiative photon rate.

        For the five-level model,

            F
            =
            k_r0 p_e0
            +
            k_r1 p_e1.

        Detector collection efficiency and background counts
        are not included here.
        """

        rates = (
            self.optical_cycle.rates
        )

        p_e0 = float(
            populations[2]
        )

        p_e1 = float(
            populations[3]
        )

        return float(
            rates.radiative_ms0
            * p_e0
            +
            rates.radiative_ms1
            * p_e1
        )

    # =========================================================
    # Useful derived quantities
    # =========================================================

    @property
    def weak_drive_fwhm(
        self,
    ) -> float:
        """
        Homogeneous weak-drive FWHM in GHz.

        For the Bloch-derived response,

            Gamma
            =
            1 / (pi T2).
        """

        return float(
            1.0
            /
            (
                np.pi
                * self.T2
            )
        )

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(
        self,
    ) -> ODMRResult:
        """
        Simulate the CW-ODMR spectrum.

        For every microwave frequency:

        1. calculate exact Hamiltonian transition frequencies
           and microwave matrix elements;

        2. calculate each transition-specific Rabi frequency;

        3. derive the effective microwave transition rate from
           rabi_frequency, T2, and detuning;

        4. sum the transition contributions to obtain W(f);

        5. construct the five-level optical rate matrix;

        6. solve

               K(f) p_ss(f) = 0;

        7. calculate

               F(f)
               =
               k_r0 p_e0(f)
               +
               k_r1 p_e1(f).

        No independent phenomenological ODMR linewidth is
        imposed.

        No fixed ODMR contrast is imposed.
        """

        frequencies = np.linspace(
            self.frequency_range[0],
            self.frequency_range[1],
            self.points,
        )

        transitions = (
            self.calculate_transitions()
        )

        fluorescence = np.zeros_like(
            frequencies
        )

        for (
            index,
            frequency,
        ) in enumerate(
            frequencies
        ):

            microwave_mixing = (
                self._microwave_mixing_rate(
                    float(frequency),
                    transitions,
                )
            )

            rate_matrix = (
                self._rate_matrix(
                    microwave_mixing
                )
            )

            populations = (
                self._steady_state(
                    rate_matrix
                )
            )

            fluorescence[index] = (
                self._emission_rate(
                    populations
                )
            )

        # Normalize only for presentation.
        #
        # The ODMR contrast and spectral structure have
        # already emerged from the Hamiltonian, microwave
        # response, and optical rate equations.
        maximum = float(
            np.max(
                fluorescence
            )
        )

        if maximum > 0.0:

            fluorescence = (
                fluorescence
                / maximum
            )

        return ODMRResult(
            frequencies=frequencies,
            fluorescence=fluorescence,
            transitions=transitions,
        )