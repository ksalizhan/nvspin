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
    Continuous-wave ODMR simulator using coupled coherent
    microwave dynamics and a five-level optical cycle.

    The model contains five optical populations,

        p_g0
        p_g1
        p_e0
        p_e1
        p_s

    together with the two real components of the ground-state
    microwave coherence,

        u = 2 Re(rho_01)
        v = 2 Im(rho_01).

    The complete steady-state variable is

        x =
        [
            p_g0,
            p_g1,
            p_e0,
            p_e1,
            p_s,
            u,
            v,
        ].

    For each microwave-accessible Hamiltonian transition, the
    transition frequency and matrix element are calculated
    exactly from the spin Hamiltonian.

    The transition-specific Rabi frequency is

        Omega_ij
        =
        Omega_ref
        sqrt(M_ij / M_ref).

    No phenomenological ODMR contrast is supplied.

    No phenomenological linewidth is supplied.

    No effective microwave population-transfer rate W is
    supplied.

    Instead, microwave saturation and the frequency response
    emerge from the coherent Bloch dynamics.

    Parameters
    ----------
    system
        NVCenter spin system.

    optical_cycle
        Five-level NV optical-cycle model.

    rabi_frequency
        Reference resonant Rabi frequency in GHz.

        A pure spin-1 transverse transition with

            |<f|Sx|i>|^2 = 1/2

        has this Rabi frequency.

    T2
        Intrinsic ground-state transverse coherence time in ns.

        This represents the non-optical contribution to
        transverse decoherence.

        Optical excitation contributes additional ground-state
        coherence loss during CW illumination.

    frequency_range
        Microwave frequency range in GHz.

    points
        Number of frequency points.

    microwave_direction
        Microwave magnetic-field direction in the NV frame.

    strength_cutoff
        Ignore transitions weaker than this value.

    initial_manifold
        "ms0" or "all".
    """

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
    # Electronic operators
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
    # Spectral weights
    # =========================================================

    def initial_populations(
        self,
        states: ComplexArray,
    ) -> FloatArray:

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
    # Hamiltonian transitions
    # =========================================================

    def calculate_transitions(
        self,
    ) -> list[ODMRTransition]:

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
    # Effective transverse coherence time under laser
    # =========================================================

    @property
    def optical_transverse_decay_rate(
        self,
    ) -> float:
        """
        Optical contribution to ground-state coherence decay.

        Both g0 and g1 are excited at rate k_ex.

        Loss from either endpoint contributes half its
        population-loss rate to rho_01:

            gamma_opt
            =
            (k_ex + k_ex) / 2
            =
            k_ex.
        """

        rates = (
            self.optical_cycle.rates
        )

        return float(
            rates.excitation
        )

    @property
    def transverse_decay_rate(
        self,
    ) -> float:
        """
        Total transverse decay rate during CW illumination:

            gamma_2
            =
            1/T2
            +
            gamma_opt.
        """

        return float(
            1.0 / self.T2
            + self.optical_transverse_decay_rate
        )

    @property
    def effective_T2(
        self,
    ) -> float:
        """
        Effective transverse coherence time during CW optical
        illumination.
        """

        return float(
            1.0
            / self.transverse_decay_rate
        )

    @property
    def weak_drive_fwhm(
        self,
    ) -> float:
        """
        Weak-drive homogeneous FWHM in GHz during optical
        illumination:

            Gamma
            =
            gamma_2 / pi.
        """

        return float(
            self.transverse_decay_rate
            / np.pi
        )

    # =========================================================
    # Seven-variable steady-state matrix for one transition
    # =========================================================

    def _bloch_rate_matrix(
        self,
        frequency: float,
        transition: ODMRTransition,
    ) -> FloatArray:
        """
        Construct the seven-variable optical-Bloch matrix.

        Variable ordering:

            0 : p_g0
            1 : p_g1
            2 : p_e0
            3 : p_e1
            4 : p_s
            5 : u = 2 Re(rho_01)
            6 : v = 2 Im(rho_01)

        Evolution:

            dx/dt = A x.

        The coherent microwave Hamiltonian is

            H/h
            =
            Delta/2 sigma_z
            +
            Omega/2 sigma_x

        in ordinary-frequency units.

        Therefore

            Omega_w = 2*pi*Omega

        and

            Delta_w = 2*pi*Delta.
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

        k_i0 = float(
            rates.isc_ms0
        )

        k_i1 = float(
            rates.isc_ms1
        )

        k_s0 = float(
            rates.singlet_to_ms0
        )

        k_s1 = float(
            rates.singlet_to_ms1
        )

        omega = (
            self._transition_rabi_frequency(
                transition
            )
        )

        detuning = (
            float(frequency)
            - transition.frequency
        )

        omega_w = (
            2.0
            * np.pi
            * omega
        )

        delta_w = (
            2.0
            * np.pi
            * detuning
        )

        gamma_2 = (
            self.transverse_decay_rate
        )

        # =====================================================
        # Variable ordering:
        #
        # [pg0, pg1, pe0, pe1, ps, u, v]
        #
        # Every row has exactly seven entries.
        # =====================================================

        matrix = np.array(
            [
                # -------------------------------------------------
                # d p_g0 / dt
                # -------------------------------------------------
                [
                    -k_ex,
                    0.0,
                    k_r0,
                    0.0,
                    k_s0,
                    0.0,
                    -0.5 * omega_w,
                ],

                # -------------------------------------------------
                # d p_g1 / dt
                # -------------------------------------------------
                [
                    0.0,
                    -k_ex,
                    0.0,
                    k_r1,
                    k_s1,
                    0.0,
                    +0.5 * omega_w,
                ],

                # -------------------------------------------------
                # d p_e0 / dt
                # -------------------------------------------------
                [
                    k_ex,
                    0.0,
                    -(k_r0 + k_i0),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],

                # -------------------------------------------------
                # d p_e1 / dt
                # -------------------------------------------------
                [
                    0.0,
                    k_ex,
                    0.0,
                    -(k_r1 + k_i1),
                    0.0,
                    0.0,
                    0.0,
                ],

                # -------------------------------------------------
                # d p_s / dt
                # -------------------------------------------------
                [
                    0.0,
                    0.0,
                    k_i0,
                    k_i1,
                    -(k_s0 + k_s1),
                    0.0,
                    0.0,
                ],

                # -------------------------------------------------
                # d u / dt
                #
                # u = 2 Re(rho_01)
                # -------------------------------------------------
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -gamma_2,
                    -delta_w,
                ],

                # -------------------------------------------------
                # d v / dt
                #
                # v = 2 Im(rho_01)
                # -------------------------------------------------
                [
                    +omega_w,
                    -omega_w,
                    0.0,
                    0.0,
                    0.0,
                    +delta_w,
                    -gamma_2,
                ],
            ],
            dtype=float,
        )

        return matrix

    # =========================================================
    # Solve one coherent transition
    # =========================================================

    @staticmethod
    def _bloch_steady_state(
        matrix: FloatArray,
    ) -> FloatArray:
        """
        Solve the seven-variable steady state.

        Population normalization:

            p_g0
            + p_g1
            + p_e0
            + p_e1
            + p_s
            =
            1.

        u and v are coherences and do not contribute to
        population normalization.
        """

        system = np.array(
            matrix,
            dtype=float,
            copy=True,
        )

        rhs = np.zeros(
            7,
            dtype=float,
        )

        # Replace one redundant population equation with
        # population normalization.
        system[4, :] = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        rhs[4] = 1.0

        state = np.linalg.solve(
            system,
            rhs,
        )

        state[
            np.abs(
                state
            ) < 1e-14
        ] = 0.0

        populations = (
            state[:5]
        )

        if np.any(
            populations
            < -1e-10
        ):
            raise RuntimeError(
                "Optical-Bloch steady state contains "
                "negative populations."
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
                "Optical-Bloch steady state has "
                "zero population."
            )

        # Normally this is already one to numerical precision.
        state[:5] = (
            populations
            / total
        )

        return state

    # =========================================================
    # Fluorescence
    # =========================================================

    def _emission_rate_from_state(
        self,
        state: FloatArray,
    ) -> float:

        rates = (
            self.optical_cycle.rates
        )

        return float(
            rates.radiative_ms0
            * state[2]
            +
            rates.radiative_ms1
            * state[3]
        )

    # =========================================================
    # Single-transition fluorescence
    # =========================================================

    def _transition_fluorescence(
        self,
        frequency: float,
        transition: ODMRTransition,
    ) -> float:
        """
        Calculate fluorescence for one coherently driven
        transition.
        """

        matrix = (
            self._bloch_rate_matrix(
                frequency,
                transition,
            )
        )

        state = (
            self._bloch_steady_state(
                matrix
            )
        )

        return (
            self._emission_rate_from_state(
                state
            )
        )

    # =========================================================
    # Off-resonant optical fluorescence
    # =========================================================

    def _off_resonant_fluorescence(
        self,
    ) -> float:
        """
        Calculate the optical steady-state fluorescence with
        zero microwave drive.
        """

        dummy = ODMRTransition(
            initial_state=0,
            final_state=1,
            frequency=0.0,
            strength=0.0,
            population_weight=1.0,
        )

        matrix = (
            self._bloch_rate_matrix(
                frequency=0.0,
                transition=dummy,
            )
        )

        state = (
            self._bloch_steady_state(
                matrix
            )
        )

        return (
            self._emission_rate_from_state(
                state
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

        The seven-variable solver is exact for the reduced
        optical model with one coherently driven effective
        ground-state transition.

        If the complete Hamiltonian produces several resolved
        transitions, each transition is solved independently
        with its own coherent Bloch problem.

        Their fluorescence deficits are then combined.

        This is an approximation for a multi-transition
        electron+nuclear system.

        A future complete density-matrix / Liouvillian model
        can treat all resolved transitions and coherences
        simultaneously.
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

        off_resonant = (
            self._off_resonant_fluorescence()
        )

        if off_resonant <= 0.0:
            raise RuntimeError(
                "Off-resonant fluorescence must be positive."
            )

        for (
            index,
            frequency,
        ) in enumerate(
            frequencies
        ):

            total_deficit = 0.0

            for transition in transitions:

                transition_fluorescence = (
                    self._transition_fluorescence(
                        float(
                            frequency
                        ),
                        transition,
                    )
                )

                deficit = (
                    off_resonant
                    - transition_fluorescence
                )

                total_deficit += (
                    transition.population_weight
                    * deficit
                )

            fluorescence[index] = (
                off_resonant
                - total_deficit
            )

        # Numerical guard.
        fluorescence = np.clip(
            fluorescence,
            0.0,
            None,
        )

        # Normalize only for presentation.
        #
        # Contrast and linewidth have already emerged from
        # the optical-Bloch calculation.
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