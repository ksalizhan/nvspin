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
        Relative microwave transition strength.

    population_weight
        Initial-state population weight used in the
        microwave spectral response.
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
    optical rate-equation model.

    Transition frequencies and microwave transition strengths
    are calculated from the full NV spin Hamiltonian.

    The microwave spectrum determines a frequency-dependent
    ground-state mixing rate,

        W(f) = microwave_rate * A(f),

    where A(f) contains the Hamiltonian transition
    frequencies, microwave matrix elements, and Lorentzian
    broadening.

    At every microwave frequency, the optical steady state is
    obtained by solving

        K(f) p_ss(f) = 0

    subject to

        sum(p_ss) = 1.

    The five optical populations are ordered as

        [
            g_ms0,
            g_ms1,
            e_ms0,
            e_ms1,
            singlet,
        ].

    The emitted fluorescence rate is then

        F(f)
        =
        k_r0 * p_e0(f)
        +
        k_r1 * p_e1(f).

    The returned fluorescence is normalized so that the
    maximum fluorescence is one.

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

    microwave_rate
        Maximum effective microwave-induced ground-state
        population-mixing rate in inverse ns.

        This sets the overall microwave drive strength in the
        rate-equation model.

    frequency_range
        Microwave-frequency range in GHz.

    points
        Number of microwave-frequency points.

    linewidth
        Lorentzian full width at half maximum (FWHM), GHz.

    microwave_direction
        Microwave magnetic-field direction in the NV frame.

    strength_cutoff
        Ignore transitions with relative microwave strength
        below this value.

    initial_manifold
        Currently "ms0" or "all".

        "ms0" weights transitions according to the ms=0
        character of the initial eigenstate.

        "all" assigns equal initial spectral weight to all
        eigenstates.

    Notes
    -----
    The optical rates are constant during each steady-state
    calculation.

    What changes across the ODMR frequency sweep is the
    microwave-induced mixing rate W(f).

    Therefore the ODMR contrast is not supplied as an input.
    It emerges from the competition between microwave mixing,
    optical excitation, radiative decay, intersystem crossing,
    and singlet return.
    """

    def __init__(
        self,
        system: NVCenter,
        optical_cycle: NVOpticalCycle,
        microwave_rate: float,
        frequency_range: tuple[float, float] = (
            2.5,
            3.2,
        ),
        points: int = 4000,
        linewidth: float = 0.002,
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

        if linewidth <= 0:
            raise ValueError(
                "linewidth must be positive."
            )

        if microwave_rate < 0:
            raise ValueError(
                "microwave_rate must be non-negative."
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

        self.microwave_rate = float(
            microwave_rate
        )

        self.frequency_range = (
            float(f_min),
            float(f_max),
        )

        self.points = int(
            points
        )

        self.linewidth = float(
            linewidth
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
        Return the dimensionless microwave coupling operator.

        For microwave-field direction n,

            H1 proportional to n . S.

        This operator determines the relative microwave
        transition matrix elements.

        The overall CW microwave population-transfer scale is
        supplied separately through microwave_rate.
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
        Construct initial spectral weights.

        For initial_manifold="ms0", the transition spectral
        weight is distributed according to each eigenstate's
        ms=0 character.

        For initial_manifold="all", all eigenstates receive
        equal spectral weight.

        These weights are used only to construct the
        microwave spectral response A(f).

        The actual optical populations are subsequently
        calculated from the rate equations.
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
    # Lorentzian
    # =========================================================

    def _lorentzian(
        self,
        frequencies: FloatArray,
        center: float,
    ) -> FloatArray:
        """
        Return a unit-height Lorentzian centered at center.

        linewidth is the full width at half maximum.
        """

        half_width = (
            self.linewidth
            / 2.0
        )

        return (
            half_width**2
            /
            (
                (
                    frequencies
                    - center
                ) ** 2
                +
                half_width**2
            )
        )

    # =========================================================
    # Microwave spectral response
    # =========================================================

    def _microwave_response(
        self,
        frequencies: FloatArray,
        transitions: list[
            ODMRTransition
        ],
    ) -> FloatArray:
        """
        Construct normalized microwave response A(f).

        The response contains

            transition frequency,
            microwave matrix element,
            initial spectral weight,
            Lorentzian broadening.

        The actual microwave-induced population mixing is

            W(f)
            =
            microwave_rate * A(f).
        """

        response = np.zeros_like(
            frequencies
        )

        for transition in transitions:

            weight = (
                transition.strength
                * transition.population_weight
            )

            response += (
                weight
                * self._lorentzian(
                    frequencies,
                    transition.frequency,
                )
            )

        maximum = float(
            np.max(
                response
            )
        )

        if maximum > 0.0:

            response = (
                response
                / maximum
            )

        return response

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

        # Remove tiny floating-point roundoff.
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

        This is emitted fluorescence before detector
        collection efficiency or background counts.
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
    # Simulation
    # =========================================================

    def simulate(
        self,
    ) -> ODMRResult:
        """
        Simulate the CW-ODMR spectrum.

        For every microwave frequency:

        1. calculate the Hamiltonian-derived microwave
           response A(f);

        2. convert it into the population-mixing rate

               W(f)
               =
               microwave_rate * A(f);

        3. construct the five-level rate matrix K(f);

        4. solve

               K(f) p_ss(f) = 0;

        5. calculate

               F(f)
               =
               k_r0 p_e0(f)
               +
               k_r1 p_e1(f).

        The final fluorescence is normalized to its maximum
        value for convenient plotting.
        """

        frequencies = np.linspace(
            self.frequency_range[0],
            self.frequency_range[1],
            self.points,
        )

        transitions = (
            self.calculate_transitions()
        )

        microwave_response = (
            self._microwave_response(
                frequencies,
                transitions,
            )
        )

        fluorescence = np.zeros_like(
            frequencies
        )

        for (
            index,
            spectral_weight,
        ) in enumerate(
            microwave_response
        ):

            microwave_mixing = (
                self.microwave_rate
                * float(
                    spectral_weight
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
        # The ODMR dip itself has already emerged from
        # the rate equations. No contrast parameter is
        # imposed here.
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