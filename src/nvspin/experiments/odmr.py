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


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


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
        Initial-state population weight used in the spectrum.
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
        """Return transitions ordered by spectral weight."""

        return sorted(
            self.transitions,
            key=lambda transition:
            transition.strength
            * transition.population_weight,
            reverse=True,
        )[:number]


class ODMR:
    """
    Phenomenological CW-ODMR simulator.

    Transition frequencies and microwave strengths are
    calculated from the full NV spin Hamiltonian.

    ODMR fluorescence dips are represented by Lorentzian
    lines.

    Parameters
    ----------
    system
        NVCenter spin system.

    frequency_range
        Microwave-frequency range (GHz).

    points
        Number of frequency points.

    linewidth
        Lorentzian full width at half maximum (FWHM), GHz.

    contrast
        Maximum total phenomenological ODMR contrast.

    microwave_direction
        Microwave magnetic-field direction in the NV frame.

    strength_cutoff
        Ignore transitions with relative microwave strength
        below this value.

    initial_manifold
        Currently "ms0" or "all".
        "ms0" approximates optical initialization into the
        electronic ms ~ 0 manifold.
    """

    def __init__(
        self,
        system: NVCenter,
        frequency_range: tuple[float, float] = (
            2.5,
            3.2,
        ),
        points: int = 4000,
        linewidth: float = 0.002,
        contrast: float = 0.03,
        microwave_direction: tuple[
            float,
            float,
            float,
        ] = (1.0, 0.0, 0.0),
        strength_cutoff: float = 1e-8,
        initial_manifold: str = "ms0",
    ) -> None:

        f_min, f_max = frequency_range

        if f_max <= f_min:
            raise ValueError(
                "frequency_range must satisfy f_max > f_min."
            )

        if points < 2:
            raise ValueError(
                "points must be at least 2."
            )

        if linewidth <= 0:
            raise ValueError(
                "linewidth must be positive."
            )

        if contrast < 0:
            raise ValueError(
                "contrast must be non-negative."
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
                "initial_manifold must be 'ms0' or 'all'."
            )

        self.system = system

        self.frequency_range = (
            float(f_min),
            float(f_max),
        )

        self.points = int(points)
        self.linewidth = float(linewidth)
        self.contrast = float(contrast)

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

        dimensions = self._dimensions()

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

        return sx, sy, sz

    # =========================================================
    # Microwave drive operator
    # =========================================================

    def microwave_operator(
        self,
    ) -> ComplexArray:
        """
        Return the dimensionless microwave coupling operator.

        For a microwave field direction n:

            H1 proportional to n . S

        The overall microwave amplitude is irrelevant for
        relative transition strengths in this CW model.
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
        ).astype(complex)

        return embedded_operator(
            p0_electron,
            0,
            self._dimensions(),
        )

    # =========================================================
    # Initial populations
    # =========================================================

    def initial_populations(
        self,
        states: ComplexArray,
    ) -> FloatArray:
        """
        Construct phenomenological initial-state populations.

        For initial_manifold="ms0", population is distributed
        according to each eigenstate's ms=0 character.

        For initial_manifold="all", all eigenstates receive
        equal population.
        """

        dimension = states.shape[1]

        if self.initial_manifold == "all":

            return np.full(
                dimension,
                1.0 / dimension,
                dtype=float,
            )

        p0 = self.ms0_projector()

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
                "Could not identify the ms=0 manifold."
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
        """Calculate microwave-accessible transitions."""

        energies, states = eigensystem(
            self.system
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

        half_width = (
            self.linewidth / 2.0
        )

        return (
            half_width**2
            /
            (
                (frequencies - center) ** 2
                + half_width**2
            )
        )

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(
        self,
    ) -> ODMRResult:
        """Simulate the CW-ODMR spectrum."""

        frequencies = np.linspace(
            self.frequency_range[0],
            self.frequency_range[1],
            self.points,
        )

        transitions = (
            self.calculate_transitions()
        )

        absorption = np.zeros_like(
            frequencies
        )

        for transition in transitions:

            weight = (
                transition.strength
                * transition.population_weight
            )

            absorption += (
                weight
                * self._lorentzian(
                    frequencies,
                    transition.frequency,
                )
            )

        # Normalize so "contrast" controls the deepest
        # possible total ODMR dip.

        maximum = np.max(
            absorption
        )

        if maximum > 0:
            absorption = (
                absorption / maximum
            )

        fluorescence = (
            1.0
            - self.contrast
            * absorption
        )

        return ODMRResult(
            frequencies=frequencies,
            fluorescence=fluorescence,
            transitions=transitions,
        )