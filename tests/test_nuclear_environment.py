"""Tests for conditional nuclear environments."""

import numpy as np

from nvspin.physics import (
    Carbon13Spin,
    IndependentNuclearEnvironment,
    subsystem_coherence,
)


def test_zero_hyperfine_carbon_has_unit_coherence():

    carbon = Carbon13Spin(
        larmor_frequency=0.000432,
        A_parallel=0.0,
        A_perpendicular=0.0,
    )

    coherence = subsystem_coherence(
        subsystem=carbon,
        number_of_pulses=8,
        pulse_spacing=1200.0,
    )

    np.testing.assert_allclose(
        coherence,
        1.0,
        atol=1e-12,
    )


def test_single_subsystem_environment_matches_subsystem():

    carbon = Carbon13Spin(
        larmor_frequency=0.000432,
        A_parallel=0.000020,
        A_perpendicular=0.000030,
    )

    single = subsystem_coherence(
        subsystem=carbon,
        number_of_pulses=8,
        pulse_spacing=1200.0,
    )

    environment = (
        IndependentNuclearEnvironment(
            subsystems=(
                carbon,
            )
        )
    )

    combined = environment.coherence(
        number_of_pulses=8,
        pulse_spacing=1200.0,
    )

    np.testing.assert_allclose(
        combined,
        single,
        atol=1e-12,
    )


def test_independent_environment_factorizes():

    carbon_1 = Carbon13Spin(
        larmor_frequency=0.000432,
        A_parallel=-0.0000364,
        A_perpendicular=0.000025,
        label="C1",
    )

    carbon_2 = Carbon13Spin(
        larmor_frequency=0.000432,
        A_parallel=0.0000206,
        A_perpendicular=0.000043,
        label="C2",
    )

    environment = (
        IndependentNuclearEnvironment(
            subsystems=(
                carbon_1,
                carbon_2,
            )
        )
    )

    individual = (
        environment.individual_coherences(
            number_of_pulses=16,
            pulse_spacing=1200.0,
        )
    )

    total = environment.coherence(
        number_of_pulses=16,
        pulse_spacing=1200.0,
    )

    np.testing.assert_allclose(
        total,
        individual[0]
        * individual[1],
        atol=1e-12,
    )


def test_carbon_density_matrix_is_normalized():

    carbon = Carbon13Spin(
        larmor_frequency=0.000432,
        A_parallel=0.0,
        A_perpendicular=0.0,
    )

    np.testing.assert_allclose(
        np.trace(
            carbon.initial_density_matrix
        ),
        1.0,
        atol=1e-12,
    )


def test_complex_coherence_is_physical():

    carbon = Carbon13Spin(
        larmor_frequency=0.000432,
        A_parallel=0.000020,
        A_perpendicular=0.000040,
    )

    coherence = subsystem_coherence(
        subsystem=carbon,
        number_of_pulses=32,
        pulse_spacing=1100.0,
    )

    assert (
        abs(coherence)
        <= 1.0 + 1e-12
    )