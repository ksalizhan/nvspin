"""Tests for effective 13C-13C nuclear-spin pairs."""

import numpy as np

from nvspin.physics import (
    Carbon13Pair,
    IndependentNuclearEnvironment,
    subsystem_coherence,
)


def test_pair_hamiltonian_dimensions():

    pair = Carbon13Pair(
        X=0.0000002367,
        Z=0.000007890,
    )

    assert pair.H0.shape == (
        2,
        2,
    )

    assert pair.H1.shape == (
        2,
        2,
    )


def test_pair_density_matrix_is_normalized():

    pair = Carbon13Pair(
        X=0.0000002367,
        Z=0.000007890,
    )

    np.testing.assert_allclose(
        np.trace(
            pair.initial_density_matrix
        ),
        1.0,
        atol=1e-12,
    )


def test_pair_conditional_frequencies():

    pair = Carbon13Pair(
        X=0.0000002367,
        Z=0.000007890,
    )

    np.testing.assert_allclose(
        pair.frequency_0,
        abs(pair.X),
        atol=1e-15,
    )

    np.testing.assert_allclose(
        pair.frequency_1,
        np.sqrt(
            pair.X**2
            + pair.Z**2
        ),
        atol=1e-15,
    )


def test_zero_Z_gives_unit_coherence():
    """
    If Z = 0 then H0 = H1.

    The pair evolution cannot distinguish the two NV
    electron branches, so electron coherence must remain 1.
    """

    pair = Carbon13Pair(
        X=0.0000002367,
        Z=0.0,
    )

    coherence = subsystem_coherence(
        subsystem=pair,
        number_of_pulses=16,
        pulse_spacing=50000.0,
    )

    np.testing.assert_allclose(
        coherence,
        1.0,
        atol=1e-12,
    )


def test_pair_coherence_is_physical():

    pair = Carbon13Pair(
        X=0.0000002367,
        Z=0.000007890,
    )

    coherence = subsystem_coherence(
        subsystem=pair,
        number_of_pulses=16,
        pulse_spacing=100000.0,
    )

    assert (
        abs(coherence)
        <= 1.0 + 1e-12
    )


def test_pair_works_in_environment():

    pair = Carbon13Pair(
        X=0.0000002367,
        Z=0.000007890,
    )

    environment = (
        IndependentNuclearEnvironment(
            subsystems=(
                pair,
            )
        )
    )

    single = subsystem_coherence(
        subsystem=pair,
        number_of_pulses=16,
        pulse_spacing=100000.0,
    )

    combined = environment.coherence(
        number_of_pulses=16,
        pulse_spacing=100000.0,
    )

    np.testing.assert_allclose(
        combined,
        single,
        atol=1e-12,
    )