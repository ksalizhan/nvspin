"""Tests for single-carbon DD spectroscopy."""

import numpy as np
import pytest

from nvspin.experiments import (
    SingleCarbon13,
    SingleCarbonDDSpectroscopy,
)


def test_conditional_hamiltonian_dimensions():

    carbon = SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.000040,
        A_perpendicular=0.000030,
    )

    experiment = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=8,
            spacing_range=(
                500.0,
                2000.0,
            ),
            points=10,
        )
    )

    assert experiment.H0.shape == (
        2,
        2,
    )

    assert experiment.H1.shape == (
        2,
        2,
    )


def test_zero_hyperfine_gives_unit_coherence():

    carbon = SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.0,
        A_perpendicular=0.0,
    )

    experiment = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=8,
            spacing_range=(
                500.0,
                2000.0,
            ),
            points=20,
        )
    )

    result = experiment.simulate()

    np.testing.assert_allclose(
        result.coherence,
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.preservation_probability,
        1.0,
        atol=1e-12,
    )


def test_probability_is_physical():

    carbon = SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.000040,
        A_perpendicular=0.000030,
    )

    result = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=32,
            spacing_range=(
                500.0,
                2000.0,
            ),
            points=100,
        ).simulate()
    )

    assert np.all(
        result.preservation_probability
        >= -1e-12
    )

    assert np.all(
        result.preservation_probability
        <= 1.0 + 1e-12
    )


def test_spectroscopy_result_shape():

    carbon = SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.000040,
        A_perpendicular=0.000030,
    )

    result = (
        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=16,
            spacing_range=(
                500.0,
                2000.0,
            ),
            points=123,
        ).simulate()
    )

    assert result.pulse_spacings.shape == (
        123,
    )

    assert result.coherence.shape == (
        123,
    )

    assert (
        result.preservation_probability.shape
        == (123,)
    )


def test_invalid_number_of_pulses():

    carbon = SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.000040,
        A_perpendicular=0.000030,
    )

    with pytest.raises(
        ValueError
    ):

        SingleCarbonDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=0,
            spacing_range=(
                500.0,
                2000.0,
            ),
        )