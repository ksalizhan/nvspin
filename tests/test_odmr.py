import numpy as np
import pytest

from nvspin import NVCenter
from nvspin.experiments import ODMR


def test_odmr_output_shape():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    result = ODMR(
        nv,
        points=1000,
    ).simulate()

    assert result.frequencies.shape == (
        1000,
    )

    assert result.fluorescence.shape == (
        1000,
    )


def test_odmr_baseline():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    result = ODMR(
        nv,
        points=1000,
    ).simulate()

    assert np.max(
        result.fluorescence
    ) <= 1.0 + 1e-12

    assert np.min(
        result.fluorescence
    ) >= 0.0


def test_pure_nv_transition_frequencies():

    D = 2.870
    gamma_e = 28.025
    Bz = 0.005

    nv = NVCenter(
        D=D,
        E=0.0,
        B=[0, 0, Bz],
        nitrogen=None,
    )

    experiment = ODMR(
        nv,
        frequency_range=(
            2.5,
            3.2,
        ),
    )

    transitions = (
        experiment.calculate_transitions()
    )

    frequencies = sorted(
        transition.frequency
        for transition in transitions
        if transition.population_weight > 0
    )

    expected = sorted(
        [
            D - gamma_e * Bz,
            D + gamma_e * Bz,
        ]
    )

    assert len(frequencies) == 2

    np.testing.assert_allclose(
        frequencies,
        expected,
        atol=1e-12,
    )


def test_pure_nv_transverse_strengths_equal():

    nv = NVCenter(
        E=0.0,
        B=[0, 0, 0.005],
        nitrogen=None,
    )

    transitions = ODMR(
        nv,
        microwave_direction=(1, 0, 0),
    ).calculate_transitions()

    strengths = sorted(
        transition.strength
        for transition in transitions
    )

    assert len(strengths) == 2

    np.testing.assert_allclose(
        strengths[0],
        strengths[1],
        atol=1e-12,
    )


def test_n14_produces_hyperfine_transitions():

    nv = NVCenter(
        B=[0, 0, 0.005],
    )

    transitions = ODMR(
        nv,
        frequency_range=(
            2.5,
            3.2,
        ),
    ).calculate_transitions()

    assert len(
        transitions
    ) >= 6


def test_carbon_can_be_added():

    nv = NVCenter(
        B=[0, 0, 0.005],
    )

    nv.add_carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    result = ODMR(
        nv,
        points=500,
    ).simulate()

    assert len(
        result.transitions
    ) > 0


def test_invalid_frequency_range():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        ODMR(
            nv,
            frequency_range=(
                3.0,
                2.0,
            ),
        )


def test_invalid_linewidth():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        ODMR(
            nv,
            linewidth=0.0,
        )


def test_zero_microwave_direction():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        ODMR(
            nv,
            microwave_direction=(
                0,
                0,
                0,
            ),
        )