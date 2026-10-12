import numpy as np

from nvspin.experiments import Ramsey


def test_ramsey_output_shape():

    result = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        points=501,
    ).simulate()

    assert result.delays.shape == (501,)
    assert result.ground_population.shape == (501,)
    assert result.excited_population.shape == (501,)


def test_ramsey_population_conservation():

    result = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        points=501,
    ).simulate()

    total = (
        result.ground_population
        + result.excited_population
    )

    np.testing.assert_allclose(
        total,
        1.0,
        atol=1e-12,
    )


def test_ramsey_detuning():

    experiment = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
    )

    assert np.isclose(
        experiment.detuning,
        0.005,
    )


def test_pi_over_2_duration():

    experiment = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
    )

    assert np.isclose(
        experiment.pi_over_2_duration,
        25.0,
    )


def test_resonant_ramsey_is_constant():

    result = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.870,
        rabi_frequency=0.010,
        delay_range=(0.0, 1000.0),
        points=501,
    ).simulate()

    # Two identical pi/2 pulses produce a pi rotation.
    # Therefore the ideal final ground-state population
    # remains zero for all tau when Delta = 0.

    np.testing.assert_allclose(
        result.ground_population,
        0.0,
        atol=1e-11,
    )