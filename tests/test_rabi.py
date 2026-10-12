import numpy as np

from nvspin.experiments import Rabi


def test_rabi_output_shape():

    result = Rabi(
        transition_frequency=2.870,
        rabi_frequency=0.010,
        points=501,
    ).simulate()

    assert result.durations.shape == (
        501,
    )

    assert result.ground_population.shape == (
        501,
    )

    assert result.excited_population.shape == (
        501,
    )


def test_rabi_population_conservation():

    result = Rabi(
        transition_frequency=2.870,
        rabi_frequency=0.010,
        points=501,
    ).simulate()

    total = (
        result.ground_population
        +
        result.excited_population
    )

    np.testing.assert_allclose(
        total,
        1.0,
        atol=1e-12,
    )


def test_resonant_rabi_pi_pulse():

    result = Rabi(
        transition_frequency=2.870,
        rabi_frequency=0.010,
        duration_range=(
            0.0,
            50.0,
        ),
        points=501,
    ).simulate()

    assert np.isclose(
        result.ground_population[-1],
        0.0,
        atol=1e-12,
    )

    assert np.isclose(
        result.excited_population[-1],
        1.0,
        atol=1e-12,
    )


def test_resonant_rabi_full_period():

    result = Rabi(
        transition_frequency=2.870,
        rabi_frequency=0.010,
        duration_range=(
            0.0,
            100.0,
        ),
        points=501,
    ).simulate()

    assert np.isclose(
        result.ground_population[-1],
        1.0,
        atol=1e-12,
    )


def test_analytical_rabi_solution():

    f_rabi = 0.010

    result = Rabi(
        transition_frequency=2.870,
        rabi_frequency=f_rabi,
        duration_range=(
            0.0,
            300.0,
        ),
        points=601,
    ).simulate()

    expected = np.cos(
        np.pi
        * f_rabi
        * result.durations
    ) ** 2

    np.testing.assert_allclose(
        result.ground_population,
        expected,
        atol=1e-11,
    )


def test_detuning_reduces_maximum_transfer():

    result = Rabi(
        transition_frequency=2.870,
        microwave_frequency=2.850,
        rabi_frequency=0.010,
        duration_range=(
            0.0,
            500.0,
        ),
        points=1001,
    ).simulate()

    assert np.max(
        result.excited_population
    ) < 1.0