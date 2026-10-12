import numpy as np

from nvspin.experiments import HahnEcho


def test_hahn_output_shape():

    result = HahnEcho(
        free_detuning=0.0,
        rabi_frequency=0.010,
        points=501,
    ).simulate()

    assert result.delays.shape == (501,)
    assert result.total_free_evolution_times.shape == (501,)
    assert result.ground_population.shape == (501,)
    assert result.excited_population.shape == (501,)


def test_hahn_population_conservation():

    result = HahnEcho(
        free_detuning=0.005,
        rabi_frequency=0.010,
        points=501,
    ).simulate()

    np.testing.assert_allclose(
        result.ground_population
        + result.excited_population,
        1.0,
        atol=1e-12,
    )


def test_hahn_pulse_durations():

    experiment = HahnEcho(
        free_detuning=0.005,
        rabi_frequency=0.010,
    )

    assert np.isclose(
        experiment.pi_over_2_duration,
        25.0,
    )

    assert np.isclose(
        experiment.pi_duration,
        50.0,
    )


def test_total_free_evolution_time():

    result = HahnEcho(
        free_detuning=0.005,
        rabi_frequency=0.010,
        delay_range=(0.0, 1000.0),
        points=101,
    ).simulate()

    np.testing.assert_allclose(
        result.total_free_evolution_times,
        2.0 * result.delays,
        atol=1e-12,
    )


def test_resonant_hahn_is_constant():

    result = HahnEcho(
        free_detuning=0.0,
        pulse_detuning=0.0,
        rabi_frequency=0.010,
        delay_range=(0.0, 1000.0),
        points=101,
    ).simulate()

    np.testing.assert_allclose(
        result.ground_population,
        1.0,
        atol=1e-11,
    )


def test_finite_resonant_pulses_refocus_static_detuning():
    """
    Static free-precession detuning should be refocused when
    the finite microwave pulses themselves are perfectly
    resonant.
    """

    result = HahnEcho(
        free_detuning=0.005,
        pulse_detuning=0.0,
        rabi_frequency=0.010,
        delay_range=(0.0, 2000.0),
        points=501,
    ).simulate()

    np.testing.assert_allclose(
        result.ground_population,
        1.0,
        atol=1e-10,
    )


def test_pulse_detuning_produces_imperfect_echo():
    """
    A nonzero pulse detuning should make the finite microwave
    rotations imperfect.
    """

    result = HahnEcho(
        free_detuning=0.005,
        pulse_detuning=0.005,
        rabi_frequency=0.010,
        delay_range=(0.0, 2000.0),
        points=501,
    ).simulate()

    assert np.min(
        result.ground_population
    ) < 0.99


def test_zero_pulse_detuning_better_than_large_pulse_detuning():

    ideal_pulses = HahnEcho(
        free_detuning=0.005,
        pulse_detuning=0.0,
        rabi_frequency=0.010,
        delay_range=(0.0, 2000.0),
        points=501,
    ).simulate()

    detuned_pulses = HahnEcho(
        free_detuning=0.005,
        pulse_detuning=0.005,
        rabi_frequency=0.010,
        delay_range=(0.0, 2000.0),
        points=501,
    ).simulate()

    ideal_error = np.max(
        np.abs(
            ideal_pulses.ground_population
            - 1.0
        )
    )

    detuned_error = np.max(
        np.abs(
            detuned_pulses.ground_population
            - 1.0
        )
    )

    assert ideal_error < detuned_error