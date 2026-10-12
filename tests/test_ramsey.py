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
def test_ramsey_no_decoherence_matches_coherent_result():
    result = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        delay_range=(0.0, 2000.0),
        points=501,
        T2_star=None,
    ).simulate()

    np.testing.assert_allclose(
        result.ground_population,
        result.coherent_ground_population,
        atol=1e-12,
    )
def test_ramsey_long_time_approaches_half_population():
    result = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        delay_range=(0.0, 10000.0),
        points=501,
        T2_star=500.0,
        decay_exponent=1.0,
    ).simulate()

    assert np.isclose(
        result.ground_population[-1],
        0.5,
        atol=1e-6,
    )
def test_ramsey_population_conserved_with_dephasing():

    result = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        delay_range=(0.0, 5000.0),
        points=501,
        T2_star=1000.0,
        decay_exponent=2.0,
    ).simulate()

    np.testing.assert_allclose(
        result.ground_population
        + result.excited_population,
        1.0,
        atol=1e-12,
    )


def test_ramsey_exponential_envelope_at_T2_star():

    T2_star = 1000.0

    experiment = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        T2_star=T2_star,
        decay_exponent=1.0,
    )

    envelope = experiment._coherence_envelope(
        np.array([T2_star])
    )

    np.testing.assert_allclose(
        envelope,
        [np.exp(-1.0)],
        atol=1e-12,
    )


def test_ramsey_gaussian_envelope_at_T2_star():

    T2_star = 1000.0

    experiment = Ramsey(
        transition_frequency=2.870,
        microwave_frequency=2.865,
        rabi_frequency=0.010,
        T2_star=T2_star,
        decay_exponent=2.0,
    )

    envelope = experiment._coherence_envelope(
        np.array([T2_star])
    )

    np.testing.assert_allclose(
        envelope,
        [np.exp(-1.0)],
        atol=1e-12,
    )