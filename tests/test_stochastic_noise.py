import numpy as np
import pytest

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


def test_ou_output_shape():

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.001,
        correlation_time=1000.0,
    )

    times, values = noise.generate(
        duration=1000.0,
        dt=10.0,
        rng=np.random.default_rng(1),
    )

    assert times.shape == values.shape

    assert len(times) == 101


def test_ou_reproducible_with_seed():

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.001,
        correlation_time=1000.0,
    )

    _, first = noise.generate(
        duration=1000.0,
        dt=10.0,
        rng=np.random.default_rng(123),
    )

    _, second = noise.generate(
        duration=1000.0,
        dt=10.0,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(
        first,
        second,
        atol=0.0,
    )


def test_zero_sigma_gives_zero_noise():

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.0,
        correlation_time=1000.0,
    )

    _, values = noise.generate(
        duration=1000.0,
        dt=10.0,
        rng=np.random.default_rng(1),
    )

    np.testing.assert_allclose(
        values,
        0.0,
        atol=1e-15,
    )


def test_invalid_sigma():

    with pytest.raises(
        ValueError
    ):
        OrnsteinUhlenbeckNoise(
            sigma=-0.001,
            correlation_time=1000.0,
        )


def test_invalid_correlation_time():

    with pytest.raises(
        ValueError
    ):
        OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=0.0,
        )


def test_long_ou_trace_has_approximately_correct_variance():

    sigma = 0.002

    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma,
        correlation_time=200.0,
    )

    _, values = noise.generate(
        duration=200000.0,
        dt=10.0,
        rng=np.random.default_rng(42),
    )

    # Ignore the first section even though initialization
    # is already stationary.
    values = values[1000:]

    measured_std = np.std(
        values
    )

    assert np.isclose(
        measured_std,
        sigma,
        rtol=0.10,
    )

def test_ou_dd_zero_noise_has_unit_coherence():
    """Zero OU noise must preserve coherence exactly."""

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.0,
        correlation_time=1000.0,
    )

    coherence = noise.coherence_under_dd(
        pulse_times=np.array(
            [
                500.0,
            ]
        ),
        total_time=1000.0,
    )

    np.testing.assert_allclose(
        coherence,
        1.0,
        atol=1e-12,
    )


def test_ou_dd_coherence_is_physical():
    """OU DD coherence must remain between zero and one."""

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.0001,
        correlation_time=1000.0,
    )

    coherence = noise.coherence_under_dd(
        pulse_times=np.array(
            [
                250.0,
                750.0,
            ]
        ),
        total_time=1000.0,
    )

    assert (
        0.0
        <= coherence
        <= 1.0
    )
def test_hahn_refocuses_slow_ou_noise():
    """Hahn echo should suppress sufficiently slow OU dephasing."""

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.001,
        correlation_time=10000.0,
    )

    total_time = 1000.0

    free_induction = noise.coherence_under_dd(
        pulse_times=np.array(
            [],
            dtype=float,
        ),
        total_time=total_time,
    )

    hahn = noise.coherence_under_dd(
        pulse_times=np.array(
            [
                total_time / 2.0,
            ]
        ),
        total_time=total_time,
    )

    assert (
        hahn
        > free_induction
    )