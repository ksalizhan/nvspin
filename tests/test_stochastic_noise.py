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