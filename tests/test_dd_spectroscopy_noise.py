"""Tests for noisy single-carbon DD spectroscopy."""

import numpy as np

from nvspin.experiments import (
    SingleCarbon13,
    SingleCarbonDDSpectroscopy,
    SingleCarbonNoisyDDSpectroscopy,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


def build_carbon():
    """Return the standard weakly coupled test carbon."""

    return SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.000040,
        A_perpendicular=0.000030,
    )


# ============================================================
# Zero-noise limit
# ============================================================


def test_zero_noise_recovers_clean_spectroscopy():
    """
    With sigma = 0, the OU bath coherence must be unity.

    Therefore the noisy spectroscopy model must reduce
    exactly to the clean single-13C model.
    """

    carbon = build_carbon()

    clean = SingleCarbonDDSpectroscopy(
        carbon=carbon,
        number_of_pulses=16,
        spacing_range=(
            1000.0,
            1230.0,
        ),
        points=40,
    ).simulate()

    noisy = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=16,
            spacing_range=(
                1000.0,
                1230.0,
            ),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.0,
                correlation_time=1000.0,
            ),
            points=40,
        ).simulate()
    )

    np.testing.assert_allclose(
        noisy.bath_coherence,
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        noisy.total_coherence,
        clean.coherence,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        noisy.preservation_probability,
        clean.preservation_probability,
        atol=1e-12,
    )


# ============================================================
# Zero-hyperfine limit
# ============================================================


def test_no_hyperfine_leaves_only_bath():
    """
    With zero hyperfine coupling, the explicit carbon cannot
    distinguish the electron states.

    Therefore

        L_13C = 1

    and

        L_total = L_bath.
    """

    carbon = SingleCarbon13(
        larmor_frequency=0.0004282,
        A_parallel=0.0,
        A_perpendicular=0.0,
    )

    result = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=carbon,
            number_of_pulses=8,
            spacing_range=(
                1000.0,
                1200.0,
            ),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            points=10,
        ).simulate()
    )

    np.testing.assert_allclose(
        result.carbon_coherence,
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.total_coherence,
        result.bath_coherence,
        atol=1e-12,
    )


# ============================================================
# Independent-environment factorization
# ============================================================


def test_total_coherence_factorizes():
    """
    For an independent explicit carbon and classical OU bath,

        L_total = L_13C * L_bath.
    """

    result = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=build_carbon(),
            number_of_pulses=8,
            spacing_range=(
                1000.0,
                1200.0,
            ),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            points=10,
        ).simulate()
    )

    np.testing.assert_allclose(
        result.total_coherence,
        (
            result.carbon_coherence
            * result.bath_coherence
        ),
        atol=1e-12,
    )


# ============================================================
# Physical probability
# ============================================================


def test_probability_is_physical():
    """
    The electron preservation probability must remain
    between zero and one.
    """

    result = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=build_carbon(),
            number_of_pulses=8,
            spacing_range=(
                1000.0,
                1200.0,
            ),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            points=10,
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


# ============================================================
# OU coherence properties
# ============================================================


def test_bath_coherence_is_physical():
    """
    Analytic OU coherence must remain between zero and one.
    """

    result = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=build_carbon(),
            number_of_pulses=16,
            spacing_range=(
                1000.0,
                1200.0,
            ),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.0001,
                correlation_time=1000.0,
            ),
            points=20,
        ).simulate()
    )

    assert np.all(
        result.bath_coherence
        >= 0.0
    )

    assert np.all(
        result.bath_coherence
        <= 1.0
    )


def test_total_times_are_correct():
    """
    For N ideal pulses with center-to-center spacing tau,

        T = N * tau.
    """

    number_of_pulses = 8

    result = (
        SingleCarbonNoisyDDSpectroscopy(
            carbon=build_carbon(),
            number_of_pulses=(
                number_of_pulses
            ),
            spacing_range=(
                1000.0,
                1200.0,
            ),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.0001,
                correlation_time=1000.0,
            ),
            points=10,
        ).simulate()
    )

    np.testing.assert_allclose(
        result.total_times,
        (
            number_of_pulses
            * result.pulse_spacings
        ),
        atol=1e-12,
    )