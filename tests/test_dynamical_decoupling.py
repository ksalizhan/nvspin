"""Tests for dynamical-decoupling simulations."""

import numpy as np
import pytest

from nvspin.control import (
    free_induction,
    pdd,
    xy8,
)

from nvspin.experiments import (
    DynamicalDecoupling,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


# ============================================================
# Existing tau-mode tests
# ============================================================


def test_dd_total_time():
    """
    In tau mode:

        T = N * tau

    For PDD-8 with tau = 100 ns:

        T = 800 ns.
    """

    experiment = DynamicalDecoupling(
        sequence=pdd(8),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=1000.0,
        ),
        tau=100.0,
        realizations=2,
        dt=10.0,
        seed=1,
    )

    assert np.isclose(
        experiment.total_time,
        800.0,
    )


def test_dd_pulse_times():
    """
    For four pulses with tau = 100 ns, the symmetric pulse
    centers should be:

        50, 150, 250, 350 ns.
    """

    experiment = DynamicalDecoupling(
        sequence=pdd(4),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=1000.0,
        ),
        tau=100.0,
        realizations=2,
        dt=10.0,
        seed=1,
    )

    np.testing.assert_allclose(
        experiment.pulse_times,
        [
            50.0,
            150.0,
            250.0,
            350.0,
        ],
        atol=1e-12,
    )


def test_zero_noise_preserves_coherence():
    """
    With no frequency noise and ideal pulses, transverse
    coherence must remain unity.
    """

    experiment = DynamicalDecoupling(
        sequence=xy8(1),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.0,
            correlation_time=1000.0,
        ),
        tau=100.0,
        realizations=5,
        dt=10.0,
        seed=1,
    )

    result = experiment.simulate()

    np.testing.assert_allclose(
        result.coherence,
        1.0,
        atol=1e-12,
    )


def test_dd_result_is_reproducible():
    """
    Identical random seeds must produce identical stochastic
    simulations.
    """

    kwargs = dict(
        sequence=pdd(4),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=1000.0,
        ),
        tau=100.0,
        realizations=10,
        dt=10.0,
        seed=123,
    )

    first = DynamicalDecoupling(
        **kwargs
    ).simulate()

    second = DynamicalDecoupling(
        **kwargs
    ).simulate()

    np.testing.assert_allclose(
        first.average_bloch_vector,
        second.average_bloch_vector,
        atol=0.0,
    )


def test_coherence_is_physical():
    """
    Ensemble transverse coherence must remain between
    zero and one.
    """

    result = DynamicalDecoupling(
        sequence=pdd(4),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.002,
            correlation_time=500.0,
        ),
        tau=100.0,
        realizations=20,
        dt=10.0,
        seed=7,
    ).simulate()

    assert (
        0.0
        <= result.coherence
        <= 1.0 + 1e-12
    )


def test_invalid_tau():
    """tau must be positive."""

    with pytest.raises(
        ValueError
    ):
        DynamicalDecoupling(
            sequence=pdd(4),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            tau=0.0,
        )


# ============================================================
# total_time mode
# ============================================================


def test_total_time_mode():
    """
    In total-time mode:

        tau = T / N

    For

        T = 4000 ns
        N = 8

    we should obtain

        tau = 500 ns.
    """

    experiment = DynamicalDecoupling(
        sequence=pdd(8),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=1000.0,
        ),
        total_time=4000.0,
        realizations=2,
        dt=10.0,
        seed=1,
    )

    assert np.isclose(
        experiment.total_time,
        4000.0,
    )

    assert np.isclose(
        experiment.tau,
        500.0,
    )


def test_total_time_mode_pulse_positions():
    """
    For

        T = 1000 ns
        N = 4

    tau = 250 ns.

    Symmetric pulse centers are therefore:

        125, 375, 625, 875 ns.
    """

    experiment = DynamicalDecoupling(
        sequence=pdd(4),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=1000.0,
        ),
        total_time=1000.0,
        realizations=2,
        dt=10.0,
        seed=1,
    )

    np.testing.assert_allclose(
        experiment.pulse_times,
        [
            125.0,
            375.0,
            625.0,
            875.0,
        ],
        atol=1e-12,
    )


# ============================================================
# Timing-interface validation
# ============================================================


def test_tau_and_total_time_cannot_both_be_given():
    """
    tau and total_time represent alternative timing modes
    and must not be supplied simultaneously.
    """

    with pytest.raises(
        ValueError
    ):
        DynamicalDecoupling(
            sequence=pdd(4),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            tau=100.0,
            total_time=400.0,
        )


def test_timing_parameter_required():
    """
    Exactly one timing specification must be supplied.
    """

    with pytest.raises(
        ValueError
    ):
        DynamicalDecoupling(
            sequence=pdd(4),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
        )


def test_invalid_total_time():
    """total_time must be positive."""

    with pytest.raises(
        ValueError
    ):
        DynamicalDecoupling(
            sequence=pdd(4),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            total_time=0.0,
        )


# ============================================================
# tau mode and total_time mode equivalence
# ============================================================


def test_tau_and_total_time_modes_are_equivalent():
    """
    These two specifications describe exactly the same
    experiment:

        N = 8, tau = 100 ns

    and

        N = 8, T = 800 ns.

    With the same random seed they must produce identical
    stochastic evolution.
    """

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.001,
        correlation_time=1000.0,
    )

    tau_mode = DynamicalDecoupling(
        sequence=pdd(8),
        noise=noise,
        tau=100.0,
        realizations=20,
        dt=10.0,
        seed=123,
    ).simulate()

    total_time_mode = (
        DynamicalDecoupling(
            sequence=pdd(8),
            noise=noise,
            total_time=800.0,
            realizations=20,
            dt=10.0,
            seed=123,
        ).simulate()
    )

    np.testing.assert_allclose(
        tau_mode.average_bloch_vector,
        total_time_mode.average_bloch_vector,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        tau_mode.coherence,
        total_time_mode.coherence,
        atol=1e-12,
    )


# ============================================================
# Free-induction reference
# ============================================================


def test_free_induction_total_time():
    """
    Free induction contains no refocusing pulses.

    It therefore requires total_time mode rather than
    interpulse-spacing mode.
    """

    experiment = DynamicalDecoupling(
        sequence=free_induction(),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.001,
            correlation_time=1000.0,
        ),
        total_time=1000.0,
        realizations=5,
        dt=10.0,
        seed=1,
    )

    assert np.isclose(
        experiment.total_time,
        1000.0,
    )

    assert (
        experiment.tau is None
    )

    assert (
        len(
            experiment.pulse_times
        )
        == 0
    )


def test_free_induction_rejects_tau_mode():
    """
    A zero-pulse sequence has no interpulse spacing.
    """

    with pytest.raises(
        ValueError
    ):
        DynamicalDecoupling(
            sequence=free_induction(),
            noise=OrnsteinUhlenbeckNoise(
                sigma=0.001,
                correlation_time=1000.0,
            ),
            tau=100.0,
        )


def test_free_induction_zero_noise_preserves_coherence():
    """
    Free evolution without noise must preserve transverse
    coherence.
    """

    result = DynamicalDecoupling(
        sequence=free_induction(),
        noise=OrnsteinUhlenbeckNoise(
            sigma=0.0,
            correlation_time=1000.0,
        ),
        total_time=1000.0,
        realizations=5,
        dt=10.0,
        seed=1,
    ).simulate()

    np.testing.assert_allclose(
        result.coherence,
        1.0,
        atol=1e-12,
    )