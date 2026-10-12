import numpy as np
import pytest

from nvspin.control import (
    FreeEvolution,
    MicrowavePulse,
    PulseSequence,
    pi_over_2_pulse,
    pi_pulse,
)


def test_pi_pulse_duration():

    pulse = pi_pulse(
        frequency=2.870,
        rabi_frequency=0.010,
    )

    assert np.isclose(
        pulse.duration,
        50.0,
    )


def test_pi_over_2_pulse_duration():

    pulse = pi_over_2_pulse(
        frequency=2.870,
        rabi_frequency=0.010,
    )

    assert np.isclose(
        pulse.duration,
        25.0,
    )


def test_y_pulse_phase():

    pulse = pi_pulse(
        frequency=2.870,
        rabi_frequency=0.010,
        phase=np.pi / 2,
    )

    assert np.isclose(
        pulse.phase_degrees,
        90.0,
    )


def test_negative_pulse_duration_rejected():

    with pytest.raises(
        ValueError
    ):
        MicrowavePulse(
            duration=-10.0,
            frequency=2.870,
            rabi_frequency=0.010,
        )


def test_negative_free_evolution_rejected():

    with pytest.raises(
        ValueError
    ):
        FreeEvolution(
            duration=-1.0
        )


def test_sequence_duration():

    sequence = PulseSequence()

    sequence.add_pulse(
        pi_over_2_pulse(
            frequency=2.870,
            rabi_frequency=0.010,
        )
    )

    sequence.add_free_evolution(
        100.0
    )

    sequence.add_pulse(
        pi_pulse(
            frequency=2.870,
            rabi_frequency=0.010,
        )
    )

    # 25 ns + 100 ns + 50 ns
    assert np.isclose(
        sequence.total_duration,
        175.0,
    )