import numpy as np

from nvspin.control.two_level import (
    rotating_frame_hamiltonian,
)
from nvspin.solvers import propagate_state


def test_rotating_frame_hamiltonian_is_hermitian():

    H = rotating_frame_hamiltonian(
        transition_frequency=2.870,
        microwave_frequency=2.860,
        rabi_frequency=0.010,
    )

    np.testing.assert_allclose(
        H,
        H.conj().T,
        atol=1e-12,
    )


def test_resonant_x_drive():

    H = rotating_frame_hamiltonian(
        transition_frequency=2.870,
        microwave_frequency=2.870,
        rabi_frequency=0.010,
        phase=0.0,
    )

    expected = np.array(
        [
            [0.0, 0.005],
            [0.005, 0.0],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(
        H,
        expected,
        atol=1e-12,
    )


def test_pi_pulse_population_transfer():

    H = rotating_frame_hamiltonian(
        transition_frequency=2.870,
        microwave_frequency=2.870,
        rabi_frequency=0.010,
    )

    initial_state = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    final_state = propagate_state(
        initial_state,
        H,
        50.0,
    )

    populations = (
        np.abs(final_state) ** 2
    )

    np.testing.assert_allclose(
        populations,
        [0.0, 1.0],
        atol=1e-12,
    )


def test_pi_over_2_population():

    H = rotating_frame_hamiltonian(
        transition_frequency=2.870,
        microwave_frequency=2.870,
        rabi_frequency=0.010,
    )

    initial_state = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    final_state = propagate_state(
        initial_state,
        H,
        25.0,
    )

    populations = (
        np.abs(final_state) ** 2
    )

    np.testing.assert_allclose(
        populations,
        [0.5, 0.5],
        atol=1e-12,
    )


def test_full_rabi_period():

    H = rotating_frame_hamiltonian(
        transition_frequency=2.870,
        microwave_frequency=2.870,
        rabi_frequency=0.010,
    )

    initial_state = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    final_state = propagate_state(
        initial_state,
        H,
        100.0,
    )

    populations = (
        np.abs(final_state) ** 2
    )

    np.testing.assert_allclose(
        populations,
        [1.0, 0.0],
        atol=1e-12,
    )


def test_off_resonant_drive_reduces_transfer():

    H = rotating_frame_hamiltonian(
        transition_frequency=2.870,
        microwave_frequency=2.850,
        rabi_frequency=0.010,
    )

    initial_state = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    final_state = propagate_state(
        initial_state,
        H,
        50.0,
    )

    excited_population = (
        np.abs(final_state[1]) ** 2
    )

    assert excited_population < 1.0