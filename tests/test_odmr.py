import numpy as np
import pytest

from nvspin import NVCenter
from nvspin.experiments import ODMR

from nvspin.readout import (
    NVOpticalCycle,
    NVOpticalRates,
)


# ============================================================
# Test helpers
# ============================================================


def build_optical_cycle():

    rates = NVOpticalRates(
        excitation=0.05,
        radiative_ms0=0.077,
        radiative_ms1=0.077,
        isc_ms0=0.005,
        isc_ms1=0.050,
        singlet_to_ms0=0.004,
        singlet_to_ms1=0.001,
    )

    return NVOpticalCycle(
        rates
    )


def build_odmr(
    nv,
    **kwargs,
):

    return ODMR(
        nv,
        optical_cycle=build_optical_cycle(),

        # 5 MHz reference Rabi frequency.
        rabi_frequency=0.005,

        # Homogeneous transverse coherence time.
        #
        # T2 = 300 ns corresponds to weak-drive
        # homogeneous FWHM
        #
        #     1 / (pi T2)
        #
        # ~= 1.06 MHz.
        T2=300.0,

        **kwargs,
    )


# ============================================================
# Basic output
# ============================================================


def test_odmr_output_shape():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    result = build_odmr(
        nv,
        points=1000,
    ).simulate()

    assert result.frequencies.shape == (
        1000,
    )

    assert result.fluorescence.shape == (
        1000,
    )


def test_odmr_baseline():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    result = build_odmr(
        nv,
        points=1000,
    ).simulate()

    assert np.max(
        result.fluorescence
    ) <= 1.0 + 1e-12

    assert np.min(
        result.fluorescence
    ) >= 0.0


# ============================================================
# Hamiltonian / transition tests
# ============================================================


def test_pure_nv_transition_frequencies():

    D = 2.870
    gamma_e = 28.025
    Bz = 0.005

    nv = NVCenter(
        D=D,
        E=0.0,
        B=[0, 0, Bz],
        nitrogen=None,
    )

    experiment = build_odmr(
        nv,
        frequency_range=(
            2.5,
            3.2,
        ),
    )

    transitions = (
        experiment.calculate_transitions()
    )

    frequencies = sorted(
        transition.frequency
        for transition in transitions
        if transition.population_weight > 0
    )

    expected = sorted(
        [
            D - gamma_e * Bz,
            D + gamma_e * Bz,
        ]
    )

    assert len(frequencies) == 2

    np.testing.assert_allclose(
        frequencies,
        expected,
        atol=1e-12,
    )


def test_pure_nv_transverse_strengths_equal():

    nv = NVCenter(
        E=0.0,
        B=[0, 0, 0.005],
        nitrogen=None,
    )

    transitions = build_odmr(
        nv,
        microwave_direction=(
            1,
            0,
            0,
        ),
    ).calculate_transitions()

    strengths = sorted(
        transition.strength
        for transition in transitions
    )

    assert len(strengths) == 2

    np.testing.assert_allclose(
        strengths[0],
        strengths[1],
        atol=1e-12,
    )


def test_n14_produces_hyperfine_transitions():

    nv = NVCenter(
        B=[0, 0, 0.005],
    )

    transitions = build_odmr(
        nv,
        frequency_range=(
            2.5,
            3.2,
        ),
    ).calculate_transitions()

    assert len(
        transitions
    ) >= 6


def test_carbon_can_be_added():

    nv = NVCenter(
        B=[0, 0, 0.005],
    )

    nv.add_carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    result = build_odmr(
        nv,
        points=500,
    ).simulate()

    assert len(
        result.transitions
    ) > 0


# ============================================================
# Constructor validation
# ============================================================


def test_invalid_frequency_range():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        build_odmr(
            nv,
            frequency_range=(
                3.0,
                2.0,
            ),
        )


def test_invalid_rabi_frequency():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        ODMR(
            nv,
            optical_cycle=build_optical_cycle(),
            rabi_frequency=-0.001,
            T2=300.0,
        )


def test_invalid_t2():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        ODMR(
            nv,
            optical_cycle=build_optical_cycle(),
            rabi_frequency=0.005,
            T2=0.0,
        )


def test_zero_microwave_direction():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        build_odmr(
            nv,
            microwave_direction=(
                0,
                0,
                0,
            ),
        )


# ============================================================
# Optical rate-equation tests
# ============================================================


def test_odmr_rate_matrix_conserves_population():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    rate_matrix = (
        experiment._rate_matrix(
            microwave_mixing=0.01
        )
    )

    np.testing.assert_allclose(
        np.sum(
            rate_matrix,
            axis=0,
        ),
        0.0,
        atol=1e-12,
    )


def test_odmr_steady_state_is_physical():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    rate_matrix = (
        experiment._rate_matrix(
            microwave_mixing=0.01
        )
    )

    populations = (
        experiment._steady_state(
            rate_matrix
        )
    )

    np.testing.assert_allclose(
        np.sum(
            populations
        ),
        1.0,
        atol=1e-12,
    )

    assert np.all(
        populations
        >= -1e-12
    )

    assert np.all(
        populations
        <= 1.0 + 1e-12
    )


# ============================================================
# T2-derived microwave response
# ============================================================


def test_t2_sets_weak_drive_linewidth():

    nv = NVCenter(
        nitrogen=None,
    )

    experiment = build_odmr(
        nv
    )

    expected = (
        1.0
        /
        (
            np.pi
            * 300.0
        )
    )

    np.testing.assert_allclose(
        experiment.weak_drive_fwhm,
        expected,
        atol=1e-15,
    )


def test_transition_rabi_frequency_for_pure_nv():

    nv = NVCenter(
        E=0.0,
        B=[0, 0, 0.005],
        nitrogen=None,
    )

    experiment = build_odmr(
        nv,
        microwave_direction=(
            1.0,
            0.0,
            0.0,
        ),
    )

    transitions = (
        experiment.calculate_transitions()
    )

    assert len(
        transitions
    ) == 2

    for transition in transitions:

        omega = (
            experiment
            ._transition_rabi_frequency(
                transition
            )
        )

        np.testing.assert_allclose(
            omega,
            experiment.rabi_frequency,
            atol=1e-12,
        )


def test_transition_mixing_rate_is_maximal_on_resonance():

    nv = NVCenter(
        E=0.0,
        B=[0, 0, 0.005],
        nitrogen=None,
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    on_resonance = (
        experiment._transition_mixing_rate(
            transition.frequency,
            transition,
        )
    )

    off_resonance = (
        experiment._transition_mixing_rate(
            transition.frequency
            + 0.01,
            transition,
        )
    )

    assert (
        on_resonance
        > off_resonance
    )


def test_transition_mixing_rate_is_symmetric_in_detuning():

    nv = NVCenter(
        E=0.0,
        B=[0, 0, 0.005],
        nitrogen=None,
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    detuning = 0.003

    lower = (
        experiment._transition_mixing_rate(
            transition.frequency
            - detuning,
            transition,
        )
    )

    upper = (
        experiment._transition_mixing_rate(
            transition.frequency
            + detuning,
            transition,
        )
    )

    np.testing.assert_allclose(
        lower,
        upper,
        atol=1e-15,
    )


def test_transition_mixing_rate_half_max_at_t2_half_width():

    nv = NVCenter(
        E=0.0,
        B=[0, 0, 0.005],
        nitrogen=None,
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    on_resonance = (
        experiment._transition_mixing_rate(
            transition.frequency,
            transition,
        )
    )

    half_width = (
        1.0
        /
        (
            2.0
            * np.pi
            * experiment.T2
        )
    )

    at_half_width = (
        experiment._transition_mixing_rate(
            transition.frequency
            + half_width,
            transition,
        )
    )

    np.testing.assert_allclose(
        at_half_width,
        0.5 * on_resonance,
        rtol=1e-12,
        atol=1e-15,
    )


# ============================================================
# ODMR physics
# ============================================================


def test_zero_rabi_frequency_gives_flat_odmr():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = ODMR(
        nv,
        optical_cycle=build_optical_cycle(),
        rabi_frequency=0.0,
        T2=300.0,
        points=500,
    )

    result = (
        experiment.simulate()
    )

    np.testing.assert_allclose(
        result.fluorescence,
        1.0,
        atol=1e-12,
    )


def test_spin_dependent_isc_produces_odmr_dip():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    result = build_odmr(
        nv,
        points=1000,
        frequency_range=(
            2.65,
            3.10,
        ),
    ).simulate()

    assert np.min(
        result.fluorescence
    ) < 1.0 - 1e-8


def test_spin_symmetric_optical_cycle_has_no_odmr_contrast():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    rates = NVOpticalRates(
        excitation=0.05,

        radiative_ms0=0.077,
        radiative_ms1=0.077,

        isc_ms0=0.020,
        isc_ms1=0.020,

        singlet_to_ms0=0.002,
        singlet_to_ms1=0.002,
    )

    optical_cycle = (
        NVOpticalCycle(
            rates
        )
    )

    result = ODMR(
        nv,
        optical_cycle=optical_cycle,
        rabi_frequency=0.005,
        T2=300.0,
        points=500,
    ).simulate()

    np.testing.assert_allclose(
        result.fluorescence,
        1.0,
        atol=1e-10,
    )