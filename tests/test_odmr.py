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
        rabi_frequency=0.005,
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
# Transition-specific Rabi frequency
# ============================================================


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


# ============================================================
# Seven-variable optical-Bloch matrix
# ============================================================


def test_bloch_matrix_has_correct_shape():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    matrix = (
        experiment._bloch_rate_matrix(
            transition.frequency,
            transition,
        )
    )

    assert matrix.shape == (
        7,
        7,
    )


def test_bloch_steady_state_is_physical():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    matrix = (
        experiment._bloch_rate_matrix(
            transition.frequency,
            transition,
        )
    )

    state = (
        experiment._bloch_steady_state(
            matrix
        )
    )

    populations = (
        state[:5]
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

    assert np.all(
        np.isfinite(
            state
        )
    )


def test_bloch_population_equations_conserve_probability():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    matrix = (
        experiment._bloch_rate_matrix(
            transition.frequency,
            transition,
        )
    )

    # Sum only the five population equations.
    #
    # The resulting row vector must vanish, including
    # the microwave-coherence columns. This expresses
    #
    # d/dt (
    #     p_g0
    #     + p_g1
    #     + p_e0
    #     + p_e1
    #     + p_s
    # ) = 0.
    population_sum_derivative = (
        np.sum(
            matrix[:5, :],
            axis=0,
        )
    )

    np.testing.assert_allclose(
        population_sum_derivative,
        0.0,
        atol=1e-12,
    )


def test_bloch_matrix_zero_rabi_decouples_coherences():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = ODMR(
        nv,
        optical_cycle=build_optical_cycle(),
        rabi_frequency=0.0,
        T2=300.0,
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    matrix = (
        experiment._bloch_rate_matrix(
            transition.frequency,
            transition,
        )
    )

    np.testing.assert_allclose(
        matrix[0, 6],
        0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        matrix[1, 6],
        0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        matrix[6, 0],
        0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        matrix[6, 1],
        0.0,
        atol=1e-15,
    )


def test_bloch_matrix_has_zero_detuning_on_resonance():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    matrix = (
        experiment._bloch_rate_matrix(
            transition.frequency,
            transition,
        )
    )

    # On resonance:
    #
    # Delta = 0
    #
    # so the u-v detuning coupling vanishes.
    np.testing.assert_allclose(
        matrix[5, 6],
        0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        matrix[6, 5],
        0.0,
        atol=1e-15,
    )


def test_bloch_microwave_population_terms_cancel():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = build_odmr(
        nv
    )

    transition = (
        experiment.calculate_transitions()[0]
    )

    matrix = (
        experiment._bloch_rate_matrix(
            transition.frequency,
            transition,
        )
    )

    # The microwave contribution from v transfers
    # population between g0 and g1 but cannot create
    # or destroy total population.
    np.testing.assert_allclose(
        matrix[0, 6]
        + matrix[1, 6],
        0.0,
        atol=1e-15,
    )


# ============================================================
# Transverse coherence / optical broadening
# ============================================================


def test_transverse_decay_includes_optical_excitation():

    nv = NVCenter(
        nitrogen=None,
    )

    experiment = build_odmr(
        nv
    )

    k_ex = (
        experiment
        .optical_cycle
        .rates
        .excitation
    )

    expected = (
        1.0 / experiment.T2
        + k_ex
    )

    np.testing.assert_allclose(
        experiment.transverse_decay_rate,
        expected,
        atol=1e-15,
    )


def test_effective_t2_includes_optical_excitation():

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
            1.0 / experiment.T2
            + experiment
            .optical_cycle
            .rates
            .excitation
        )
    )

    np.testing.assert_allclose(
        experiment.effective_T2,
        expected,
        atol=1e-15,
    )


def test_cw_weak_drive_linewidth_includes_optical_dephasing():

    nv = NVCenter(
        nitrogen=None,
    )

    experiment = build_odmr(
        nv
    )

    k_ex = (
        experiment
        .optical_cycle
        .rates
        .excitation
    )

    expected_gamma_2 = (
        1.0 / experiment.T2
        + k_ex
    )

    expected_fwhm = (
        expected_gamma_2
        / np.pi
    )

    np.testing.assert_allclose(
        experiment.transverse_decay_rate,
        expected_gamma_2,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        experiment.weak_drive_fwhm,
        expected_fwhm,
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