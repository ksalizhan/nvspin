import numpy as np
import pytest

from nvspin import NVCenter
from nvspin.experiments import ODMR

from nvspin.readout import (
    NVOpticalCycle,
    NVOpticalRates,
)


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
        microwave_rate=0.01,
        **kwargs,
    )

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
        microwave_direction=(1, 0, 0),
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


def test_invalid_linewidth():

    nv = NVCenter()

    with pytest.raises(
        ValueError
    ):
        build_odmr(
            nv,
            linewidth=0.0,
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
        populations >= -1e-12
    )

    assert np.all(
        populations <= 1.0 + 1e-12
    )


def test_zero_microwave_rate_gives_flat_odmr():

    nv = NVCenter(
        nitrogen=None,
        B=[0, 0, 0.005],
    )

    experiment = ODMR(
        nv,
        optical_cycle=build_optical_cycle(),
        microwave_rate=0.0,
        points=500,
    )

    result = experiment.simulate()

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
        microwave_rate=0.01,
        points=500,
    ).simulate()

    np.testing.assert_allclose(
        result.fluorescence,
        1.0,
        atol=1e-10,
    )