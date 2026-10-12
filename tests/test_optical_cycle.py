import numpy as np
import pytest

from nvspin.readout import (
    NVOpticalCycle,
    NVOpticalRates,
)


def make_rates():

    return NVOpticalRates(
        excitation=0.05,

        radiative_ms0=0.077,
        radiative_ms1=0.077,

        isc_ms0=0.005,
        isc_ms1=0.050,

        singlet_to_ms0=0.004,
        singlet_to_ms1=0.001,
    )


def test_rate_matrix_shape():

    cycle = NVOpticalCycle(
        make_rates()
    )

    assert (
        cycle.rate_matrix().shape
        == (5, 5)
    )


def test_rate_matrix_conserves_population():

    cycle = NVOpticalCycle(
        make_rates()
    )

    M = cycle.rate_matrix()

    np.testing.assert_allclose(
        np.sum(
            M,
            axis=0,
        ),
        0.0,
        atol=1e-12,
    )


def test_propagation_conserves_population():

    cycle = NVOpticalCycle(
        make_rates()
    )

    initial = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    final = cycle.propagate(
        initial,
        1000.0,
    )

    np.testing.assert_allclose(
        np.sum(final),
        1.0,
        atol=1e-12,
    )


def test_steady_state_normalized():

    cycle = NVOpticalCycle(
        make_rates()
    )

    populations = (
        cycle.steady_state()
    )

    np.testing.assert_allclose(
        np.sum(populations),
        1.0,
        atol=1e-12,
    )


def test_optical_pumping_favors_ms0():

    cycle = NVOpticalCycle(
        make_rates()
    )

    populations = (
        cycle.steady_state()
    )

    assert (
        populations[0]
        >
        populations[1]
    )


def test_negative_rate_rejected():

    with pytest.raises(
        ValueError
    ):
        NVOpticalRates(
            excitation=-1.0,
            radiative_ms0=0.077,
            radiative_ms1=0.077,
            isc_ms0=0.005,
            isc_ms1=0.050,
            singlet_to_ms0=0.004,
            singlet_to_ms1=0.001,
        )