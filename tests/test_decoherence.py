import numpy as np
import pytest

from nvspin.noise import (
    DecoherenceModel,
)


def test_empty_decoherence_model():

    noise = DecoherenceModel()

    assert noise.gamma1 == 0.0
    assert noise.gamma_phi == 0.0
    assert np.isinf(noise.T2)

    assert (
        noise.collapse_operators()
        == []
    )


def test_T1_rate():

    noise = DecoherenceModel(
        T1=1000.0,
    )

    assert np.isclose(
        noise.gamma1,
        0.001,
    )


def test_Tphi_rate():

    noise = DecoherenceModel(
        T_phi=500.0,
    )

    assert np.isclose(
        noise.gamma_phi,
        0.002,
    )


def test_effective_T2():

    noise = DecoherenceModel(
        T1=1000.0,
        T_phi=500.0,
    )

    expected_rate = (
        1.0 / (2.0 * 1000.0)
        +
        1.0 / 500.0
    )

    expected_T2 = (
        1.0 / expected_rate
    )

    assert np.isclose(
        noise.T2,
        expected_T2,
    )


def test_invalid_T1():

    with pytest.raises(
        ValueError
    ):
        DecoherenceModel(
            T1=0.0
        )


def test_invalid_Tphi():

    with pytest.raises(
        ValueError
    ):
        DecoherenceModel(
            T_phi=-10.0
        )