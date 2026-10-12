import numpy as np
import pytest

from nvspin import (
    Nitrogen14,
    Nitrogen15,
    NVCenter,
)


def test_default_nv_has_nitrogen14():
    nv = NVCenter()

    assert isinstance(
        nv.nitrogen,
        Nitrogen14,
    )


def test_default_dimension():
    nv = NVCenter()

    # electron S=1 -> 3
    # 14N I=1    -> 3
    assert nv.dimension == 9


def test_nv_without_nitrogen():
    nv = NVCenter(
        nitrogen=None,
    )

    assert nv.dimension == 3


def test_switch_to_nitrogen15():
    nv = NVCenter()

    nv.set_nitrogen("15N")

    assert isinstance(
        nv.nitrogen,
        Nitrogen15,
    )

    # electron: 3
    # 15N:      2
    assert nv.dimension == 6


def test_add_one_carbon():
    nv = NVCenter()

    carbon = nv.add_carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    assert len(nv.carbons) == 1

    # electron 3 * 14N 3 * 13C 2
    assert nv.dimension == 18

    assert carbon.label == "13C-1"


def test_add_multiple_carbons():
    nv = NVCenter()

    nv.add_carbon13()
    nv.add_carbon13()
    nv.add_carbon13()

    assert len(nv.carbons) == 3

    # 3 * 3 * 2 * 2 * 2
    assert nv.dimension == 72


def test_remove_carbon():
    nv = NVCenter()

    carbon1 = nv.add_carbon13()
    carbon2 = nv.add_carbon13()

    nv.remove_carbon13(carbon1)

    assert len(nv.carbons) == 1
    assert nv.carbons[0] is carbon2


def test_clear_carbons():
    nv = NVCenter()

    nv.add_carbon13()
    nv.add_carbon13()

    nv.clear_carbons()

    assert len(nv.carbons) == 0


def test_set_magnetic_field():
    nv = NVCenter()

    nv.set_magnetic_field(
        0.001,
        0.002,
        0.010,
    )

    np.testing.assert_allclose(
        nv.B,
        [0.001, 0.002, 0.010],
    )


def test_magnetic_field_magnitude():
    nv = NVCenter(
        B=[0.003, 0.004, 0.0],
    )

    assert np.isclose(
        nv.magnetic_field_magnitude,
        0.005,
    )


def test_invalid_nitrogen():
    nv = NVCenter()

    with pytest.raises(ValueError):
        nv.set_nitrogen("16N")