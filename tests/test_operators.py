import numpy as np
import pytest

from nvspin.physics.operators import (
    nv_electron_operators,
    spin_operators,
)


def test_spin_one_dimensions():
    sx, sy, sz = spin_operators(1)

    assert sx.shape == (3, 3)
    assert sy.shape == (3, 3)
    assert sz.shape == (3, 3)


def test_spin_half_dimensions():
    sx, sy, sz = spin_operators(0.5)

    assert sx.shape == (2, 2)
    assert sy.shape == (2, 2)
    assert sz.shape == (2, 2)


@pytest.mark.parametrize(
    "spin",
    [0.5, 1.0, 1.5, 2.0],
)
def test_spin_commutation_relations(spin):
    sx, sy, sz = spin_operators(spin)

    np.testing.assert_allclose(
        sx @ sy - sy @ sx,
        1j * sz,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        sy @ sz - sz @ sy,
        1j * sx,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        sz @ sx - sx @ sz,
        1j * sy,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "spin",
    [0.5, 1.0, 1.5, 2.0],
)
def test_spin_squared(spin):
    sx, sy, sz = spin_operators(spin)

    s_squared = (
        sx @ sx
        + sy @ sy
        + sz @ sz
    )

    expected = (
        spin
        * (spin + 1.0)
        * np.eye(int(2 * spin + 1))
    )

    np.testing.assert_allclose(
        s_squared,
        expected,
        atol=1e-12,
    )


def test_nv_electron_operator_dimensions():
    sx, sy, sz = nv_electron_operators()

    assert sx.shape == (3, 3)
    assert sy.shape == (3, 3)
    assert sz.shape == (3, 3)


def test_spin_operators_are_hermitian():
    sx, sy, sz = spin_operators(1)

    for operator in (sx, sy, sz):
        np.testing.assert_allclose(
            operator,
            operator.conj().T,
            atol=1e-12,
        )


def test_invalid_spin_raises_error():
    with pytest.raises(ValueError):
        spin_operators(0.3)

    with pytest.raises(ValueError):
        spin_operators(-0.5)