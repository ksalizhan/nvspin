import numpy as np

from nvspin import (
    Carbon13,
    Nitrogen14,
    Nitrogen15,
)


def test_carbon13_properties():
    carbon = Carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    assert carbon.spin == 0.5
    assert carbon.dimension == 2
    assert carbon.quadrupole == 0.0

    np.testing.assert_allclose(
        carbon.A,
        np.diag(
            [
                30e-6,
                30e-6,
                50e-6,
            ]
        ),
    )


def test_nitrogen14_properties():
    nitrogen = Nitrogen14()

    assert nitrogen.spin == 1.0
    assert nitrogen.dimension == 3
    assert nitrogen.quadrupole != 0.0


def test_nitrogen15_properties():
    nitrogen = Nitrogen15()

    assert nitrogen.spin == 0.5
    assert nitrogen.dimension == 2
    assert nitrogen.quadrupole == 0.0


def test_update_axial_hyperfine():
    carbon = Carbon13()

    carbon.set_axial_hyperfine(
        A_parallel=100e-6,
        A_perp=40e-6,
    )

    assert np.isclose(
        carbon.A_parallel,
        100e-6,
    )

    assert np.isclose(
        carbon.A_perp,
        40e-6,
    )