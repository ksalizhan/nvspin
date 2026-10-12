import numpy as np

from nvspin import NVCenter
from nvspin.physics.hamiltonian import (
    build_hamiltonian,
    eigensystem,
)


def test_pure_nv_hamiltonian_dimension():
    nv = NVCenter(
        nitrogen=None,
    )

    H = build_hamiltonian(nv)

    assert H.shape == (3, 3)


def test_nv_n14_dimension():
    nv = NVCenter()

    H = build_hamiltonian(nv)

    assert H.shape == (9, 9)


def test_nv_n15_dimension():
    nv = NVCenter()

    nv.set_nitrogen("15N")

    H = build_hamiltonian(nv)

    assert H.shape == (6, 6)


def test_dimension_with_one_carbon():
    nv = NVCenter()

    nv.add_carbon13()

    H = build_hamiltonian(nv)

    assert H.shape == (18, 18)


def test_dimension_with_two_carbons():
    nv = NVCenter()

    nv.add_carbon13()
    nv.add_carbon13()

    H = build_hamiltonian(nv)

    assert H.shape == (36, 36)


def test_hamiltonian_is_hermitian():
    nv = NVCenter(
        E=0.002,
        B=[0.003, 0.002, 0.010],
    )

    nv.add_carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    H = build_hamiltonian(nv)

    np.testing.assert_allclose(
        H,
        H.conj().T,
        atol=1e-12,
    )


def test_pure_nv_zero_field():
    nv = NVCenter(
        D=2.870,
        E=0.0,
        B=[0.0, 0.0, 0.0],
        nitrogen=None,
    )

    energies, _ = eigensystem(nv)

    np.testing.assert_allclose(
        energies,
        [0.0, 2.870, 2.870],
        atol=1e-12,
    )


def test_aligned_field_transition_frequencies():
    D = 2.870
    gamma_e = 28.025
    Bz = 0.010

    nv = NVCenter(
        D=D,
        E=0.0,
        B=[0.0, 0.0, Bz],
        nitrogen=None,
    )

    energies, _ = eigensystem(nv)

    expected = np.array(
        [
            0.0,
            D - gamma_e * Bz,
            D + gamma_e * Bz,
        ]
    )

    np.testing.assert_allclose(
        energies,
        expected,
        atol=1e-12,
    )


def test_transverse_strain_zero_field():
    D = 2.870
    E = 0.003

    nv = NVCenter(
        D=D,
        E=E,
        B=[0.0, 0.0, 0.0],
        nitrogen=None,
    )

    energies, _ = eigensystem(nv)

    expected = np.array(
        [
            0.0,
            D - E,
            D + E,
        ]
    )

    np.testing.assert_allclose(
        energies,
        expected,
        atol=1e-12,
    )


def test_eigensystem_dimensions():
    nv = NVCenter()

    nv.add_carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    energies, states = eigensystem(nv)

    assert energies.shape == (18,)
    assert states.shape == (18, 18)