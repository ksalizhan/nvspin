import numpy as np
import pytest

from nvspin import NVCenter
from nvspin.experiments.energy_levels import (
    EnergyLevels,
)


def test_pure_nv_energy_level_shape():
    nv = NVCenter(
        nitrogen=None,
    )

    experiment = EnergyLevels(
        nv,
        B_min=0.0,
        B_max=0.010,
        points=101,
    )

    result = experiment.simulate()

    assert result.magnetic_fields.shape == (101,)
    assert result.energies.shape == (101, 3)


def test_n14_energy_level_shape():
    nv = NVCenter()

    experiment = EnergyLevels(
        nv,
        points=51,
    )

    result = experiment.simulate()

    assert result.energies.shape == (51, 9)


def test_energy_levels_with_carbon():
    nv = NVCenter()

    nv.add_carbon13(
        A_parallel=50e-6,
        A_perp=30e-6,
    )

    result = EnergyLevels(
        nv,
        points=21,
    ).simulate()

    assert result.energies.shape == (21, 18)


def test_aligned_pure_nv_final_field():
    D = 2.870
    gamma_e = 28.025
    B_max = 0.010

    nv = NVCenter(
        D=D,
        E=0.0,
        nitrogen=None,
    )

    result = EnergyLevels(
        nv,
        B_min=0.0,
        B_max=B_max,
        points=11,
        direction=(0, 0, 1),
    ).simulate()

    final_energies = result.energies[-1]

    expected = np.array(
        [
            0.0,
            D - gamma_e * B_max,
            D + gamma_e * B_max,
        ]
    )

    np.testing.assert_allclose(
        final_energies,
        expected,
        atol=1e-12,
    )


def test_field_direction_is_normalized():
    nv = NVCenter(
        nitrogen=None,
    )

    experiment_a = EnergyLevels(
        nv,
        B_max=0.010,
        points=10,
        direction=(0, 0, 1),
    )

    experiment_b = EnergyLevels(
        nv,
        B_max=0.010,
        points=10,
        direction=(0, 0, 10),
    )

    result_a = experiment_a.simulate()
    result_b = experiment_b.simulate()

    np.testing.assert_allclose(
        result_a.energies,
        result_b.energies,
        atol=1e-12,
    )


def test_simulation_restores_original_field():
    original_B = np.array(
        [0.001, 0.002, 0.003]
    )

    nv = NVCenter(
        B=original_B.copy(),
        nitrogen=None,
    )

    EnergyLevels(
        nv,
        B_max=0.010,
        points=10,
    ).simulate()

    np.testing.assert_allclose(
        nv.B,
        original_B,
    )


def test_transition_energies_reference_ground_state():
    nv = NVCenter(
        nitrogen=None,
    )

    result = EnergyLevels(
        nv,
        points=10,
    ).simulate()

    transitions = result.transition_energies

    np.testing.assert_allclose(
        transitions[:, 0],
        0.0,
        atol=1e-12,
    )


def test_invalid_points():
    nv = NVCenter()

    with pytest.raises(ValueError):
        EnergyLevels(
            nv,
            points=1,
        )


def test_invalid_field_range():
    nv = NVCenter()

    with pytest.raises(ValueError):
        EnergyLevels(
            nv,
            B_min=0.010,
            B_max=0.001,
        )


def test_zero_direction_rejected():
    nv = NVCenter()

    with pytest.raises(ValueError):
        EnergyLevels(
            nv,
            direction=(0, 0, 0),
        )