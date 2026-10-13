import pytest

from nvspin.control import (
    cpmg,
    pdd,
    xy8,
)


def test_pdd_axes():

    sequence = pdd(
        n_pulses=4
    )

    assert sequence.axes == (
        "X",
        "X",
        "X",
        "X",
    )


def test_cpmg_axes():

    sequence = cpmg(
        n_pulses=4
    )

    assert sequence.axes == (
        "Y",
        "Y",
        "Y",
        "Y",
    )


def test_xy8_axes():

    sequence = xy8(
        blocks=1
    )

    assert sequence.axes == (
        "X",
        "Y",
        "X",
        "Y",
        "Y",
        "X",
        "Y",
        "X",
    )


def test_xy8_two_blocks():

    sequence = xy8(
        blocks=2
    )

    assert (
        sequence.number_of_pulses
        == 16
    )

    assert sequence.axes[:8] == (
        "X",
        "Y",
        "X",
        "Y",
        "Y",
        "X",
        "Y",
        "X",
    )

    assert sequence.axes[8:] == (
        "X",
        "Y",
        "X",
        "Y",
        "Y",
        "X",
        "Y",
        "X",
    )


def test_invalid_pdd():

    with pytest.raises(
        ValueError
    ):
        pdd(0)


def test_invalid_cpmg():

    with pytest.raises(
        ValueError
    ):
        cpmg(0)


def test_invalid_xy8():

    with pytest.raises(
        ValueError
    ):
        xy8(0)