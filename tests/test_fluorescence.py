import numpy as np

from nvspin.readout import (
    FluorescenceReadout,
    NVOpticalCycle,
    NVOpticalRates,
)


def make_readout(
    background_rate=0.0,
):

    rates = NVOpticalRates(
        excitation=0.05,

        radiative_ms0=0.077,
        radiative_ms1=0.077,

        isc_ms0=0.005,
        isc_ms1=0.050,

        singlet_to_ms0=0.004,
        singlet_to_ms1=0.001,
    )

    cycle = NVOpticalCycle(
        rates
    )

    return FluorescenceReadout(
        optical_cycle=cycle,
        collection_efficiency=1.0,
        background_rate=background_rate,
    )


def test_bright_state_produces_more_photons():

    readout = make_readout()

    # Start readout in ground ms=0.
    bright_initial = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # Start readout in ground ms=+/-1.
    dark_initial = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    bright = readout.integrated_photons(
        bright_initial,
        readout_time=300.0,
    )

    dark = readout.integrated_photons(
        dark_initial,
        readout_time=300.0,
    )

    assert bright > dark


def test_contrast_is_positive():

    readout = make_readout()

    bright_initial = np.array(
        [1, 0, 0, 0, 0],
        dtype=float,
    )

    dark_initial = np.array(
        [0, 1, 0, 0, 0],
        dtype=float,
    )

    bright = readout.integrated_photons(
        bright_initial,
        readout_time=300.0,
    )

    dark = readout.integrated_photons(
        dark_initial,
        readout_time=300.0,
    )

    contrast = readout.contrast(
        bright,
        dark,
    )

    assert contrast > 0.0
    assert contrast < 1.0


def test_background_reduces_contrast():

    no_background = make_readout(
        background_rate=0.0
    )

    with_background = make_readout(
        background_rate=0.02
    )

    bright_initial = np.array(
        [1, 0, 0, 0, 0],
        dtype=float,
    )

    dark_initial = np.array(
        [0, 1, 0, 0, 0],
        dtype=float,
    )

    bright_0 = (
        no_background.integrated_photons(
            bright_initial,
            readout_time=300.0,
        )
    )

    dark_0 = (
        no_background.integrated_photons(
            dark_initial,
            readout_time=300.0,
        )
    )

    bright_bg = (
        with_background.integrated_photons(
            bright_initial,
            readout_time=300.0,
        )
    )

    dark_bg = (
        with_background.integrated_photons(
            dark_initial,
            readout_time=300.0,
        )
    )

    contrast_0 = (
        no_background.contrast(
            bright_0,
            dark_0,
        )
    )

    contrast_bg = (
        with_background.contrast(
            bright_bg,
            dark_bg,
        )
    )

    assert contrast_bg < contrast_0