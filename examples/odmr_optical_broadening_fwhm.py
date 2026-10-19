"""
Quantitative optical-power broadening test for CW ODMR.

This diagnostic uses a pure NV electronic spin:

    nitrogen=None

so that the FWHM of one isolated ODMR transition can be
measured without overlapping 14N hyperfine lines.

Fixed parameters
----------------
B = 5 mT parallel to NV
T2 = 300 ns
Omega/(2*pi) = 0.05 MHz

Sweep
-----
Optical excitation rate:

    k_ex =
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.010,
        0.030,
        0.050 ns^-1

For the current seven-variable ODMR model,

    gamma_2 = 1/T2 + k_ex

so in the weak-microwave limit the predicted homogeneous
linewidth is

    Gamma_pred = gamma_2 / pi.

For each simulated spectrum we extract the numerical FWHM
and compare it directly with this prediction.

Important
---------
k_ex is an optical excitation RATE.

This script does not yet convert experimental laser power
(mW) into k_ex.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin import NVCenter
from nvspin.experiments import ODMR
from nvspin.readout import (
    NVOpticalCycle,
    NVOpticalRates,
)


# ============================================================
# Fixed parameters
# ============================================================

T2 = 300.0

# Ordinary Rabi frequency in GHz.
#
# 0.00005 GHz = 0.05 MHz.
#
# We deliberately keep the microwave drive weak so that
# microwave power broadening is small compared with the
# optical broadening we want to isolate.
RABI_FREQUENCY = 0.00005

EXCITATION_RATES = np.array(
    [
        0.0001,
        0.0003,
        0.0010,
        0.0030,
        0.0100,
        0.0300,
        0.0500,
    ],
    dtype=float,
)


# ============================================================
# Optical cycle
# ============================================================


def build_optical_cycle(
    excitation_rate,
):

    rates = NVOpticalRates(
        excitation=float(
            excitation_rate
        ),

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


# ============================================================
# FWHM extraction
# ============================================================


def interpolate_crossing(
    x1,
    y1,
    x2,
    y2,
    target,
):
    """
    Linearly interpolate the x position at which y=target.
    """

    if np.isclose(
        y2,
        y1,
    ):
        return 0.5 * (
            x1 + x2
        )

    fraction = (
        target - y1
    ) / (
        y2 - y1
    )

    return (
        x1
        + fraction
        * (
            x2 - x1
        )
    )


def extract_fwhm(
    frequencies,
    fluorescence,
):
    """
    Extract the FWHM of one ODMR fluorescence dip.

    Let

        F_off = off-resonant baseline
        F_min = resonant minimum.

    The half-depth level is

        F_half
        =
        F_min
        +
        (F_off - F_min)/2.

    The FWHM is the separation between the two frequencies
    at which the spectrum crosses F_half.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    fluorescence = np.asarray(
        fluorescence,
        dtype=float,
    )

    baseline = float(
        np.max(
            fluorescence
        )
    )

    minimum_index = int(
        np.argmin(
            fluorescence
        )
    )

    minimum = float(
        fluorescence[
            minimum_index
        ]
    )

    half_level = (
        minimum
        + 0.5
        * (
            baseline
            - minimum
        )
    )

    # --------------------------------------------------------
    # Left crossing
    # --------------------------------------------------------

    left_crossing = None

    for index in range(
        minimum_index,
        0,
        -1,
    ):

        y_inside = (
            fluorescence[index]
        )

        y_outside = (
            fluorescence[index - 1]
        )

        if (
            y_inside <= half_level
            and
            y_outside >= half_level
        ):

            left_crossing = (
                interpolate_crossing(
                    frequencies[index],
                    y_inside,
                    frequencies[index - 1],
                    y_outside,
                    half_level,
                )
            )

            break

    # --------------------------------------------------------
    # Right crossing
    # --------------------------------------------------------

    right_crossing = None

    for index in range(
        minimum_index,
        len(
            frequencies
        ) - 1,
    ):

        y_inside = (
            fluorescence[index]
        )

        y_outside = (
            fluorescence[index + 1]
        )

        if (
            y_inside <= half_level
            and
            y_outside >= half_level
        ):

            right_crossing = (
                interpolate_crossing(
                    frequencies[index],
                    y_inside,
                    frequencies[index + 1],
                    y_outside,
                    half_level,
                )
            )

            break

    if (
        left_crossing is None
        or
        right_crossing is None
    ):
        raise RuntimeError(
            "Could not determine FWHM. "
            "Increase the frequency window."
        )

    width = (
        right_crossing
        - left_crossing
    )

    return (
        width,
        left_crossing,
        right_crossing,
        half_level,
    )


# ============================================================
# Main
# ============================================================


def main():

    # ========================================================
    # Pure electronic NV
    # ========================================================

    nv = NVCenter(
        D=2.870,
        E=0.0,
        B=[
            0.0,
            0.0,
            0.005,
        ],
        nitrogen=None,
    )

    # Lower electronic resonance:
    #
    # f0 = D - gamma_e Bz
    expected_resonance = (
        2.870
        - 28.025 * 0.005
    )

    print()
    print(
        "ODMR optical-broadening FWHM diagnostic"
    )
    print(
        "======================================="
    )

    print(
        f"Expected lower resonance = "
        f"{expected_resonance:.9f} GHz"
    )

    print(
        f"T2 = {T2:.3f} ns"
    )

    print(
        f"Omega/2pi = "
        f"{1000.0 * RABI_FREQUENCY:.3f} MHz"
    )

    print()

    results = []

    # ========================================================
    # Optical-excitation sweep
    # ========================================================

    for k_ex in EXCITATION_RATES:

        optical_cycle = (
            build_optical_cycle(
                k_ex
            )
        )

        experiment = ODMR(
            nv,

            optical_cycle=optical_cycle,

            rabi_frequency=(
                RABI_FREQUENCY
            ),

            T2=T2,

            # The strongest optical-excitation case has a
            # predicted linewidth of roughly 17 MHz.
            #
            # +/- 40 MHz therefore gives sufficient baseline
            # on both sides.
            frequency_range=(
                expected_resonance
                - 0.040,

                expected_resonance
                + 0.040,
            ),

            points=16001,

            microwave_direction=(
                1.0,
                0.0,
                0.0,
            ),
        )

        result = (
            experiment.simulate()
        )

        (
            measured_fwhm,
            left,
            right,
            half_level,
        ) = extract_fwhm(
            result.frequencies,
            result.fluorescence,
        )

        gamma_2 = (
            1.0 / T2
            + k_ex
        )

        predicted_fwhm = (
            gamma_2
            / np.pi
        )

        minimum = float(
            np.min(
                result.fluorescence
            )
        )

        contrast = (
            1.0
            - minimum
        )

        minimum_frequency = float(
            result.frequencies[
                np.argmin(
                    result.fluorescence
                )
            ]
        )

        results.append(
            {
                "k_ex":
                    k_ex,

                "experiment":
                    experiment,

                "result":
                    result,

                "measured_fwhm":
                    measured_fwhm,

                "predicted_fwhm":
                    predicted_fwhm,

                "contrast":
                    contrast,

                "minimum_frequency":
                    minimum_frequency,

                "left":
                    left,

                "right":
                    right,

                "half_level":
                    half_level,
            }
        )

        relative_error = (
            measured_fwhm
            - predicted_fwhm
        ) / predicted_fwhm

        print(
            f"k_ex = "
            f"{k_ex:8.5f} ns^-1"
            f"    "
            f"T2_eff = "
            f"{experiment.effective_T2:8.3f} ns"
            f"    "
            f"Gamma_meas = "
            f"{1000.0 * measured_fwhm:8.4f} MHz"
            f"    "
            f"Gamma_pred = "
            f"{1000.0 * predicted_fwhm:8.4f} MHz"
            f"    "
            f"error = "
            f"{100.0 * relative_error:+7.3f} %"
            f"    "
            f"contrast = "
            f"{100.0 * contrast:7.3f} %"
        )

    # ========================================================
    # Figure 1:
    # Representative ODMR spectra
    # ========================================================

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    selected_rates = {
        0.0001,
        0.0010,
        0.0030,
        0.0100,
        0.0500,
    }

    for entry in results:

        k_ex = (
            entry[
                "k_ex"
            ]
        )

        if not any(
            np.isclose(
                k_ex,
                value,
            )
            for value in selected_rates
        ):
            continue

        result = (
            entry[
                "result"
            ]
        )

        detuning_mhz = (
            result.frequencies
            - expected_resonance
        ) * 1000.0

        plt.plot(
            detuning_mhz,
            result.fluorescence,
            linewidth=1.4,
            label=(
                rf"$k_{{ex}}="
                rf"{k_ex:g}$ ns$^{{-1}}$"
            ),
        )

    plt.xlabel(
        "Detuning from resonance (MHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        "CW-ODMR Optical Broadening"
        "\n"
        r"Pure NV, "
        r"$T_2=300$ ns, "
        r"$\Omega/2\pi=0.05$ MHz"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 2:
    # Measured versus predicted FWHM
    # ========================================================

    k_ex_array = np.asarray(
        [
            entry[
                "k_ex"
            ]
            for entry in results
        ]
    )

    measured_mhz = np.asarray(
        [
            1000.0
            * entry[
                "measured_fwhm"
            ]
            for entry in results
        ]
    )

    predicted_mhz = np.asarray(
        [
            1000.0
            * entry[
                "predicted_fwhm"
            ]
            for entry in results
        ]
    )

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        k_ex_array,
        measured_mhz,
        marker="o",
        linewidth=1.5,
        label="Measured from ODMR",
    )

    plt.plot(
        k_ex_array,
        predicted_mhz,
        linestyle="--",
        linewidth=1.5,
        label=(
            r"$(1/T_2+k_{\rm ex})/\pi$"
        ),
    )

    plt.xlabel(
        r"Optical excitation rate "
        r"$k_{\rm ex}$ (ns$^{-1}$)"
    )

    plt.ylabel(
        "ODMR FWHM (MHz)"
    )

    plt.title(
        "Optical-Excitation Broadening"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 3:
    # Measured linewidth versus predicted linewidth
    #
    # Perfect agreement would lie on y=x.
    # ========================================================

    minimum_width = min(
        np.min(
            measured_mhz
        ),
        np.min(
            predicted_mhz
        ),
    )

    maximum_width = max(
        np.max(
            measured_mhz
        ),
        np.max(
            predicted_mhz
        ),
    )

    comparison_axis = np.linspace(
        minimum_width,
        maximum_width,
        200,
    )

    plt.figure(
        figsize=(
            7,
            6,
        )
    )

    plt.plot(
        predicted_mhz,
        measured_mhz,
        marker="o",
        linewidth=1.5,
        label="Simulation",
    )

    plt.plot(
        comparison_axis,
        comparison_axis,
        linestyle="--",
        linewidth=1.2,
        label="Measured = predicted",
    )

    plt.xlabel(
        "Predicted FWHM (MHz)"
    )

    plt.ylabel(
        "Measured FWHM (MHz)"
    )

    plt.title(
        "Optical-Broadening Consistency Check"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 4:
    # ODMR contrast versus optical excitation rate
    # ========================================================

    contrast_percent = np.asarray(
        [
            100.0
            * entry[
                "contrast"
            ]
            for entry in results
        ]
    )

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.semilogx(
        k_ex_array,
        contrast_percent,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        r"Optical excitation rate "
        r"$k_{\rm ex}$ (ns$^{-1}$)"
    )

    plt.ylabel(
        "ODMR contrast (%)"
    )

    plt.title(
        "ODMR Contrast versus Optical Excitation"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()