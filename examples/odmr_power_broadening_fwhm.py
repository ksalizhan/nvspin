"""
Quantitative microwave power-broadening test for CW ODMR.

This diagnostic uses a pure NV electronic spin:

    nitrogen=None

so that the FWHM of one isolated ODMR transition can be
measured without overlapping 14N hyperfine lines.

Fixed parameters
----------------
B = 5 mT parallel to NV
T2 = 300 ns
k_ex = 0.001 ns^-1

Sweep
-----
Reference Rabi frequency:

    Omega/(2*pi)
        =
        0.05, 0.1, 0.2, 0.3, 0.5,
        0.7, 1, 1.5, 2, 3, 5 MHz

For each spectrum we extract:

    - minimum fluorescence
    - ODMR contrast
    - numerical FWHM

and compare the measured linewidth with the zero/weak-drive
coherence width

    Gamma_0 = gamma_2 / pi

where

    gamma_2 = 1/T2 + k_ex.

The purpose is to verify that microwave power broadening
emerges from the explicit optical-Bloch dynamics.
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

K_EX = 0.001

RABI_FREQUENCIES = np.array(
    [
        0.00005,   # 0.05 MHz
        0.00010,   # 0.10 MHz
        0.00020,   # 0.20 MHz
        0.00030,   # 0.30 MHz
        0.00050,   # 0.50 MHz
        0.00070,   # 0.70 MHz
        0.00100,   # 1.00 MHz
        0.00150,   # 1.50 MHz
        0.00200,   # 2.00 MHz
        0.00300,   # 3.00 MHz
        0.00500,   # 5.00 MHz
    ],
    dtype=float,
)


# ============================================================
# Optical cycle
# ============================================================


def build_optical_cycle():

    rates = NVOpticalRates(
        excitation=K_EX,

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
    Linear interpolation of x at y = target.
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
    Extract the FWHM of one ODMR dip.

    The normalized off-resonant baseline is estimated as the
    maximum fluorescence.

    If

        F_off = baseline
        F_min = dip minimum,

    then the half-depth fluorescence is

        F_half
        =
        F_min
        +
        (F_off - F_min)/2.

    The left and right crossings are found around the global
    minimum and linearly interpolated.

    Returns
    -------
    width
        FWHM in GHz.

    left
        Left half-depth crossing in GHz.

    right
        Right half-depth crossing in GHz.

    half_level
        Half-depth fluorescence level.
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
    # Search left from the minimum
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
            y_inside
            <= half_level
            and
            y_outside
            >= half_level
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
    # Search right from the minimum
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
            y_inside
            <= half_level
            and
            y_outside
            >= half_level
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
    #
    # No 14N here deliberately.
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

    optical_cycle = (
        build_optical_cycle()
    )

    # Lower electronic resonance.
    expected_resonance = (
        2.870
        - 28.025 * 0.005
    )

    print()
    print(
        "ODMR power-broadening FWHM diagnostic"
    )
    print(
        "===================================="
    )

    print(
        f"Expected lower resonance = "
        f"{expected_resonance:.9f} GHz"
    )

    print(
        f"T2 = {T2:.3f} ns"
    )

    print(
        f"k_ex = {K_EX:.6f} ns^-1"
    )

    print()

    results = []

    # ========================================================
    # Rabi-frequency sweep
    # ========================================================

    for rabi_frequency in RABI_FREQUENCIES:

        experiment = ODMR(
            nv,

            optical_cycle=optical_cycle,

            rabi_frequency=float(
                rabi_frequency
            ),

            T2=T2,

            # Wide enough for the strongly broadened cases,
            # while still containing only the lower electronic
            # resonance.
            frequency_range=(
                expected_resonance
                - 0.030,

                expected_resonance
                + 0.030,
            ),

            points=12001,

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
            fwhm,
            left,
            right,
            half_level,
        ) = extract_fwhm(
            result.frequencies,
            result.fluorescence,
        )

        minimum = float(
            np.min(
                result.fluorescence
            )
        )

        minimum_frequency = float(
            result.frequencies[
                np.argmin(
                    result.fluorescence
                )
            ]
        )

        contrast = (
            1.0
            - minimum
        )

        results.append(
            {
                "rabi_frequency":
                    rabi_frequency,

                "experiment":
                    experiment,

                "result":
                    result,

                "fwhm":
                    fwhm,

                "left":
                    left,

                "right":
                    right,

                "half_level":
                    half_level,

                "contrast":
                    contrast,

                "minimum_frequency":
                    minimum_frequency,
            }
        )

        print(
            f"Omega/2pi = "
            f"{1000.0 * rabi_frequency:7.3f} MHz"
            f"    "
            f"FWHM = "
            f"{1000.0 * fwhm:8.4f} MHz"
            f"    "
            f"contrast = "
            f"{100.0 * contrast:8.4f} %"
            f"    "
            f"f_min = "
            f"{minimum_frequency:.9f} GHz"
        )

    # ========================================================
    # Baseline weak-drive width
    # ========================================================

    gamma_2 = (
        1.0 / T2
        + K_EX
    )

    gamma_zero = (
        gamma_2
        / np.pi
    )

    print()

    print(
        "Zero/weak-drive coherence width"
    )
    print(
        "==============================="
    )

    print(
        f"gamma_2 = "
        f"{gamma_2:.9f} ns^-1"
    )

    print(
        f"Gamma_0 = gamma_2/pi = "
        f"{1000.0 * gamma_zero:.6f} MHz"
    )

    # ========================================================
    # Figure 1:
    # Representative spectra
    # ========================================================

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    selected = {
        0.00005,
        0.00030,
        0.00100,
        0.00200,
        0.00500,
    }

    for entry in results:

        rabi_frequency = (
            entry[
                "rabi_frequency"
            ]
        )

        if not any(
            np.isclose(
                rabi_frequency,
                value,
            )
            for value in selected
        ):
            continue

        result = (
            entry[
                "result"
            ]
        )

        plt.plot(
            (
                result.frequencies
                - expected_resonance
            )
            * 1000.0,
            result.fluorescence,
            linewidth=1.4,
            label=(
                rf"$\Omega/2\pi="
                rf"{1000.0 * rabi_frequency:g}$ MHz"
            ),
        )

    plt.xlabel(
        "Detuning from resonance (MHz)"
    )

    plt.ylabel(
        "Normalized fluorescence"
    )

    plt.title(
        "CW-ODMR Microwave Power Broadening"
        "\n"
        r"Pure NV, "
        r"$T_2=300$ ns, "
        r"$k_{\rm ex}=0.001$ ns$^{-1}$"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 2:
    # FWHM versus Rabi frequency
    # ========================================================

    rabi_mhz = np.asarray(
        [
            1000.0
            * entry[
                "rabi_frequency"
            ]
            for entry in results
        ]
    )

    fwhm_mhz = np.asarray(
        [
            1000.0
            * entry[
                "fwhm"
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
        rabi_mhz,
        fwhm_mhz,
        marker="o",
        linewidth=1.5,
        label="Simulated FWHM",
    )

    plt.axhline(
        1000.0
        * gamma_zero,
        linestyle="--",
        linewidth=1.2,
        label=(
            r"$\Gamma_0="
            r"(1/T_2+k_{\rm ex})/\pi$"
        ),
    )

    plt.xlabel(
        r"Reference Rabi frequency "
        r"$\Omega/2\pi$ (MHz)"
    )

    plt.ylabel(
        "ODMR FWHM (MHz)"
    )

    plt.title(
        "Emergent Microwave Power Broadening"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 3:
    # Squared linewidth
    #
    # Bloch-type power broadening often gives
    #
    #     Gamma^2
    #         ~
    #     Gamma_0^2
    #     + const * Omega^2.
    #
    # Therefore this is a useful diagnostic without assuming
    # the coefficient beforehand.
    # ========================================================

    omega_squared = (
        rabi_mhz**2
    )

    linewidth_squared = (
        fwhm_mhz**2
    )

    fit = np.polyfit(
        omega_squared,
        linewidth_squared,
        1,
    )

    fitted_linewidth_squared = (
        fit[0]
        * omega_squared
        + fit[1]
    )

    print()

    print(
        "Fit of Gamma^2 versus (Omega/2pi)^2"
    )
    print(
        "===================================="
    )

    print(
        f"slope     = "
        f"{fit[0]:.6f}"
    )

    print(
        f"intercept = "
        f"{fit[1]:.6f} MHz^2"
    )

    if fit[1] > 0.0:

        print(
            f"sqrt(intercept) = "
            f"{np.sqrt(fit[1]):.6f} MHz"
        )

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        omega_squared,
        linewidth_squared,
        marker="o",
        linewidth=0.0,
        label="Simulation",
    )

    plt.plot(
        omega_squared,
        fitted_linewidth_squared,
        linewidth=1.5,
        label="Linear fit",
    )

    plt.xlabel(
        r"$(\Omega/2\pi)^2$ (MHz$^2$)"
    )

    plt.ylabel(
        r"$\Gamma_{\rm FWHM}^2$ (MHz$^2$)"
    )

    plt.title(
        "Bloch Power-Broadening Diagnostic"
    )

    plt.legend()

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    # ========================================================
    # Figure 4:
    # Contrast versus microwave drive
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
        rabi_mhz,
        contrast_percent,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        r"Reference Rabi frequency "
        r"$\Omega/2\pi$ (MHz)"
    )

    plt.ylabel(
        "ODMR contrast (%)"
    )

    plt.title(
        "CW-ODMR Saturation"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()