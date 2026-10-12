"""
Vertically stacked CW-ODMR comparison:

    1. Pure NV-
    2. NV- + 13C
    3. NV- + 14N

All simulations use the same magnetic field, microwave
polarization, linewidth, contrast, and frequency range.

The purpose is to demonstrate how nuclear hyperfine
interactions modify the ODMR spectrum.
"""

import matplotlib.pyplot as plt

from nvspin import NVCenter
from nvspin.experiments import ODMR


# ============================================================
# Common experimental parameters
# ============================================================

B_FIELD = [
    0.0,
    0.0,
    0.005,      # 5 mT, parallel to NV axis
]

FREQUENCY_RANGE = (
    2.68,
    3.06,
)

POINTS = 8000

LINEWIDTH = 0.001       # GHz = 1 MHz
CONTRAST = 0.03


def simulate_odmr(nv):
    """Run ODMR using common experimental parameters."""

    experiment = ODMR(
        nv,
        frequency_range=FREQUENCY_RANGE,
        points=POINTS,
        linewidth=LINEWIDTH,
        contrast=CONTRAST,
        microwave_direction=(
            1.0,
            0.0,
            0.0,
        ),
    )

    return experiment.simulate()


def main():

    # ========================================================
    # 1. PURE NV-
    # ========================================================

    nv_pure = NVCenter(
        D=2.870,
        E=0.0,
        B=B_FIELD,
        nitrogen=None,
    )

    result_pure = simulate_odmr(
        nv_pure
    )

    # ========================================================
    # 2. NV- + 13C
    # ========================================================
    #
    # Illustrative hyperfine coupling:
    #
    # A_parallel = 30 MHz
    # A_perp     = 20 MHz
    #
    # These are intentionally large enough that the
    # hyperfine splitting is clearly visible.
    # ========================================================

    nv_c13 = NVCenter(
        D=2.870,
        E=0.0,
        B=B_FIELD,
        nitrogen=None,
    )

    nv_c13.add_carbon13(
        A_parallel=30e-3,
        A_perp=20e-3,
    )

    result_c13 = simulate_odmr(
        nv_c13
    )

    # ========================================================
    # 3. NV- + 14N
    # ========================================================
    #
    # NVCenter uses intrinsic 14N by default.
    #
    # Default model:
    #
    # A_parallel = -2.14 MHz
    # A_perp     = -2.70 MHz
    # Q          = -4.95 MHz
    # ========================================================

    nv_n14 = NVCenter(
        D=2.870,
        E=0.0,
        B=B_FIELD,
    )

    result_n14 = simulate_odmr(
        nv_n14
    )

    # ========================================================
    # Create vertically stacked figure
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    # ========================================================
    # Pure NV-
    # ========================================================

    axes[0].plot(
        result_pure.frequencies,
        result_pure.fluorescence,
        color="black",
        linewidth=1.4,
    )

    axes[0].set_title(
        r"Pure NV$^-$"
    )

    # ========================================================
    # NV- + 13C
    # ========================================================

    axes[1].plot(
        result_c13.frequencies,
        result_c13.fluorescence,
        color="black",
        linewidth=1.4,
    )

    axes[1].set_title(
        r"NV$^-$ + $^{13}$C"
        r"   ($A_\parallel=30$ MHz, "
        r"$A_\perp=20$ MHz)"
    )

    # ========================================================
    # NV- + 14N
    # ========================================================

    axes[2].plot(
        result_n14.frequencies,
        result_n14.fluorescence,
        color="black",
        linewidth=1.4,
    )

    axes[2].set_title(
        r"NV$^-$ + $^{14}$N"
    )

    # ========================================================
    # Common formatting
    # ========================================================

    for ax in axes:

        ax.set_ylabel(
            "Normalized\nfluorescence"
        )

        ax.grid(
            alpha=0.20
        )

        ax.set_ylim(
            0.965,
            1.002,
        )

    axes[-1].set_xlabel(
        "Microwave frequency (GHz)"
    )

    fig.suptitle(
        r"CW-ODMR Comparison"
        "\n"
        r"$B=5$ mT, $B\parallel$ NV",
        fontsize=14,
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()