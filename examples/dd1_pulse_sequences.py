"""
Pulse-sequence comparison for dynamical decoupling.

Shows:

    PDD-8
    CPMG-8
    XY8-1

using the symmetric timing convention

    pi/2 - tau/2 - pi - tau - pi - ... - tau/2 - pi/2

The central refocusing pulses are supplied by nvspin.control.

This example is a visualization only. It does not yet
simulate spin dynamics.
"""

import matplotlib.pyplot as plt

from nvspin.control import (
    cpmg,
    pdd,
    xy8,
)


def draw_sequence(
    ax,
    sequence,
    tau=1.0,
    preparation_axis="X",
    readout_axis="X",
):
    """
    Draw one DD sequence.

    Refocusing pulse centers occur at

        tau/2,
        3*tau/2,
        5*tau/2,
        ...

    so the first and final free-evolution intervals
    are tau/2.
    """

    n_pulses = (
        sequence.number_of_pulses
    )

    total_free_time = (
        n_pulses * tau
    )

    # --------------------------------------------------------
    # Time positions
    # --------------------------------------------------------

    refocusing_times = [
        (index + 0.5) * tau
        for index in range(n_pulses)
    ]

    preparation_time = (
        -0.35 * tau
    )

    readout_time = (
        total_free_time
        + 0.35 * tau
    )

    # --------------------------------------------------------
    # Free-evolution baseline
    # --------------------------------------------------------

    ax.hlines(
        y=0.0,
        xmin=0.0,
        xmax=total_free_time,
        color="black",
        linewidth=1.2,
    )

    # --------------------------------------------------------
    # Preparation pi/2 pulse
    # --------------------------------------------------------

    ax.vlines(
        preparation_time,
        0.0,
        0.70,
        color="black",
        linewidth=3.0,
    )

    ax.text(
        preparation_time,
        0.78,
        rf"$\pi/2_{{{preparation_axis}}}$",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Refocusing pi pulses
    # --------------------------------------------------------

    for time, pulse in zip(
        refocusing_times,
        sequence.pulses,
    ):

        ax.vlines(
            time,
            0.0,
            1.0,
            color="black",
            linewidth=3.0,
        )

        ax.text(
            time,
            1.08,
            rf"$\pi_{{{pulse.axis}}}$",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Readout pi/2 pulse
    # --------------------------------------------------------

    ax.vlines(
        readout_time,
        0.0,
        0.70,
        color="black",
        linewidth=3.0,
    )

    ax.text(
        readout_time,
        0.78,
        rf"$\pi/2_{{{readout_axis}}}$",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Show first tau/2 interval
    # --------------------------------------------------------

    first_pulse = (
        refocusing_times[0]
    )

    ax.annotate(
        "",
        xy=(
            first_pulse,
            -0.18,
        ),
        xytext=(
            0.0,
            -0.18,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.0,
        ),
    )

    ax.text(
        first_pulse / 2.0,
        -0.28,
        r"$\tau/2$",
        ha="center",
        va="top",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Show one full tau interval
    # --------------------------------------------------------

    if n_pulses >= 2:

        first = (
            refocusing_times[0]
        )

        second = (
            refocusing_times[1]
        )

        ax.annotate(
            "",
            xy=(
                second,
                -0.18,
            ),
            xytext=(
                first,
                -0.18,
            ),
            arrowprops=dict(
                arrowstyle="<->",
                linewidth=1.0,
            ),
        )

        ax.text(
            0.5 * (
                first
                + second
            ),
            -0.28,
            r"$\tau$",
            ha="center",
            va="top",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Final tau/2 interval
    # --------------------------------------------------------

    last_pulse = (
        refocusing_times[-1]
    )

    ax.annotate(
        "",
        xy=(
            total_free_time,
            -0.18,
        ),
        xytext=(
            last_pulse,
            -0.18,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.0,
        ),
    )

    ax.text(
        0.5 * (
            last_pulse
            + total_free_time
        ),
        -0.28,
        r"$\tau/2$",
        ha="center",
        va="top",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlim(
        preparation_time
        - 0.4 * tau,
        readout_time
        + 0.4 * tau,
    )

    ax.set_ylim(
        -0.55,
        1.40,
    )

    ax.set_yticks([])

    ax.set_xticks(
        range(
            0,
            int(total_free_time) + 1,
        )
    )

    ax.set_xlabel(
        r"Time / $\tau$"
    )

    ax.set_title(
        sequence.name,
        fontsize=13,
    )

    for spine in [
        "left",
        "right",
        "top",
    ]:
        ax.spines[
            spine
        ].set_visible(False)


def main():

    # ========================================================
    # Build the three sequences
    # ========================================================

    sequences = [
        (
            pdd(
                n_pulses=8
            ),
            "X",
            "X",
        ),
        (
            cpmg(
                n_pulses=8
            ),
            "X",
            "X",
        ),
        (
            xy8(
                blocks=1
            ),
            "X",
            "X",
        ),
    ]

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
    )

    for ax, (
        sequence,
        preparation_axis,
        readout_axis,
    ) in zip(
        axes,
        sequences,
    ):

        draw_sequence(
            ax,
            sequence,
            tau=1.0,
            preparation_axis=(
                preparation_axis
            ),
            readout_axis=(
                readout_axis
            ),
        )

    fig.suptitle(
        "Dynamical-Decoupling Pulse Sequences",
        fontsize=15,
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()