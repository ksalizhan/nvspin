"""
Larmor-synchronized DD probing of a structured nuclear environment.

Inspired by:

    Abobeih et al.
    Nature Communications 9, 2552 (2018).

Physical environment
--------------------

    7 isolated 13C spins
    +
    6 interacting 13C-13C pseudospin pairs

The nuclear Hamiltonians are unchanged from the previous
structured-environment example.

The difference is the DD timing protocol.

Instead of scanning arbitrary pulse spacings, we select

    tau_center = 2*m / f_L

where m is a positive integer.

This corresponds to the Larmor synchronization condition

    tau_paper = m / f_L

used to suppress interactions with isolated 13C nuclei.

The remaining electron coherence can reveal interacting
nuclear-spin pairs.

Assumptions
-----------

- Ideal instantaneous electron pi pulses.
- Independent resolved subsystems.
- Explicit nuclear-nuclear interactions within each pair.
- No OU bath.
- No T1 relaxation.
- No finite pulse errors.
- No optical readout noise.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.physics import (
    IndependentNuclearEnvironment,
)

# Reuse the validated nuclear parameters.
from dd_spectroscopy10_abobeih_environment import (
    CARBONS,
    PAIRS,
)


# ============================================================
# Parameters
# ============================================================

NUMBER_OF_PULSES = 32

# Nuclear Larmor reference frequency in GHz.
#
# Use the average of the seven measured ms=0 frequencies.

F_LARMOR = np.mean(
    [
        carbon.larmor_frequency
        for carbon in CARBONS
    ]
)

# Larmor synchronization orders.

M_MIN = 1
M_MAX = 65

# Optional timing offset in ns.
#
# Zero corresponds to exact synchronization with the
# chosen reference frequency.

TIMING_OFFSET = 0.0


# ============================================================
# Construct environments
# ============================================================

def build_environments():

    singles = IndependentNuclearEnvironment(
        subsystems=CARBONS
    )

    pairs = IndependentNuclearEnvironment(
        subsystems=PAIRS
    )

    return singles, pairs


# ============================================================
# Larmor synchronization
# ============================================================

def synchronized_spacing(
    m,
    offset=0.0,
):
    """
    Return center-to-center pulse spacing in ns.

        tau_center = 2*m/f_L + offset

    f_L is in GHz.
    """

    return (
        2.0 * m / F_LARMOR
        + offset
    )


# ============================================================
# Simulate one timing point
# ============================================================

def simulate_point(
    singles,
    pairs,
    spacing,
):

    L_singles = singles.coherence(
        number_of_pulses=NUMBER_OF_PULSES,
        pulse_spacing=float(spacing),
    )

    L_pairs = pairs.coherence(
        number_of_pulses=NUMBER_OF_PULSES,
        pulse_spacing=float(spacing),
    )

    L_total = (
        L_singles * L_pairs
    )

    return (
        L_singles,
        L_pairs,
        L_total,
    )


# ============================================================
# Main
# ============================================================

def main():

    singles, pairs = build_environments()

    m_values = np.arange(
        M_MIN,
        M_MAX + 1,
        dtype=int,
    )

    pulse_spacings = np.array(
        [
            synchronized_spacing(
                m,
                TIMING_OFFSET,
            )
            for m in m_values
        ]
    )

    L_singles = np.zeros(
        len(m_values),
        dtype=complex,
    )

    L_pairs = np.zeros(
        len(m_values),
        dtype=complex,
    )

    L_total = np.zeros(
        len(m_values),
        dtype=complex,
    )

    # ========================================================
    # Simulate synchronized DD
    # ========================================================

    for index, spacing in enumerate(
        pulse_spacings
    ):

        (
            L_singles[index],
            L_pairs[index],
            L_total[index],
        ) = simulate_point(
            singles,
            pairs,
            spacing,
        )

    # ========================================================
    # Convert to electron preservation probabilities
    # ========================================================

    Px_singles = (
        0.5 * (1.0 + np.real(L_singles))
    )

    Px_pairs = (
        0.5 * (1.0 + np.real(L_pairs))
    )

    Px_total = (
        0.5 * (1.0 + np.real(L_total))
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print(
        "Larmor-synchronized DD spectroscopy"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Reference Larmor frequency: "
        f"{F_LARMOR * 1000:.6f} MHz"
    )

    print(
        f"Number of pulses: "
        f"{NUMBER_OF_PULSES}"
    )

    print(
        f"Larmor period: "
        f"{1.0 / F_LARMOR / 1000:.6f} us"
    )

    print()
    print(
        "m       spacing (us)       "
        "Px singles       Px pairs       Px total"
    )

    for index, m in enumerate(
        m_values
    ):

        print(
            f"{m:3d}      "
            f"{pulse_spacings[index] / 1000:10.4f}       "
            f"{Px_singles[index]:10.5f}       "
            f"{Px_pairs[index]:10.5f}       "
            f"{Px_total[index]:10.5f}"
        )

    # ========================================================
    # Figure 1
    #
    # Larmor-synchronized responses
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        pulse_spacings / 1000.0,
        Px_singles,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Seven isolated 13C spins",
    )

    plt.plot(
        pulse_spacings / 1000.0,
        Px_pairs,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Six interacting nuclear pairs",
    )

    plt.plot(
        pulse_spacings / 1000.0,
        Px_total,
        marker="o",
        markersize=3,
        color="black",
        linewidth=1.5,
        label="Combined environment",
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        "Larmor-Synchronized DD Probing"
        "\n"
        rf"$N={NUMBER_OF_PULSES}$"
    )

    plt.ylim(
        -0.03,
        1.03,
    )

    plt.grid(
        alpha=0.2
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 2
    #
    # Pair-sensitive signal
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        pulse_spacings / 1000.0,
        Px_pairs,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Interacting pairs only",
    )

    plt.plot(
        pulse_spacings / 1000.0,
        Px_total,
        marker="o",
        markersize=3,
        color="black",
        linewidth=1.3,
        label="Complete environment",
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Electron preservation probability $P_x$"
    )

    plt.title(
        "Pair-Sensitive Nuclear Spectroscopy"
        "\n"
        "Larmor-Synchronized DD"
    )

    plt.ylim(
        -0.03,
        1.03,
    )

    plt.grid(
        alpha=0.2
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 3
    #
    # Residual isolated-carbon response
    # ========================================================

    residual_single_signal = (
        1.0 - Px_singles
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        pulse_spacings / 1000.0,
        residual_single_signal,
        marker="o",
        markersize=3,
        linewidth=1.3,
    )

    plt.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        r"Pulse-center spacing ($\mu$s)"
    )

    plt.ylabel(
        r"Residual isolated-carbon response $1-P_x$"
    )

    plt.title(
        "Residual Coupling to Isolated Nuclear Spins"
        "\n"
        "Under Larmor Synchronization"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()