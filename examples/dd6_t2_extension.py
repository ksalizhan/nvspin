"""
Demonstrate coherence-time extension by dynamical decoupling.

The same Ornstein-Uhlenbeck frequency-noise model is used
for:

    Free induction   N = 0
    Hahn / PDD-1     N = 1
    PDD-2            N = 2
    PDD-4            N = 4
    PDD-8            N = 8
    PDD-16           N = 16

For every sequence we calculate the ensemble transverse
coherence

    L(T) = sqrt(<sigma_x>^2 + <sigma_y>^2)

as a function of total storage time T.

An effective coherence time is estimated from

    L(T2) = exp(-1).

This example demonstrates DD suppression of correlated
classical frequency noise. It is not yet 13C spectroscopy.
"""

import matplotlib.pyplot as plt
import numpy as np

from nvspin.control import (
    free_induction,
    pdd,
)

from nvspin.experiments import (
    DynamicalDecoupling,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


# ============================================================
# Simulation parameters
# ============================================================

SIGMA = 0.0015
CORRELATION_TIME = 1000.0

REALIZATIONS = 100
DT = 10.0

SEED = 12345

TIME_MIN = 100.0
TIME_MAX = 12000.0
TIME_POINTS = 45


# ============================================================
# Effective T2 extraction
# ============================================================

def estimate_t2(
    times,
    coherence,
):
    """
    Estimate T2 from

        L(T2) = exp(-1)

    using linear interpolation between neighboring samples.

    Returns None if the simulated time range never reaches
    the threshold.
    """

    threshold = np.exp(
        -1.0
    )

    for index in range(
        1,
        len(times),
    ):

        previous_value = (
            coherence[index - 1]
        )

        current_value = (
            coherence[index]
        )

        if (
            previous_value >= threshold
            and current_value <= threshold
        ):

            t1 = times[
                index - 1
            ]

            t2 = times[
                index
            ]

            c1 = previous_value
            c2 = current_value

            if np.isclose(
                c1,
                c2,
            ):
                return float(t2)

            fraction = (
                threshold - c1
            ) / (
                c2 - c1
            )

            return float(
                t1
                +
                fraction
                * (
                    t2 - t1
                )
            )

    return None


# ============================================================
# Coherence curve
# ============================================================

def simulate_curve(
    sequence,
    times,
    noise,
):
    """
    Calculate L(T) for one pulse sequence.
    """

    coherence = np.zeros_like(
        times,
        dtype=float,
    )

    for index, total_time in enumerate(
        times
    ):

        experiment = (
            DynamicalDecoupling(
                sequence=sequence,
                noise=noise,
                total_time=float(
                    total_time
                ),
                realizations=REALIZATIONS,
                dt=DT,
                seed=SEED,
            )
        )

        result = (
            experiment.simulate()
        )

        coherence[index] = (
            result.coherence
        )

        print(
            f"{sequence.name:15s} "
            f"T = {total_time:8.1f} ns   "
            f"L = {result.coherence:.5f}"
        )

    return coherence


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Same correlated-noise model for every sequence
    # --------------------------------------------------------

    noise = OrnsteinUhlenbeckNoise(
        sigma=SIGMA,
        correlation_time=(
            CORRELATION_TIME
        ),
    )

    # --------------------------------------------------------
    # Total storage-time sweep
    # --------------------------------------------------------

    times = np.linspace(
        TIME_MIN,
        TIME_MAX,
        TIME_POINTS,
    )

    # --------------------------------------------------------
    # Pulse-number comparison
    # --------------------------------------------------------

    sequences = [
        free_induction(),
        pdd(1),
        pdd(2),
        pdd(4),
        pdd(8),
        pdd(16),
    ]

    results = {}

    print()
    print("DD coherence-extension simulation")
    print("---------------------------------")

    print(
        f"Noise sigma = "
        f"{SIGMA * 1000:.3f} MHz"
    )

    print(
        f"Correlation time = "
        f"{CORRELATION_TIME:.1f} ns"
    )

    print(
        f"Realizations = "
        f"{REALIZATIONS}"
    )

    print()

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------

    for sequence in sequences:

        print()
        print(
            f"Simulating {sequence.name}"
        )

        print(
            "-" * (
                11
                + len(
                    sequence.name
                )
            )
        )

        coherence = simulate_curve(
            sequence,
            times,
            noise,
        )

        t2 = estimate_t2(
            times,
            coherence,
        )

        results[
            sequence.name
        ] = {
            "coherence": coherence,
            "T2": t2,
            "pulses": (
                sequence.number_of_pulses
            ),
        }

    # ========================================================
    # Print extracted coherence times
    # ========================================================

    print()
    print()
    print("Effective coherence times")
    print("-------------------------")

    for sequence in sequences:

        result = results[
            sequence.name
        ]

        t2 = result[
            "T2"
        ]

        if t2 is None:

            print(
                f"{sequence.name:15s} "
                f"T2 > {TIME_MAX:.1f} ns"
            )

        else:

            print(
                f"{sequence.name:15s} "
                f"T2 = {t2:.1f} ns"
            )

    # ========================================================
    # Plot coherence curves
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    for sequence in sequences:

        coherence = results[
            sequence.name
        ][
            "coherence"
        ]

        plt.plot(
            times / 1000.0,
            coherence,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=sequence.name,
        )

    # 1/e threshold
    plt.axhline(
        np.exp(-1.0),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=r"$1/e$",
    )

    plt.xlabel(
        r"Total evolution time $T$ ($\mu$s)"
    )

    plt.ylabel(
        r"Transverse coherence $L(T)$"
    )

    plt.title(
        "Dynamical-Decoupling Coherence Extension\n"
        "Ornstein-Uhlenbeck Frequency Noise"
    )

    plt.ylim(
        -0.03,
        1.03,
    )

    plt.grid(
        alpha=0.20
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Plot extracted T2 versus pulse number
    # ========================================================

    pulse_numbers = []
    t2_values = []

    for sequence in sequences:

        result = results[
            sequence.name
        ]

        if result["T2"] is not None:

            pulse_numbers.append(
                result["pulses"]
            )

            t2_values.append(
                result["T2"]
                / 1000.0
            )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        pulse_numbers,
        t2_values,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        "Number of refocusing pulses"
    )

    plt.ylabel(
        r"Effective $T_2$ ($\mu$s)"
    )

    plt.title(
        r"Coherence-Time Extension with DD"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()