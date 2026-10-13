"""
Study effective coherence time versus DD pulse number.

Calculates:

    T2(N)
    T2(N) / T2(0)
    marginal improvement with pulse number

under the same Ornstein-Uhlenbeck frequency-noise model.

Important
---------
This simulation currently assumes ideal instantaneous pulses.

Therefore an intrinsic optimum pulse number is not guaranteed
to exist. The script reports the best pulse number within the
scanned range and shows whether the improvement is beginning
to saturate.

The effective coherence time is defined operationally by

    L(T2) = exp(-1).
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
# Noise model
# ============================================================

SIGMA = 0.0015               # GHz = 1.5 MHz
CORRELATION_TIME = 1000.0    # ns = 1 us


# ============================================================
# Monte Carlo parameters
# ============================================================

REALIZATIONS = 150
DT = 10.0
SEED = 12345


# ============================================================
# Pulse numbers
# ============================================================

PULSE_NUMBERS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
]


# ============================================================
# Time sweep
# ============================================================

TIME_MIN = 50.0
TIME_MAX = 25000.0

TIME_POINTS = 60

THRESHOLD = np.exp(-1.0)


# ============================================================
# Sequence construction
# ============================================================

def sequence_for_pulse_number(
    number_of_pulses,
):
    """Return free induction for N=0 and PDD-N otherwise."""

    if number_of_pulses == 0:
        return free_induction()

    return pdd(
        number_of_pulses
    )


# ============================================================
# T2 extraction
# ============================================================

def estimate_t2(
    times,
    coherence,
):
    """
    Estimate the first crossing of

        L(T) = exp(-1)

    using linear interpolation.

    Returns None if the threshold is not reached.
    """

    for index in range(
        1,
        len(times),
    ):

        c_previous = (
            coherence[index - 1]
        )

        c_current = (
            coherence[index]
        )

        if (
            c_previous >= THRESHOLD
            and c_current <= THRESHOLD
        ):

            t_previous = (
                times[index - 1]
            )

            t_current = (
                times[index]
            )

            if np.isclose(
                c_previous,
                c_current,
            ):
                return float(
                    t_current
                )

            fraction = (
                THRESHOLD
                - c_previous
            ) / (
                c_current
                - c_previous
            )

            return float(
                t_previous
                +
                fraction
                * (
                    t_current
                    - t_previous
                )
            )

    return None


# ============================================================
# Simulate one coherence curve
# ============================================================

def simulate_coherence_curve(
    number_of_pulses,
    times,
    noise,
):
    """Calculate L(T) for one pulse number."""

    sequence = (
        sequence_for_pulse_number(
            number_of_pulses
        )
    )

    coherence = np.zeros(
        len(times),
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

    return coherence


# ============================================================
# Main
# ============================================================

def main():

    noise = OrnsteinUhlenbeckNoise(
        sigma=SIGMA,
        correlation_time=(
            CORRELATION_TIME
        ),
    )

    # --------------------------------------------------------
    # Include the exact T=0 condition explicitly.
    # --------------------------------------------------------

    simulated_times = np.linspace(
        TIME_MIN,
        TIME_MAX,
        TIME_POINTS,
    )

    plot_times = np.concatenate(
        (
            [0.0],
            simulated_times,
        )
    )

    results = {}

    print()
    print("DD pulse-number study")
    print("---------------------")

    print(
        f"Noise sigma: "
        f"{SIGMA * 1000:.3f} MHz"
    )

    print(
        f"Noise correlation time: "
        f"{CORRELATION_TIME / 1000:.3f} us"
    )

    print(
        f"Monte Carlo realizations: "
        f"{REALIZATIONS}"
    )

    print()

    # --------------------------------------------------------
    # Simulate each pulse number
    # --------------------------------------------------------

    for number_of_pulses in (
        PULSE_NUMBERS
    ):

        print(
            f"Simulating N = "
            f"{number_of_pulses}"
        )

        coherence = (
            simulate_coherence_curve(
                number_of_pulses,
                simulated_times,
                noise,
            )
        )

        # Exact initial condition
        plot_coherence = np.concatenate(
            (
                [1.0],
                coherence,
            )
        )

        t2 = estimate_t2(
            plot_times,
            plot_coherence,
        )

        results[
            number_of_pulses
        ] = {
            "coherence": (
                plot_coherence
            ),
            "T2": t2,
        }

        if t2 is None:

            print(
                f"    T2 > "
                f"{TIME_MAX / 1000:.2f} us"
            )

        else:

            print(
                f"    T2 = "
                f"{t2 / 1000:.3f} us"
            )

    # ========================================================
    # Extract valid T2 values
    # ========================================================

    valid_pulse_numbers = []
    valid_t2 = []

    for number_of_pulses in (
        PULSE_NUMBERS
    ):

        t2 = results[
            number_of_pulses
        ][
            "T2"
        ]

        if t2 is not None:

            valid_pulse_numbers.append(
                number_of_pulses
            )

            valid_t2.append(
                t2
            )

    valid_pulse_numbers = np.asarray(
        valid_pulse_numbers,
        dtype=int,
    )

    valid_t2 = np.asarray(
        valid_t2,
        dtype=float,
    )

    # ========================================================
    # Baseline and improvement factors
    # ========================================================

    free_induction_t2 = (
        results[0]["T2"]
    )

    if free_induction_t2 is None:

        raise RuntimeError(
            "Free-induction T2 was not reached. "
            "Increase TIME_MAX."
        )

    improvement_factor = (
        valid_t2
        / free_induction_t2
    )

    # --------------------------------------------------------
    # Marginal improvement
    #
    # Fractional increase relative to the previous sampled
    # pulse number.
    # --------------------------------------------------------

    marginal_gain = np.full(
        len(valid_t2),
        np.nan,
        dtype=float,
    )

    for index in range(
        1,
        len(valid_t2),
    ):

        marginal_gain[index] = (
            valid_t2[index]
            - valid_t2[index - 1]
        ) / valid_t2[index - 1]

    # ========================================================
    # Report results
    # ========================================================

    print()
    print("Summary")
    print("-------")

    print(
        "N pulses    T2 (us)    "
        "T2/T2_FID    marginal gain"
    )

    for index, (
        number_of_pulses,
        t2,
        factor,
    ) in enumerate(
        zip(
            valid_pulse_numbers,
            valid_t2,
            improvement_factor,
        )
    ):

        if index == 0:

            marginal_text = (
                "---"
            )

        else:

            marginal_text = (
                f"{100.0 * marginal_gain[index]:.1f}%"
            )

        print(
            f"{number_of_pulses:8d}    "
            f"{t2 / 1000:7.3f}    "
            f"{factor:9.3f}    "
            f"{marginal_text:>13s}"
        )

    # ========================================================
    # Best value within scanned range
    # ========================================================

    best_index = np.argmax(
        valid_t2
    )

    best_pulse_number = (
        valid_pulse_numbers[
            best_index
        ]
    )

    best_t2 = (
        valid_t2[
            best_index
        ]
    )

    print()
    print(
        "Best T2 within scanned range:"
    )

    print(
        f"N = {best_pulse_number}, "
        f"T2 = {best_t2 / 1000:.3f} us"
    )

    print()
    print(
        "Note: with ideal instantaneous pulses, "
        "this is not necessarily a true physical optimum."
    )

    # ========================================================
    # Figure 1: coherence curves
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    for number_of_pulses in (
        PULSE_NUMBERS
    ):

        coherence = results[
            number_of_pulses
        ][
            "coherence"
        ]

        if number_of_pulses == 0:

            label = (
                "Free induction (N=0)"
            )

        elif number_of_pulses == 1:

            label = (
                "Hahn echo (N=1)"
            )

        else:

            label = (
                f"PDD-{number_of_pulses}"
            )

        plt.plot(
            plot_times / 1000.0,
            coherence,
            linewidth=1.4,
            marker="o",
            markersize=2.5,
            label=label,
        )

    plt.axhline(
        THRESHOLD,
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
        "DD Coherence versus Pulse Number\n"
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
    # Figure 2: T2 versus pulse number
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        valid_pulse_numbers,
        valid_t2 / 1000.0,
        marker="o",
        linewidth=1.6,
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"Effective coherence time $T_2$ ($\mu$s)"
    )

    plt.title(
        r"$T_2$ Extension versus DD Pulse Number"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 3: improvement factor
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        valid_pulse_numbers,
        improvement_factor,
        marker="o",
        linewidth=1.6,
    )

    plt.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )

    plt.xlabel(
        "Number of refocusing pulses N"
    )

    plt.ylabel(
        r"$T_2(N) / T_2(N=0)$"
    )

    plt.title(
        "DD Coherence-Time Improvement Factor"
    )

    plt.grid(
        alpha=0.20
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Figure 4: marginal gain
    # ========================================================

    if len(
        valid_pulse_numbers
    ) > 1:

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            valid_pulse_numbers[1:],
            100.0
            * marginal_gain[1:],
            marker="o",
            linewidth=1.6,
        )

        plt.axhline(
            0.0,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )

        plt.xlabel(
            "Number of refocusing pulses N"
        )

        plt.ylabel(
            "Marginal T2 gain (%)"
        )

        plt.title(
            "Marginal Benefit of Increasing DD Pulse Number"
        )

        plt.grid(
            alpha=0.20
        )

        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()