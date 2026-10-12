"""
Ideal Hahn-echo demonstration.

Compares Ramsey phase accumulation with Hahn refocusing
for a static frequency detuning.

The ideal Hahn reference is calculated using instantaneous
pulse operators so that the refocusing mechanism is shown
without finite-pulse errors.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():

    # ========================================================
    # Parameters
    # ========================================================

    detuning = 0.005      # GHz = 5 MHz

    tau = np.linspace(
        0.0,
        2000.0,
        2000,
    )

    total_time = (
        2.0 * tau
    )

    # ========================================================
    # Pauli operators
    # ========================================================

    sigma_x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=complex,
    )

    sigma_z = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=complex,
    )

    identity = np.eye(
        2,
        dtype=complex,
    )

    # ========================================================
    # Ideal instantaneous pulses
    # ========================================================

    U_pi_over_2 = (
        identity
        - 1j * sigma_x
    ) / np.sqrt(2.0)

    U_pi = (
        -1j * sigma_x
    )

    initial_state = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=complex,
    )

    H_free = (
        0.5
        * detuning
        * sigma_z
    )

    # ========================================================
    # Ramsey and Hahn signals
    # ========================================================

    ramsey_signal = np.zeros_like(
        tau
    )

    hahn_signal = np.zeros_like(
        tau
    )

    for index, delay in enumerate(
        tau
    ):

        phases = np.exp(
            -1j
            * 2.0
            * np.pi
            * np.diag(H_free)
            * delay
        )

        U_free = np.diag(
            phases
        )

        # ----------------------------------------------------
        # Ramsey:
        #
        # pi/2 - 2*tau - pi/2
        # ----------------------------------------------------

        phases_ramsey = np.exp(
            -1j
            * 2.0
            * np.pi
            * np.diag(H_free)
            * (2.0 * delay)
        )

        U_free_ramsey = np.diag(
            phases_ramsey
        )

        state_ramsey = (
            U_pi_over_2
            @ U_free_ramsey
            @ U_pi_over_2
            @ initial_state
        )

        ramsey_signal[index] = (
            np.abs(
                state_ramsey[0]
            ) ** 2
        )

        # ----------------------------------------------------
        # Hahn:
        #
        # pi/2 - tau - pi - tau - pi/2
        # ----------------------------------------------------

        state_hahn = (
            U_pi_over_2
            @ U_free
            @ U_pi
            @ U_free
            @ U_pi_over_2
            @ initial_state
        )

        hahn_signal[index] = (
            np.abs(
                state_hahn[0]
            ) ** 2
        )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        total_time,
        ramsey_signal,
        color="gray",
        linewidth=1.1,
        label="Ramsey",
    )

    plt.plot(
        total_time,
        hahn_signal,
        color="black",
        linewidth=1.8,
        label="Hahn echo",
    )

    plt.xlabel(
        r"Total free-evolution time $2\tau$ (ns)"
    )

    plt.ylabel(
        r"$P_0$"
    )

    plt.title(
        "Static-Detuning Refocusing by Hahn Echo\n"
        r"$\Delta=5$ MHz"
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


if __name__ == "__main__":
    main()