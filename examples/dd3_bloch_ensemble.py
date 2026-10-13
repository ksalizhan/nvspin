"""
Bloch-sphere visualization of ensemble dephasing and Hahn refocusing.

This example demonstrates quasi-static inhomogeneous dephasing.

An ensemble of spins experiences different static detunings:

    Delta_i in [-Delta_max, +Delta_max]

After a pi/2_X preparation pulse, the spins begin at the
same point on the Bloch sphere.

During free evolution:

    positive detunings precess one way around z
    negative detunings precess the opposite way

The ensemble-average Bloch vector therefore shrinks.

An ideal pi_X pulse at t = tau reverses the phase ordering,
causing the ensemble to rephase at t = 2*tau.

This is an ensemble visualization of T2*-type dephasing.
It is not irreversible T2 decoherence.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
)

from nvspin.control import (
    bloch_vector,
    rotate_density_matrix,
)

from nvspin.solvers import (
    evolve_density_matrix,
)


# ============================================================
# Parameters
# ============================================================

N_SPINS = 31

DELTA_MAX = 0.005       # GHz = 5 MHz

TAU = 250.0             # ns

FREE_FRAMES = 100
PULSE_FRAMES = 20

FPS = 25


# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = (
    Path(__file__).parent
    / "output"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "hahn_ensemble_dephasing.gif"
)


# ============================================================
# Operators
# ============================================================

SIGMA_Z = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0],
    ],
    dtype=complex,
)


def free_hamiltonian(
    detuning,
):
    """
    Quasi-static free-precession Hamiltonian.

        H = Delta/2 sigma_z
    """

    return (
        0.5
        * detuning
        * SIGMA_Z
    )


# ============================================================
# Ensemble initialization
# ============================================================

def initial_ensemble():
    """
    Create an ensemble initially prepared in |0>.
    """

    rho0 = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    return [
        rho0.copy()
        for _ in range(N_SPINS)
    ]


def prepare_ensemble(
    ensemble,
):
    """
    Apply an ideal pi/2_X preparation pulse.

    |0> is rotated onto the equator.
    """

    return [
        rotate_density_matrix(
            rho,
            "X",
            np.pi / 2.0,
        )
        for rho in ensemble
    ]


# ============================================================
# Ensemble quantities
# ============================================================

def ensemble_bloch_vectors(
    ensemble,
):
    """Return all individual Bloch vectors."""

    return np.asarray(
        [
            bloch_vector(rho)
            for rho in ensemble
        ]
    )


def average_bloch_vector(
    ensemble,
):
    """Return the ensemble-average Bloch vector."""

    vectors = ensemble_bloch_vectors(
        ensemble
    )

    return np.mean(
        vectors,
        axis=0,
    )


def transverse_coherence(
    average_vector,
):
    """
    Return magnitude of transverse ensemble coherence.

        C_perp = sqrt(<sx>^2 + <sy>^2)
    """

    return float(
        np.sqrt(
            average_vector[0] ** 2
            +
            average_vector[1] ** 2
        )
    )


# ============================================================
# Frame storage
# ============================================================

def record_frame(
    ensemble,
    all_vectors,
    average_vectors,
    coherences,
    labels,
    label,
):
    """Record the current ensemble state."""

    vectors = ensemble_bloch_vectors(
        ensemble
    )

    average = np.mean(
        vectors,
        axis=0,
    )

    all_vectors.append(
        vectors
    )

    average_vectors.append(
        average
    )

    coherences.append(
        transverse_coherence(
            average
        )
    )

    labels.append(
        label
    )


# ============================================================
# Free evolution
# ============================================================

def evolve_ensemble_free(
    ensemble,
    detunings,
    duration,
    frames,
    all_vectors,
    average_vectors,
    coherences,
    labels,
    label,
):
    """
    Evolve every member under its own static detuning.
    """

    dt = (
        duration / frames
    )

    for _ in range(frames):

        updated = []

        for rho, detuning in zip(
            ensemble,
            detunings,
        ):

            H = free_hamiltonian(
                detuning
            )

            rho_new = evolve_density_matrix(
                rho,
                H,
                dt,
                decoherence=None,
            )

            updated.append(
                rho_new
            )

        ensemble = updated

        record_frame(
            ensemble,
            all_vectors,
            average_vectors,
            coherences,
            labels,
            label,
        )

    return ensemble


# ============================================================
# Hahn pi pulse
# ============================================================

def apply_pi_x(
    ensemble,
    frames,
    all_vectors,
    average_vectors,
    coherences,
    labels,
):
    """
    Apply the pi_X pulse continuously for visualization.
    """

    angle_step = (
        np.pi / frames
    )

    for _ in range(frames):

        ensemble = [
            rotate_density_matrix(
                rho,
                "X",
                angle_step,
            )
            for rho in ensemble
        ]

        record_frame(
            ensemble,
            all_vectors,
            average_vectors,
            coherences,
            labels,
            "Hahn refocusing pulse: pi X",
        )

    return ensemble


# ============================================================
# Build trajectory
# ============================================================

def build_trajectory():

    # Symmetric detuning distribution
    detunings = np.linspace(
        -DELTA_MAX,
        DELTA_MAX,
        N_SPINS,
    )

    ensemble = initial_ensemble()

    # Preparation pulse is applied instantaneously here.
    # The animation focuses on dephasing/refocusing.
    ensemble = prepare_ensemble(
        ensemble
    )

    all_vectors = []
    average_vectors = []
    coherences = []
    labels = []

    record_frame(
        ensemble,
        all_vectors,
        average_vectors,
        coherences,
        labels,
        "Prepared ensemble after pi/2 X",
    )

    # --------------------------------------------------------
    # First free interval
    # --------------------------------------------------------

    ensemble = evolve_ensemble_free(
        ensemble,
        detunings,
        duration=TAU,
        frames=FREE_FRAMES,
        all_vectors=all_vectors,
        average_vectors=average_vectors,
        coherences=coherences,
        labels=labels,
        label="Dephasing under static detuning distribution",
    )

    # --------------------------------------------------------
    # Hahn pi pulse
    # --------------------------------------------------------

    ensemble = apply_pi_x(
        ensemble,
        frames=PULSE_FRAMES,
        all_vectors=all_vectors,
        average_vectors=average_vectors,
        coherences=coherences,
        labels=labels,
    )

    # --------------------------------------------------------
    # Second free interval
    # --------------------------------------------------------

    ensemble = evolve_ensemble_free(
        ensemble,
        detunings,
        duration=TAU,
        frames=FREE_FRAMES,
        all_vectors=all_vectors,
        average_vectors=average_vectors,
        coherences=coherences,
        labels=labels,
        label="Rephasing after pi pulse",
    )

    return (
        np.asarray(
            all_vectors
        ),
        np.asarray(
            average_vectors
        ),
        np.asarray(
            coherences
        ),
        labels,
        detunings,
    )


# ============================================================
# Bloch sphere
# ============================================================

def draw_bloch_sphere(
    ax,
):

    u = np.linspace(
        0.0,
        2.0 * np.pi,
        48,
    )

    v = np.linspace(
        0.0,
        np.pi,
        24,
    )

    x = np.outer(
        np.cos(u),
        np.sin(v),
    )

    y = np.outer(
        np.sin(u),
        np.sin(v),
    )

    z = np.outer(
        np.ones_like(u),
        np.cos(v),
    )

    ax.plot_wireframe(
        x,
        y,
        z,
        linewidth=0.3,
        alpha=0.12,
    )

    # x
    ax.plot(
        [-1.25, 1.25],
        [0.0, 0.0],
        [0.0, 0.0],
        linewidth=0.8,
        alpha=0.4,
    )

    # y
    ax.plot(
        [0.0, 0.0],
        [-1.25, 1.25],
        [0.0, 0.0],
        linewidth=0.8,
        alpha=0.4,
    )

    # z
    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [-1.25, 1.25],
        linewidth=0.8,
        alpha=0.4,
    )

    ax.text(
        1.32,
        0.0,
        0.0,
        "x",
    )

    ax.text(
        0.0,
        1.32,
        0.0,
        "y",
    )

    ax.text(
        0.0,
        0.0,
        1.32,
        r"$|0\rangle$",
    )

    ax.text(
        0.0,
        0.0,
        -1.42,
        r"$|1\rangle$",
    )

    ax.set_xlim(
        -1.4,
        1.4,
    )

    ax.set_ylim(
        -1.4,
        1.4,
    )

    ax.set_zlim(
        -1.4,
        1.4,
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    ax.set_axis_off()


# ============================================================
# Animation
# ============================================================

def create_animation():

    (
        all_vectors,
        average_vectors,
        coherences,
        labels,
        detunings,
    ) = build_trajectory()

    fig = plt.figure(
        figsize=(8, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    draw_bloch_sphere(
        ax
    )

    # --------------------------------------------------------
    # Individual ensemble members
    # --------------------------------------------------------

    points, = ax.plot(
        [],
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=4,
        alpha=0.55,
    )

    # --------------------------------------------------------
    # Individual Bloch-vector rays
    # --------------------------------------------------------

    rays = []

    for _ in range(
        N_SPINS
    ):

        ray, = ax.plot(
            [],
            [],
            [],
            linewidth=0.6,
            alpha=0.20,
        )

        rays.append(
            ray
        )

    # --------------------------------------------------------
    # Ensemble-average Bloch vector
    # --------------------------------------------------------

    average_line, = ax.plot(
        [],
        [],
        [],
        linewidth=4.0,
    )

    average_point, = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=8,
    )

    # --------------------------------------------------------
    # Active pi-pulse axis
    # --------------------------------------------------------

    pulse_axis, = ax.plot(
        [],
        [],
        [],
        linewidth=4.0,
        alpha=0.65,
    )

    title = ax.set_title(
        "Hahn Ensemble Dephasing"
    )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(frame):

        vectors = all_vectors[
            frame
        ]

        average = average_vectors[
            frame
        ]

        coherence = coherences[
            frame
        ]

        # Ensemble tips
        points.set_data(
            vectors[:, 0],
            vectors[:, 1],
        )

        points.set_3d_properties(
            vectors[:, 2]
        )

        # Individual rays
        for ray, vector in zip(
            rays,
            vectors,
        ):

            ray.set_data(
                [
                    0.0,
                    vector[0],
                ],
                [
                    0.0,
                    vector[1],
                ],
            )

            ray.set_3d_properties(
                [
                    0.0,
                    vector[2],
                ]
            )

        # Average vector
        average_line.set_data(
            [
                0.0,
                average[0],
            ],
            [
                0.0,
                average[1],
            ],
        )

        average_line.set_3d_properties(
            [
                0.0,
                average[2],
            ]
        )

        average_point.set_data(
            [average[0]],
            [average[1]],
        )

        average_point.set_3d_properties(
            [average[2]]
        )

        # Highlight x during the Hahn pulse
        if (
            labels[frame]
            == "Hahn refocusing pulse: pi X"
        ):

            pulse_axis.set_data(
                [-1.15, 1.15],
                [0.0, 0.0],
            )

            pulse_axis.set_3d_properties(
                [0.0, 0.0]
            )

        else:

            pulse_axis.set_data(
                [],
                [],
            )

            pulse_axis.set_3d_properties(
                []
            )

        title.set_text(
            "Hahn Echo: Ensemble Dephasing\n"
            f"{labels[frame]}\n"
            rf"$|\langle r_\perp\rangle|$"
            f" = {coherence:.3f}"
        )

        return (
            points,
            average_line,
            average_point,
            pulse_axis,
            *rays,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(
            all_vectors
        ),
        interval=40,
        blit=False,
    )

    animation.save(
        OUTPUT_FILE,
        writer=PillowWriter(
            fps=FPS
        ),
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Detuning distribution: "
        f"{-DELTA_MAX * 1000:.1f} "
        f"to "
        f"{DELTA_MAX * 1000:.1f} MHz"
    )

    print(
        f"Number of ensemble members: "
        f"{N_SPINS}"
    )

    print(
        f"Initial transverse coherence: "
        f"{coherences[0]:.4f}"
    )

    print(
        f"Minimum transverse coherence: "
        f"{np.min(coherences):.4f}"
    )

    print(
        f"Final transverse coherence: "
        f"{coherences[-1]:.4f}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    create_animation()