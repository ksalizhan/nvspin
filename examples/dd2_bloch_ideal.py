"""
Bloch-sphere animations for ideal dynamical decoupling.

Generates:

    examples/output/pdd_bloch.gif
    examples/output/cpmg_bloch.gif
    examples/output/xy8_bloch.gif

Visual conventions
------------------
Free evolution:
    rotation about z caused by static detuning.

X pulse:
    rotation about x.

Y pulse:
    rotation about y.

The active rotation axis is explicitly highlighted during
each part of the sequence.

This is an educational two-level visualization, not yet
the full NV + 13C DD spectroscopy solver.
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
    cpmg,
    pdd,
    rotate_density_matrix,
    xy8,
)

from nvspin.solvers import (
    evolve_density_matrix,
)


# ============================================================
# Demonstration parameters
# ============================================================

DETUNING = 0.005      # GHz = 3 MHz
TAU = 100.0           # ns

FREE_FRAMES = 20
PULSE_FRAMES = 20

FPS = 25


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIRECTORY = (
    Path(__file__).parent
    / "output"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
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
    detuning: float,
):
    """
    Free-precession Hamiltonian:

        H_free = Delta/2 sigma_z
    """

    return (
        0.5
        * detuning
        * SIGMA_Z
    )


# ============================================================
# Frame storage
# ============================================================

def add_frame(
    trajectory,
    labels,
    active_axes,
    rho,
    label,
    active_axis,
):
    """Append one animation frame."""

    trajectory.append(
        bloch_vector(rho)
    )

    labels.append(
        label
    )

    active_axes.append(
        active_axis
    )


# ============================================================
# Free evolution
# ============================================================

def add_free_evolution(
    rho,
    trajectory,
    labels,
    active_axes,
    duration,
    detuning,
    frames=FREE_FRAMES,
):
    """
    Append continuous free precession about z.
    """

    H = free_hamiltonian(
        detuning
    )

    dt = (
        duration / frames
    )

    for _ in range(frames):

        rho = evolve_density_matrix(
            rho,
            H,
            dt,
            decoherence=None,
        )

        add_frame(
            trajectory,
            labels,
            active_axes,
            rho,
            label="Free evolution about z",
            active_axis="Z",
        )

    return rho


# ============================================================
# Pulses
# ============================================================

def add_rotation(
    rho,
    trajectory,
    labels,
    active_axes,
    axis,
    total_angle,
    pulse_name,
    frames=PULSE_FRAMES,
):
    """
    Append a continuous ideal pulse rotation.
    """

    angle_step = (
        total_angle / frames
    )

    for _ in range(frames):

        rho = rotate_density_matrix(
            rho,
            axis,
            angle_step,
        )

        add_frame(
            trajectory,
            labels,
            active_axes,
            rho,
            label=pulse_name,
            active_axis=axis,
        )

    return rho


# ============================================================
# Trajectory
# ============================================================

def build_trajectory(
    sequence,
    detuning=DETUNING,
    tau=TAU,
):
    """
    Construct complete DD trajectory.

    Sequence:

        pi/2_X
        tau/2
        pi
        tau
        pi
        ...
        tau/2
        pi/2_X
    """

    # Initial |0>
    rho = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    trajectory = []
    labels = []
    active_axes = []

    add_frame(
        trajectory,
        labels,
        active_axes,
        rho,
        label="Initial |0>",
        active_axis=None,
    )

    # --------------------------------------------------------
    # Preparation pulse
    # --------------------------------------------------------

    rho = add_rotation(
        rho,
        trajectory,
        labels,
        active_axes,
        axis="X",
        total_angle=np.pi / 2.0,
        pulse_name="Preparation: pi/2 X",
    )

    # --------------------------------------------------------
    # First tau/2
    # --------------------------------------------------------

    rho = add_free_evolution(
        rho,
        trajectory,
        labels,
        active_axes,
        duration=tau / 2.0,
        detuning=detuning,
    )

    # --------------------------------------------------------
    # Refocusing train
    # --------------------------------------------------------

    for pulse_index, pulse in enumerate(
        sequence.pulses
    ):

        rho = add_rotation(
            rho,
            trajectory,
            labels,
            active_axes,
            axis=pulse.axis,
            total_angle=np.pi,
            pulse_name=(
                f"Refocusing pulse "
                f"{pulse_index + 1}: pi {pulse.axis}"
            ),
        )

        if (
            pulse_index
            == sequence.number_of_pulses - 1
        ):
            free_duration = (
                tau / 2.0
            )
        else:
            free_duration = tau

        rho = add_free_evolution(
            rho,
            trajectory,
            labels,
            active_axes,
            duration=free_duration,
            detuning=detuning,
        )

    # --------------------------------------------------------
    # Readout pulse
    # --------------------------------------------------------

    rho = add_rotation(
        rho,
        trajectory,
        labels,
        active_axes,
        axis="X",
        total_angle=np.pi / 2.0,
        pulse_name="Readout: pi/2 X",
    )

    return (
        np.asarray(
            trajectory
        ),
        labels,
        active_axes,
    )


# ============================================================
# Bloch sphere
# ============================================================

def draw_bloch_sphere(
    ax,
):
    """Draw the static Bloch sphere."""

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

    # Coordinate axes
    ax.plot(
        [-1.25, 1.25],
        [0, 0],
        [0, 0],
        linewidth=0.8,
        alpha=0.5,
    )

    ax.plot(
        [0, 0],
        [-1.25, 1.25],
        [0, 0],
        linewidth=0.8,
        alpha=0.5,
    )

    ax.plot(
        [0, 0],
        [0, 0],
        [-1.25, 1.25],
        linewidth=0.8,
        alpha=0.5,
    )

    ax.text(
        1.32,
        0,
        0,
        "x",
    )

    ax.text(
        0,
        1.32,
        0,
        "y",
    )

    ax.text(
        0,
        0,
        1.32,
        r"$|0\rangle$",
    )

    ax.text(
        0,
        0,
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
# Active rotation-axis indicator
# ============================================================

def axis_coordinates(
    axis,
):
    """Return coordinates for an active rotation axis."""

    if axis == "X":

        return (
            [-1.15, 1.15],
            [0.0, 0.0],
            [0.0, 0.0],
        )

    if axis == "Y":

        return (
            [0.0, 0.0],
            [-1.15, 1.15],
            [0.0, 0.0],
        )

    if axis == "Z":

        return (
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.15, 1.15],
        )

    return (
        [],
        [],
        [],
    )


# ============================================================
# Animation
# ============================================================

def create_animation(
    sequence,
    filename,
):
    """Create one annotated Bloch-sphere GIF."""

    (
        trajectory,
        labels,
        active_axes,
    ) = build_trajectory(
        sequence,
    )

    fig = plt.figure(
        figsize=(7, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    draw_bloch_sphere(
        ax
    )

    # --------------------------------------------------------
    # Dynamic artists
    # --------------------------------------------------------

    trail, = ax.plot(
        [],
        [],
        [],
        linewidth=1.4,
        alpha=0.7,
    )

    vector, = ax.plot(
        [],
        [],
        [],
        linewidth=3.0,
    )

    point, = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=7,
    )

    active_axis_line, = ax.plot(
        [],
        [],
        [],
        linewidth=4.0,
        alpha=0.75,
    )

    title = ax.set_title(
        sequence.name
    )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(frame):

        current = trajectory[
            frame
        ]

        travelled = trajectory[
            : frame + 1
        ]

        # Trajectory trail
        trail.set_data(
            travelled[:, 0],
            travelled[:, 1],
        )

        trail.set_3d_properties(
            travelled[:, 2]
        )

        # Bloch vector
        vector.set_data(
            [
                0.0,
                current[0],
            ],
            [
                0.0,
                current[1],
            ],
        )

        vector.set_3d_properties(
            [
                0.0,
                current[2],
            ]
        )

        # Tip
        point.set_data(
            [current[0]],
            [current[1]],
        )

        point.set_3d_properties(
            [current[2]]
        )

        # Active rotation axis
        axis = active_axes[
            frame
        ]

        (
            axis_x,
            axis_y,
            axis_z,
        ) = axis_coordinates(
            axis
        )

        active_axis_line.set_data(
            axis_x,
            axis_y,
        )

        active_axis_line.set_3d_properties(
            axis_z
        )

        # Title
        if axis is None:

            axis_text = (
                "No active rotation"
            )

        elif axis == "Z":

            axis_text = (
                "Precession axis: z"
            )

        else:

            axis_text = (
                f"Pulse rotation axis: {axis.lower()}"
            )

        title.set_text(
            f"{sequence.name}\n"
            f"{labels[frame]}\n"
            f"{axis_text}"
        )

        return (
            trail,
            vector,
            point,
            active_axis_line,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(
            trajectory
        ),
        interval=40,
        blit=False,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / filename
    )

    animation.save(
        output_path,
        writer=PillowWriter(
            fps=FPS
        ),
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("Ideal DD Bloch-sphere animations")
    print("--------------------------------")

    print(
        f"Detuning = "
        f"{DETUNING * 1000:.1f} MHz"
    )

    print(
        f"tau = {TAU:.1f} ns"
    )

    phase = (
        2.0
        * np.pi
        * DETUNING
        * TAU
    )

    print(
        f"Free-precession phase per tau = "
        f"{np.rad2deg(phase):.1f} degrees"
    )

    create_animation(
        pdd(8),
        "pdd_bloch.gif",
    )

    create_animation(
        cpmg(8),
        "cpmg_bloch.gif",
    )

    create_animation(
        xy8(1),
        "xy8_bloch.gif",
    )


if __name__ == "__main__":
    main()