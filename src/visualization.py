# visualization.py
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.viewer
import imageio

MOCAP_SAMPLING_FREQ = 300.
MOCAP_SAMPLING_TIME = 1. / MOCAP_SAMPLING_FREQ


# ============================================================
# -------------------- VISUALIZATIONS ------------------------
# ============================================================


def compute_axis_limits(data):
    """
    Compute global axis limits for X, Y and Z coordinates
    based on all joints in the dataset.
    """
    x_min = data.filter(regex=r'_X$').min().min()
    x_max = data.filter(regex=r'_X$').max().max()

    y_min = data.filter(regex=r'_Y$').min().min()
    y_max = data.filter(regex=r'_Y$').max().max()

    z_min = 0.0
    z_max = data.filter(regex=r'_Z$').max().max()
    z_max = z_max * 5 / 4  # add some headroom above max Z

    return x_min, x_max, y_min, y_max, z_min, z_max


def setup_3d_axis(x_limits, y_limits, z_limits, elev=0, azim=0):
    """
    Create and configure a 3D matplotlib axis with fixed limits and view.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)

    ax.view_init(elev=elev, azim=azim)

    return fig, ax


def plot_skeleton_at_frame(data, joint_names, frame_idx, x_limits, y_limits, z_limits):
    """
    Plot all joints as a single 3D skeleton for a given frame index.
    """
    _, ax = setup_3d_axis(x_limits, y_limits, z_limits)

    for joint in joint_names:
        x = data[f'{joint}_X'][frame_idx]
        y = data[f'{joint}_Y'][frame_idx]
        z = data[f'{joint}_Z'][frame_idx]
        ax.scatter(x, y, z)

    ax.set_title(f'MoCap skeleton frame {frame_idx}')
    plt.show()


def plot_joint_trajectories(data, joint_names, x_limits, y_limits, z_limits):
    """
    Plot full 3D trajectory for each joint separately.
    """
    for joint in joint_names:
        _, ax = setup_3d_axis(x_limits, y_limits, z_limits)

        x = data[f'{joint}_X']
        y = data[f'{joint}_Y']
        z = data[f'{joint}_Z']

        ax.scatter(x, y, z)
        ax.set_title(joint)

        plt.show()


def simulation_qpos_trajectory(
        model: mj.MjModel,
        qpos_trajectory: np.ndarray,
        timestep: float = None
):
    """
    Plays a qpos trajectory in the MuJoCo viewer.
    """

    dt = timestep if timestep is not None else model.opt.timestep
    data = mj.MjData(model)

    # Launch passive viewer for visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for qpos in qpos_trajectory:
            data.qpos[:] = qpos
            mj.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)


def render_qpos_trajectory_to_video(
    model: mj.MjModel,
    qpos_trajectory: np.ndarray,
    video_filename: str="output.mp4",
    dt: float=None,
    fps: int=None,
    width: int=1280,
    height: int=720,
    camera_azimuth=90,
    camera_elevation=-15,
    camera_distance=4.0,
    camera_lookat=(0.0, 0.0, 1.0),
):
    """
    Renders a MuJoCo qpos trajectory to an MP4 video.
    """

    # Timing
    if dt is None:
        dt = model.opt.timestep

    if fps is None:
        fps = int(1 / dt) if dt > 0 else 30

    # Initialize data & renderer
    data_mj = mj.MjData(model)
    renderer = mj.Renderer(model, height, width)

    # Camera setup
    camera = mj.MjvCamera()
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation
    camera.distance = camera_distance
    camera.lookat[:] = camera_lookat

    frames = []

    # Simulation loop
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        for t in range(len(qpos_trajectory)):
            data_mj.qpos[:] = qpos_trajectory[t]
            mj.mj_forward(model, data_mj)

            # Render frame for video
            renderer.update_scene(data_mj, camera)
            frame = renderer.render()
            frames.append(frame.copy())

            viewer.sync()
            time.sleep(dt)

    # Save video
    imageio.mimsave(video_filename, frames, fps=fps)
    print(f"Video saved to {video_filename}")

