import numpy as np
import mujoco as mj
import pandas as pd
from utils import get_names
from mocap import load_mocap_data, apply_offsets, mm_to_meters, compute_axis_scaling_factors,apply_axis_scaling
from filters import filter_marker_targets
from ik import enforce_joint_limits, ik_step_multi_site, solve_ik_for_frame
from visualization import simulation_qpos_trajectory, render_qpos_trajectory_to_video, compute_axis_limits, plot_skeleton_at_frame, plot_joint_trajectories
from clear_data import clean_mocap_data
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "03_1_1_pos.tsv"

# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================


def main(model_path, mocap_path, qpos_output_path, video_output_path):
    # ---------------- Load mocap data ----------------
    mocap_data = load_mocap_data(data_path=mocap_path)
    joint_names = get_names(mocap_data)

    # ---------------- Apply offsets ------------------
    mocap_data = apply_offsets(mocap_data)

    # ---------------- Convert mm to m ----------------
    mocap_data = mm_to_meters(mocap_data)

    # ---------------- Clean data ---------------------
    mocap_data = clean_mocap_data(mocap_data)

    # --------------- Plot MoCap data -----------------
    # Compute shared axis limits once
    x_min, x_max, y_min, y_max, z_min, z_max = compute_axis_limits(mocap_data)
    x_limits = (x_min, x_max)
    y_limits = (y_min, y_max)
    z_limits = (z_min, z_max)

    plot_skeleton_at_frame(mocap_data, joint_names, frame_idx=900, x_limits=x_limits, y_limits=y_limits, z_limits=z_limits)

    plot_joint_trajectories(mocap_data,joint_names,x_limits,y_limits,z_limits)

    # --------------- Load MuJoCo model ---------------
    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    print("Number qpos:", model.nq)
    print("Number site:", model.nsite)
    print("Number DOF:", model.nv)

    # ---------------- Axis scaling ----------------
    y_scale, z_scale = compute_axis_scaling_factors(
        model,
        data,
        mocap_data,
        marker_names=[
            "l_shoulder_pos", "r_shoulder_pos",
            "l_hip_pos", "r_hip_pos",
            "l_knee_pos", "r_knee_pos",
        ]
    )

    mocap_data = apply_axis_scaling(mocap_data, y_scale, z_scale)

    # ---------------- Marker & site configuration ----------------
    site_names = [
        "l_shoulder_pos", "r_shoulder_pos",
        "l_elbow_pos", "r_elbow_pos",
        "l_wrist_pos", "r_wrist_pos",
        "l_knee_pos", "r_knee_pos",
        "l_ankle_pos", "r_ankle_pos",
    ]

    site_weights = [
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        2.0, 2.0,
        10.0, 10.0,
    ]

    site_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    ]

    marker_column_indices = [
        mocap_data.columns.get_loc(f"{name}_X")
        for name in site_names
    ]

    # ---------------- Trajectory IK ----------------
    num_frames = 2000
    qpos_trajectory = np.zeros((num_frames, model.nq))

    data.qpos[:] = 0.0
    mj.mj_forward(model, data)

    previous_targets = np.array([
        mocap_data.iloc[0, idx:idx + 3].to_numpy()
        for idx in marker_column_indices
    ])

    for frame_idx in range(num_frames):
        print(f"Frame {frame_idx}")

        current_targets = np.array([
            mocap_data.iloc[frame_idx, idx:idx + 3].to_numpy()
            for idx in marker_column_indices
        ])

        filtered_targets = filter_marker_targets(
            current_targets,
            previous_targets,
        )

        solve_ik_for_frame(
            model,
            data,
            site_ids,
            filtered_targets.tolist(),
            site_weights
        )

        qpos_trajectory[frame_idx] = data.qpos.copy()
        previous_targets = filtered_targets.copy()

    # ---------------- Save qpos ------------------
    df_qpos = pd.DataFrame(qpos_trajectory)
    df_qpos.to_csv(str(qpos_output_path), index=False)
    print(f"Saved qpos to {qpos_output_path}")

    # ---------------- Render video ----------------
    if video_output_path is not None:
        render_qpos_trajectory_to_video(model, qpos_trajectory, str(video_output_path))
        print(f"Saved video to {video_output_path}")
    else:
        print("Video rendering skipped.")

    simulation_qpos_trajectory(model, qpos_trajectory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoCap → MuJoCo IK pipeline")

    parser.add_argument(
        "--input",
        nargs=2,
        type=Path,
        default=[Path("human_marina.xml"), DATA_PATH],
        help="Model XML & MoCap TSV"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qpos.csv"),
        help="qpos CSV"
    )

    parser.add_argument(
        "--video_out",
        type=Path,
        default=None,
        help="video MP4"
    )

    args = parser.parse_args()

    model_path, mocap_path = args.input
    qpos_out = args.output
    video_out = args.video_out

    main(model_path, mocap_path, qpos_out, video_out)
