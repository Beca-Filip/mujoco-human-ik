import numpy as np
import mujoco as mj
import pandas as pd
from utils import get_names
from mocap import load_mocap_data, apply_offsets, mm_to_meters, compute_axis_scaling_factors,apply_axis_scaling
from filters import filter_marker_targets
from ik import ik_step_multi_site, solve_ik_for_frame
# from ik_qp import solve_ik_for_frame
from visualization import simulation_qpos_trajectory, render_qpos_trajectory_to_video, compute_axis_limits, plot_skeleton_at_frame, plot_joint_trajectories
from clear_data import clean_mocap_data
from pathlib import Path
import argparse
from id import inverse_dynamics
from generate_human_model import generate_human_model
from detecting_corrupted_mocap import filter_markers_from_pairs

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "03_1_1_pos.tsv"

# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================


def main(mocap_path, model_path, out_joint_pos_path, output_video_path, output_xml, sampling_freq, filter_order,
         filter_freq, subj_height, subj_mass, subj_sex, alpha, save_plot_flag, inverse_dynamics_flag, sim_pause_flag):

    # ---------------- Subject, jump & trail ----------
    parts = mocap_path.parts
    subj_trail = parts[1][0:6]
    # print(subj_trail)

    # ---------------- Load mocap data ----------------
    mocap_data = load_mocap_data(data_path=mocap_path)
    marker_names = get_names(mocap_data)
    # print(marker_names)

    # ---------------- Apply offsets ------------------
    mocap_data = apply_offsets(mocap_data)

    # ---------------- Convert mm to m ----------------
    mocap_data = mm_to_meters(mocap_data)

    # ---------------- Clean data ---------------------
    mocap_data = clean_mocap_data(mocap_data)

    # ---------------- Checking data ------------------
    problems, corr_flag = filter_markers_from_pairs(mocap_data, dev_threshold=0.3)
    if corr_flag:
        print("Corrupted MoCap data.")
        print(problems)

    # --------------- Plot MoCap data -----------------
    # Compute shared axis limits once
    x_min, x_max, y_min, y_max, z_min, z_max = compute_axis_limits(mocap_data)
    x_limits = (x_min, x_max)
    y_limits = (y_min, y_max)
    z_limits = (z_min, z_max)

    plot_skeleton_at_frame(mocap_data, marker_names, subj_trail, save_plot_flag, frame_idx=10, x_limits=x_limits, y_limits=y_limits, z_limits=z_limits)

    # plot_joint_trajectories(mocap_data, marker_names, x_limits, y_limits, z_limits)

    # --------------- Load MuJoCo model ---------------
    if model_path is None:
        model_path = generate_human_model(filename=output_xml, mass=subj_mass, height=subj_height, sex=subj_sex, alpha=alpha)
    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    print("Number qpos:", model.nq)
    print("Number site:", model.nsite)
    print("Number DOF:", model.nv)
    print("Number of joints ", model.njnt)

    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        # print('geom_id: ', i, 'body_id: ', body_id,
        #       'body_name: ', model.body(body_id).name, ', geom_type: ', model.geom(i).type)

    all_site_names = [model.site(i).name for i in range(model.nsite)]
    # print(all_site_names)

    # --------- MuJoCo sites to MoCap marker -----------
    mujoco_to_mocap = {
        "greater_trochanter_left": "l_hip_pos",
        "greater_trochanter_right": "r_hip_pos",
        "lateral_femoral_epicondyle_left": "l_knee_pos",
        "lateral_femoral_epicondyle_right": "r_knee_pos",
        "lateral_maleollus_left": "l_ankle_pos",
        "lateral_maleollus_right": "r_ankle_pos",
        "metatarsal_fifth_left": "l_metatarsal_pos",
        "metatarsal_fifth_right": "r_metatarsal_pos",
        "acromion_left": "l_shoulder_pos",
        "acromion_right": "r_shoulder_pos",
        "lateral_humeral_epicondyle_left": "l_elbow_pos",
        "lateral_humeral_epicondyle_right": "r_elbow_pos",
        "ulnar_styloid_left": "l_wrist_pos",
        "ulnar_styloid_right": "r_wrist_pos"
    }

    # ---------------- Axis scaling ----------------
    if str(model_path) == 'human_marina.xml':
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
    if str(model_path) == 'human_marina.xml':
        site_names = [x for x in all_site_names if (x != 'r_heel_pos' and x != 'l_heel_pos')]
    else:
        site_names = [x for x in all_site_names if (x != 'thorax_front_site')]

    sites_to_weights = {
        "greater_trochanter_left": 1.0,
        "greater_trochanter_right": 1.0,
        "lateral_femoral_epicondyle_left": 1.0,
        "lateral_femoral_epicondyle_right": 1.0,
        "lateral_maleollus_left": 2.0,
        "lateral_maleollus_right": 2.0,
        "metatarsal_fifth_left": 1.0,
        "metatarsal_fifth_right": 1.0,
        "acromion_left": 1.0,
        "acromion_right": 1.0,
        "lateral_humeral_epicondyle_left": 1.0,
        "lateral_humeral_epicondyle_right": 1.0,
        "ulnar_styloid_left": 1.0,
        "ulnar_styloid_right": 1.0
    }

    site_weights = [sites_to_weights[name] for name in site_names]
    # print(site_names)
    # print(site_weights)

    site_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    ]

    if str(model_path) == 'human_marina.xml':
        marker_column_indices = [
            mocap_data.columns.get_loc(f"{name}_X")
            for name in site_names
        ]
    else:
        marker_column_indices = [
            mocap_data.columns.get_loc(f"{mujoco_to_mocap[name]}_X")
            for name in site_names
        ]

    # ---------------- Trajectory IK ----------------
    num_frames = mocap_data.shape[0]
    qpos_trajectory = np.zeros((num_frames, model.nq))

    # data.qpos[:] = 0.0
    mj.mj_forward(model, data)

    # qpos_trajectory[0] = data.qpos.copy()
    # np.set_printoptions(threshold=np.inf)
    # print(qpos_trajectory[0])

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
            site_weights,
        )

        qpos_trajectory[frame_idx] = data.qpos.copy()
        previous_targets = filtered_targets.copy()

    # ---------------- Save qpos ------------------
    if out_joint_pos_path is None:
        out_joint_pos_path = "qpos_" + subj_trail + ".csv"
    df_qpos = pd.DataFrame(qpos_trajectory)
    df_qpos.to_csv(str(out_joint_pos_path), index=False)
    print(f"Saved qpos to {out_joint_pos_path}")

    # ---------------- Render video ----------------
    if output_video_path is not None:
        render_qpos_trajectory_to_video(model, qpos_trajectory, str(output_video_path))
    else:
        print("Video rendering skipped.")

    simulation_qpos_trajectory(model, qpos_trajectory, pause_flag=sim_pause_flag)

    # --------------- Inverse dynamics ---------------
    if inverse_dynamics_flag:
        inverse_dynamics(model_path, out_joint_pos_path, subj_trail, sampling_freq, filter_order, filter_freq, save_plot_flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoCap → MuJoCo IK pipeline")

    parser.add_argument(
        "--input_tsv",
        type=Path,
        help="MoCap TSV"
    )

    parser.add_argument(
        "--input_xml",
        type=Path,
        default = None,
        help="Path to MuJoCo XML Model."
    )

    parser.add_argument(
        "--output_joint_pos",
        type=Path,
        default=None,
        help="qpos CSV"
    )


    parser.add_argument(
        "--export_video",
        type=Path,
        default=None,
        help="video MP4"
    )

    parser.add_argument(
        "--xml_output",
        default="human.xml",
        help="Output XML filename (default: human.xml)"
    )

    parser.add_argument(
        "--sampling_frequency",
        type=float,
        default=300.0,
        help="MoCap sampling frequency"
    )

    parser.add_argument(
        "--filter_order",
        type=int,
        default=4,
        help="Filter order"
    )

    parser.add_argument(
        "--filter_frequency",
        type=float,
        default=10.0,
        help="filter frequency"
    )

    parser.add_argument(
        "--subject_height",
        required=True,
        type=float,
        help="Height of the human who's MJCF we want to generate in m"
    )

    parser.add_argument(
        "--subject_mass",
        required=True,
        type=float,
        help="Mass of the human who's MJCF we want to generate in kilogram"
    )

    parser.add_argument(
        "--subject_sex",
        required=True,
        choices=["male", "female"],
        help="Sex of the human who's MJCF we want to generate ('male' or 'female')"
    )

    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=1.0,
        help="The alpha value of the generated mesh (0: transparent, 1: opaque)"
    )

    parser.add_argument(
        "--save_plots",
        type=bool,
        default=False,
        help="Do you want to save plots?"
    )

    parser.add_argument(
        "--inverse_dynamics",
        type=bool,
        default=True,
        help="Do the inverse dynamics"
    )

    parser.add_argument(
        "--pause_simulation",
        type=bool,
        default=False,
        help="Pause simulation"
    )

    args = parser.parse_args()

    mocap_path = args.input_tsv
    out_joint_pos_path = args.output_joint_pos
    output_video_path = args.export_video
    output_xml = args.xml_output
    sampling_freq = args.sampling_frequency
    filter_order = args.filter_order
    filter_freq = args.filter_frequency
    subj_height = args.subject_height
    subj_mass = args.subject_mass
    subj_sex = args.subject_sex
    alpha = args.alpha
    save_plots = args.save_plots
    model_path = args.input_xml
    inv_dyn = args.inverse_dynamics
    sim_pause_flag = args.pause_simulation

    main(mocap_path, model_path, out_joint_pos_path, output_video_path, output_xml, sampling_freq, filter_order,
         filter_freq, subj_height, subj_mass, subj_sex, alpha, save_plots, inv_dyn, sim_pause_flag)
