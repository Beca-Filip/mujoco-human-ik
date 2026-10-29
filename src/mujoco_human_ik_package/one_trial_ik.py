# one_trial_ik.py
"""
MoCap → MuJoCo inverse kinematics pipeline.

This module:
1. Load MoCap TSV data
2. Applies preprocessing and data validation
3. Generates or loads a subject-specific MuJoCo model
4. Map anatomical markers to MuJoCo sites (shoulders, elbows, wrists, knees, ankles)
5. Solve IK per frame with site weights (ankles: 10.0, knees: 2.0, upper body: 1.0)
6. Exports the resulting qpos trajectory
7. Optionally renders a video, runs a simulation, and performs inverse dynamics
"""
import numpy as np
import mujoco as mj
import pandas as pd
from .utils import get_names
from .mocap import load_mocap_data, apply_offsets, mm_to_meters, averaging_z_coord
from .filters import filter_marker_targets
from .ik_symmetry import solve_ik_for_frame
from .visualization import simulation_qpos_trajectory, render_qpos_trajectory_to_video, compute_axis_limits, plot_skeleton_at_frame, plot_joint_trajectories
from .clear_data import clean_mocap_data
from pathlib import Path
import os
import argparse
from .id import inverse_dynamics
from .generate_human_model_old import generate_human_model
from .detecting_corrupted_mocap import filter_markers_from_pairs, constant_values
from typing import Literal, TextIO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "03_1_1_pos.tsv"
MAX_HEIGHT = 2.5
MIN_HEIGHT = 1.3
MAX_MASS = 200
MIN_MASS = 20


def one_trial_ik(
    mocap_path: Path,
    model_path: Path | None = None,
    out_joint_pos_path: Path | None = None,
    output_video_path: Path | None = None,
    output_xml: str = "human.xml",
    sampling_freq: float = 300.0,
    filter_order: int = 4,
    filter_freq: float = 10.0,
    subj_height: float | None = None,
    subj_mass: float | None = None,
    subj_sex: Literal["male", "female"] | None = None,
    alpha: float = 1.0,
    save_plot_flag: bool = False,
    inverse_dynamics_flag: bool = False,
    sim_flag: bool = False,
    sim_pause_flag: bool = False,
    error_txt_path: Path = Path("failed_mocaps.txt"),
    results_dir: Path = Path("Results"),
    preprocessing_flag: bool = True,
    print_flag: bool = False,
    error_txt_file: TextIO | None = None,
):
    """
    Run the complete MoCap-to-MuJoCo inverse kinematics pipeline for a single trial.

    Parameters
    ----------
    mocap_path : Path
        Path to the input MoCap TSV file.
    model_path : Path | None, optional
        Path to an existing MuJoCo model. If None, a model is generated.
    out_joint_pos_path : Path | None, optional
        Output CSV file containing the computed qpos trajectory.
    output_video_path : Path | None, optional
        Output MP4 file for rendered motion visualization.
    output_xml : str, optional
        Name of the generated MuJoCo XML model.
    sampling_freq : float, optional
        MoCap sampling frequency in Hz.
    filter_order : int, optional
        Order of the filtering operation used in inverse dynamics.
    filter_freq : float, optional
        Cutoff frequency used in inverse dynamics.
    subj_height : float
        Subject height in meters.
    subj_mass : float
        Subject mass in kilograms.
    subj_sex : {"male", "female"}
        Subject biological sex.
    alpha : float, optional
        Mesh transparency value for the generated model.
    save_plot_flag : bool, optional
        Whether plots should be saved.
    inverse_dynamics_flag : bool, optional
        Whether inverse dynamics should be performed.
    sim_flag : bool, optional
        Whether to run the MuJoCo simulation visualization.
    sim_pause_flag : bool, optional
        Whether the simulation should be paused.
    error_txt_path : Path, optional
        Path to the .txt file used for recording problematic trials.
    results_dir : Path, optional
        Directory where outputs are stored.
    preprocessing_flag : bool, optional
        Whether preprocessing should be applied to the MoCap data.
    print_flag : bool, optional
        Whether diagnostic information should be printed.
    error_txt_file : TextIO | None, optional
        Existing error log file handle.

    """

    # Create a TXT file for collecting potential errors in MoCap data
    flag_file_errors_exists = False
    if error_txt_file is None:
        with open(error_txt_path, "w") as error_txt_file:
            error_txt_file.write('Failed mocaps:\n\n')
        flag_file_errors_exists = True

    # ---------------- Subject, jump & trail ----------
    parts = mocap_path.parts
    subj_trail = parts[1][0:6]

    # ---------------- Load mocap data ----------------
    mocap_data = load_mocap_data(data_path=mocap_path)
    if mocap_data is None:
        if flag_file_errors_exists is True:
            with open(error_txt_path, "a") as error_txt_file:
                error_txt_file.write(str(mocap_path) + '-> ' + "Empty file" + '\n')
        else:
            error_txt_file.write(str(mocap_path) + '-> ' + "Empty file" + '\n')
        return
    if mocap_data is True:
        if flag_file_errors_exists is True:
            with open(error_txt_path, "a") as error_txt_file:
                error_txt_file.write(str(mocap_path) + '-> ' + "Incorrect number of markers" + '\n')
        else:
            error_txt_file.write(str(mocap_path) + '-> ' + "Incorrect number of markers" + '\n')
        return
    marker_names = get_names(mocap_data)

    # ---------------- Preprocessing ------------------
    if preprocessing_flag is True:

        # ---------------- Apply offsets ------------------
        mocap_data = apply_offsets(mocap_data)

        # ---------------- Convert mm to m ----------------
        mocap_data = mm_to_meters(mocap_data)

        # ---------------- Clean data ---------------------
        mocap_data = clean_mocap_data(mocap_data, subj_trail)

        # ---------------- Checking data ------------------
        problems, corr_flag = filter_markers_from_pairs(mocap_data, dev_threshold=0.3)
        if corr_flag:
            if flag_file_errors_exists is True:
                with open(error_txt_path, "a") as error_txt_file:
                    error_txt_file.write(str(mocap_path) + '-> ' + "Potentially corrupted MoCap data" + '\n')
            else:
                error_txt_file.write(str(mocap_path) + '-> ' + "Potentially corrupted MoCap data" + '\n')
            if print_flag:
                print("Corrupted MoCap data.")

        const_markers, const_flag = constant_values(mocap_data)
        if const_flag:
            if flag_file_errors_exists is True:
                with open(error_txt_path, "a") as error_txt_file:
                    error_txt_file.write(str(mocap_path) + '-> Same value across all frames of those markers' + str(const_markers) + '\n')
            else:
                error_txt_file.write(str(mocap_path) + '-> Same value across all frames of those markers' + str(const_markers) + '\n')
            if print_flag:
                print("Corrupted MoCap data.")

        # ------------- Averaging z axis ------------------
        mocap_data_avr = averaging_z_coord(mocap_data, 'l_metatarsal_pos_Z', 'r_metatarsal_pos_Z')
        mocap_data_avr = averaging_z_coord(mocap_data_avr, 'l_ankle_pos_Z', 'r_ankle_pos_Z')

    # --------------- Plot MoCap data -----------------
    # Compute shared axis limits once
    x_min, x_max, y_min, y_max, z_min, z_max = compute_axis_limits(mocap_data_avr)
    x_limits = (x_min, x_max)
    y_limits = (y_min, y_max)
    z_limits = (z_min, z_max)

    # plot_skeleton_at_frame(mocap_data_avr, marker_names, subj_trail, save_plot_flag, frame_idx=10, x_limits=x_limits, y_limits=y_limits, z_limits=z_limits)

    # plot_joint_trajectories(mocap_data, marker_names, x_limits, y_limits, z_limits)

    # --------------- Load MuJoCo model ---------------
    if model_path is None:
        if subj_height is None or subj_mass is None or subj_sex is None:
            raise ValueError('Height, mass and sex must be provided when generating a model.')

        if (subj_height > MAX_HEIGHT or subj_height < MIN_HEIGHT or
                subj_mass > MAX_MASS or subj_mass < MIN_MASS):
            raise ValueError('Invalid subject information (mass or height not in range)')

        if output_xml == 'human.xml':
            output_xml = 'human_' + subj_trail[0:2] + '.xml'

        model_path = generate_human_model(filename=output_xml, mass=subj_mass, height=subj_height, sex=subj_sex, alpha=alpha)

    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    if print_flag:
        print("Number qpos:", model.nq)
        print("Number site:", model.nsite)
        print("Number DOF:", model.nv)
        print("Number of joints: ", model.njnt)
        print("Number of frames: ", mocap_data_avr.shape[0])

    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        if print_flag:
            print('geom_id: ', i, 'body_id: ', body_id,
                  'body_name: ', model.body(body_id).name, ', geom_type: ', model.geom(i).type)

    all_site_names = [model.site(i).name for i in range(model.nsite)]
    if print_flag:
        print('Site names: \n', all_site_names)

    # qpos_names = list(range(model.nq))
    # name = ''
    # for j in range(model.njnt):
    #     start = model.jnt_qposadr[j]
    #     if j < model.njnt - 1:
    #         end = model.jnt_qposadr[j + 1]
    #     else:
    #         end = model.nq
    #     for idx in range(model.nq):
    #         if start <= idx < end:
    #             name = model.joint(j).name
    #             print(f"qpos index {idx} -> {name}")
    #             if name == '':
    #                 qpos_names[idx] = idx
    #             else:
    #                 qpos_names[idx] = name
    # print(qpos_names)
    # import csv
    # with open("qpos_names.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(qpos_names)

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

    # ---------------- Marker & site configuration ----------------
    site_names = [x for x in all_site_names if (x != 'thorax_front_site' and
                                                    x != 'under_metatarsal_left' and x != 'under_metatarsal_right' and
                                                    x != 'under_maleollus_left' and x != 'under_maleollus_right')]

    sites_to_weights = {
        "greater_trochanter_left": 5.0,
        "greater_trochanter_right": 5.0,
        "lateral_femoral_epicondyle_left": 1.0,
        "lateral_femoral_epicondyle_right": 1.0,
        "lateral_maleollus_left": 2.0,
        "lateral_maleollus_right": 2.0,
        "metatarsal_fifth_left": 10.0,
        "metatarsal_fifth_right": 10.0,
        "acromion_left": 1.0,
        "acromion_right": 1.0,
        "lateral_humeral_epicondyle_left": 1.0,
        "lateral_humeral_epicondyle_right": 1.0,
        "ulnar_styloid_left": 1.0,
        "ulnar_styloid_right": 1.0
    }

    site_weights = [sites_to_weights[name] for name in site_names]
    if print_flag:
        print('Site weights: \n', site_weights)

    site_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    ]

    marker_column_indices = [
        mocap_data_avr.columns.get_loc(f"{mujoco_to_mocap[name]}_X")
        for name in site_names
    ]

    # ---------------- Trajectory IK ----------------
    num_frames = mocap_data_avr.shape[0]
    qpos_trajectory = np.zeros((num_frames, model.nq))

    mj.mj_forward(model, data)

    previous_targets = np.array([
        mocap_data_avr.iloc[0, idx:idx + 3].to_numpy()
        for idx in marker_column_indices
    ])

    for frame_idx in range(num_frames):
        if print_flag:
            print(f"Frame {frame_idx}")

        current_targets = np.array([
            mocap_data_avr.iloc[frame_idx, idx:idx + 3].to_numpy()
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
    if out_joint_pos_path is None:
        out_joint_pos_path = "qpos_" + subj_trail + ".csv"
    df_qpos = pd.DataFrame(qpos_trajectory)
    dir_path = Path(os.path.join(results_dir, Path('qpos')))
    dir_path.mkdir(parents=True, exist_ok=True)
    total_qpos_path = os.path.join(dir_path, out_joint_pos_path)
    df_qpos.to_csv(str(total_qpos_path), index=False)
    if print_flag:
        print(f"Saved qpos to {total_qpos_path}")

    # ---------------- Render video ----------------
    if output_video_path is not None:
        dir_path = Path(os.path.join(results_dir, Path('Videos')))
        dir_path.mkdir(parents=True, exist_ok=True)
        total_video_path = os.path.join(dir_path, output_video_path)
        render_qpos_trajectory_to_video(model, qpos_trajectory, str(total_video_path))
    else:
        if print_flag:
            print("Video rendering skipped.")

    if sim_flag:
        simulation_qpos_trajectory(model, qpos_trajectory, pause_flag=sim_pause_flag)

    # --------------- Inverse dynamics ---------------
    if inverse_dynamics_flag:
        inverse_dynamics(model_path, Path(total_qpos_path)  , subj_trail, sampling_freq, filter_order, filter_freq, save_plot_flag, results_dir, print_flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoCap → MuJoCo IK pipeline")

    parser.add_argument(
        "--input_tsv",
        required=True,
        type=Path,
        help="MoCap TSV"
    )

    parser.add_argument(
        "--input_xml",
        type=Path,
        default=None,
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
        type=float,
        help="Height of the human who's MJCF we want to generate in m"
    )

    parser.add_argument(
        "--subject_mass",
        type=float,
        help="Mass of the human who's MJCF we want to generate in kilogram"
    )

    parser.add_argument(
        "--subject_sex",
        choices=["male", "female"],
        help="Sex of the human who's MJCF we want to generate ('male' or 'female')"
    )

    parser.add_argument(
        "--alpha",
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
        "--simulation",
        type=bool,
        default=False,
        help="Playing simulation? [True/False]"
    )

    parser.add_argument(
        "--pause_simulation",
        type=bool,
        default=False,
        help="Pause simulation"
    )

    parser.add_argument(
        "--error_txt",
        type=Path,
        default="failed_mocaps.txt",
        help="Path to TXT file for catching errors"
    )

    parser.add_argument(
        "--print_flag",
        type=bool,
        default=False,
        help="Do you want to print text info?"
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        default='Results',
        help="Path to directory for results"
    )

    parser.add_argument(
        "--preprocessing",
        type=bool,
        default=True,
        help="Do the preprocessing"
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
    simulation_flag = args.simulation
    sim_pause_flag = args.pause_simulation
    error_txt = args.error_txt
    print_flag = args.print_flag
    results_dir = args.results_dir
    preprocess_flag = args.preprocessing

    one_trial_ik(mocap_path, model_path, out_joint_pos_path, output_video_path, output_xml, sampling_freq, filter_order,
                 filter_freq, subj_height, subj_mass, subj_sex, alpha, save_plots, inv_dyn, simulation_flag,
                 sim_pause_flag, error_txt, results_dir, preprocess_flag, print_flag)
