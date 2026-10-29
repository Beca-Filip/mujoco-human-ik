# batch.py
"""
Batch processing pipeline for MoCap -> MuJoCo inverse kinematics.

This script iterates through all TSV MoCap files in a directory,
loads subject-specific anthropometric information, validates subject data,
and runs the inverse kinematics pipeline for each trial.

For every successfully processed trial, the generated outputs may include:
- joint position trajectories (qpos)
- inverse dynamics results
- simulation videos
- subject-specific MuJoCo XML models

Any failed trials are recorded in a .txt file together with the
corresponding exception traceback.
"""

from .one_trial_ik import *
import os
import traceback

MAX_HEIGHT = 2.5
MIN_HEIGHT = 1.3
MAX_MASS = 200
MIN_MASS = 20


def run_batch(
    mocap_path_to_dir: Path,
    subj_info: Path | None = None,
    model_path: Path | None = None,
    out_joint_pos_path: Path | None = None,
    output_video_path: Path | None = None,
    output_xml: str = "human.xml",
    sampling_freq: float = 300.0,
    filter_order: int = 4,
    filter_freq: float = 10.0,
    alpha: float = 1.0,
    save_plot_flag: bool = True,
    inverse_dynamics_flag: bool = True,
    sim_flag: bool = False,
    sim_pause_flag:bool = False,
    error_txt_path: Path = Path("failed_mocaps.txt"),
    results_dir: Path = Path("Results"),
    preprocess_flag: bool = True,
):
    """
    Process all MoCap trials from a directory using the MuJoCo IK pipeline.

    Parameters
    ----------
    mocap_path_to_dir : Path
        Directory containing MoCap TSV files.
    subj_info : Path
        Path to the CSV file containing subject information.
        The file is expected to contain the following columns:
        ID, name, height, mass, and gender.
    model_path : Path, optional
        Path to an existing MuJoCo model. If None, a model is generated.
    out_joint_pos_path : Path, optional
        Directory where qpos CSV files will be stored.
    output_video_path : Path, optional
        Directory where rendered simulation videos will be stored.
    output_xml : str, optional
        Filename of the generated subject-specific MuJoCo XML model.
        Default is "human.xml".
    sampling_freq : float, optional
        Motion-capture sampling frequency in Hz.
    filter_order : int, optional
        Order of the filtering operation used in inverse dynamics.
    filter_freq : float, optional
        Cutoff frequency of the low-pass filter in Hz.
    alpha : float, optional
        Transparency of generated meshes.
    save_plot_flag : bool, optional
        Whether plots should be generated and saved.
    inverse_dynamics_flag : bool, optional
        Whether inverse dynamics should be computed.
    sim_flag : bool, optional
        Whether to run the MuJoCo simulation visualization.
    sim_pause_flag : bool, optional
        Whether the simulation should be paused.
    error_txt_path : Path | str, optional
        Path to the text file used for logging failed trials.
    results_dir : Path | str, optional
        Directory where all generated results will be stored.
    preprocess_flag : bool, optional
        Whether marker preprocessing and filtering should be applied.

    Notes:
    Subject height and mass values are validated before processing.
    Trials with missing subject information or invalid anthropometric
    values are skipped and recorded in the error log.
    """

    with (open(error_txt_path, "w") as f):
        f.write("Failed mocaps:\n\n")

        subj_data = pd.read_csv(subj_info, header=None)
        header = ['id', 'name', 'tall', 'mass', 'gender']
        subj_data.columns = header

        for file in sorted(os.listdir(mocap_path_to_dir)):
            if not file.endswith(".tsv"):
                continue

            mocap_file = os.path.join(mocap_path_to_dir, file)
            print(f"Processing: {file}")

            subj_id = int(file[0:2])
            row_for_id = subj_data['id'] == subj_id
            if row_for_id.any():
                subj_height = subj_data.loc[row_for_id, 'tall'].iloc[0]
                subj_mass = subj_data.loc[row_for_id, 'mass'].iloc[0]
                subj_sex = subj_data.loc[row_for_id, 'gender'].iloc[0]
            else:
                f.write(mocap_file + '-> ' + f"ID: {subj_id} doesn't exist." + '\n')
                continue

            if (subj_height > MAX_HEIGHT or subj_height < MIN_HEIGHT or
                    subj_mass > MAX_MASS or subj_mass < MIN_MASS):
                f.write(mocap_file + '-> ' + f"Invalid subject information (mass or height not in range)" + '\n')
                continue

            mocap_file_path = Path(mocap_file)

            try:
                one_trial_ik(mocap_file_path, model_path, out_joint_pos_path, output_video_path, output_xml,
                             sampling_freq, filter_order, filter_freq, subj_height, subj_mass, subj_sex, alpha,
                             save_plot_flag, inverse_dynamics_flag, sim_flag, sim_pause_flag, error_txt_path,
                             results_dir, preprocess_flag, print_flag=False, error_txt_file=f)
                print(f"Done: {file}")

            except Exception as e:
                print(f"FAILED: {file}")
                f.write(f"{mocap_file}\n")
                f.write(str(e) + "\n")
                f.write(traceback.format_exc())
                f.write("\n-------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoCap → MuJoCo IK pipeline")

    parser.add_argument(
        "--mocap_dir",
        type=Path,
        help="MoCap TSV"
    )

    parser.add_argument(
        "--subj_info",
        type=Path,
        help="Path to the CSV file with subject information (tall, mass, gender)"
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
        "--alpha",
        required=False,
        type=float,
        default=1.0,
        help="The alpha value of the generated mesh (0: transparent, 1: opaque)"
    )

    parser.add_argument(
        "--save_plots",
        type=bool,
        default=True,
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

    mocap_path = args.mocap_dir
    subj_info = args.subj_info
    model_path = args.input_xml
    out_joint_pos_path = args.output_joint_pos
    output_video_path = args.export_video
    output_xml = args.xml_output
    sampling_freq = args.sampling_frequency
    filter_order = args.filter_order
    filter_freq = args.filter_frequency
    alpha = args.alpha
    save_plots = args.save_plots
    inv_dyn = args.inverse_dynamics
    simulation_flag = args.simulation
    sim_pause_flag = args.pause_simulation
    error_txt = args.error_txt
    results_dir = args.results_dir
    preprocess_flag = args.preprocessing

    run_batch(mocap_path, subj_info, model_path, out_joint_pos_path, output_video_path, output_xml,
              sampling_freq, filter_order, filter_freq, alpha, save_plots, inv_dyn, simulation_flag, sim_pause_flag,
              error_txt, results_dir, preprocess_flag)