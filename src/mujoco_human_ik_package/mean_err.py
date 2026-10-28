# mean_err.py
"""
Compute marker tracking errors between MuJoCo model sites and motion capture markers.

For each trial, script compares the positions of selected MuJoCo sites against
their corresponding motion capture markers and computes:
- Root Mean Square Error (RMSE)
- Median Euclidean distance error
- 95th percentile (P95) distance error

Errors are reported in centimeters and aggregated across all processed trials.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import mujoco as mj
import argparse
from .mocap import load_mocap_data, apply_offsets, mm_to_meters, compute_axis_scaling_factors,apply_axis_scaling, averaging_z_coord
from .filters import filter_marker_targets
from .utils import get_names
from .clear_data import clean_mocap_data
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


MARKER_NUM = 14

def mean_err(qpos_path_dir: Path):
    """
    Compute global marker tracking errors for a set of qpos trajectories.

    For each CSV file in the provided directory, the function loads the
    corresponding MuJoCo model and motion capture data, reconstructs marker
    positions from the model, and computes the Euclidean distance error
    between simulated and measured marker locations.

    The following metrics are calculated:
    - Root Mean Square Error (RMSE)
    - Median Euclidean distance error
    - 95th percentile (P95) distance error

    Parameters
    ----------
    qpos_path_dir : Path
        Path to a directory containing qpos trajectory CSV files.
        Each CSV file must correspond to a trial and follow the expected
        naming convention used by the dataset.

    Notes:
    Errors are expressed in centimeters.
    The function assumes that:
    - Subject-specific MuJoCo models are available as
      `human_<subject_id>.xml`
    - Motion capture files are located in the `data` directory
    - CSV filenames follow the pattern
      `<qpos>_<subject>_<jump>_<trial>.csv`
    """

    csv_files = list(qpos_path_dir.glob("*.csv"))
    n_files = len(csv_files)

    i = 0

    trial_site = np.zeros((n_files, MARKER_NUM))
    trial_site_median = np.zeros((n_files, MARKER_NUM))
    outlier = 0
    trial_site_p95 = np.zeros((n_files, MARKER_NUM))
    all_dist = []

    for file in sorted(os.listdir(qpos_path_dir)):
        if i > n_files-1:
            continue
        if not file.endswith(".csv"):
            continue

        qpos_csv_path = os.path.join(qpos_path_dir, file)
        parts = file.split('_')
        subj = parts[1]
        jump = parts[2]
        trial = parts[3].split('.')[0]
        subj_trial = subj + '_' + jump + '_' + trial
        #
        # if subj == '14' and jump == '1':
        #     continue           # These trials are invalid for our dataset, so we skip them
        print(f"Processing: {file}")

        qpos = pd.read_csv(qpos_csv_path).to_numpy()
        T = qpos.shape[0]
        time_vector = np.arange(0, T)

        model = mj.MjModel.from_xml_path(str('human_' + subj + '.xml'))

        data = mj.MjData(model)
        mj.mj_forward(model, data)

        mocap_path = 'data/' + subj_trial + '_pos.tsv'
        mocap_data = load_mocap_data(data_path=mocap_path)
        marker_names = get_names(mocap_data)

        # ---------------- Apply offsets ------------------
        mocap_data = apply_offsets(mocap_data)

        # ---------------- Convert mm to m ----------------
        mocap_data = mm_to_meters(mocap_data)

        # ---------------- Clean data ---------------------
        mocap_data = clean_mocap_data(mocap_data, subj_trial)

        # ------------- Averaging z axis ------------------
        mocap_data_avr = averaging_z_coord(mocap_data, 'l_metatarsal_pos_Z', 'r_metatarsal_pos_Z')
        mocap_data_avr = averaging_z_coord(mocap_data_avr, 'l_ankle_pos_Z', 'r_ankle_pos_Z')

        all_site_names = [model.site(i).name for i in range(model.nsite)]
        site_names = [x for x in all_site_names if (x != 'thorax_front_site' and
                                                    x != 'under_metatarsal_left' and x != 'under_metatarsal_right' and
                                                    x != 'under_maleollus_left' and x != 'under_maleollus_right')]

        site_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
            for name in site_names
        ]

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

        marker_column_indices = [
            mocap_data_avr.columns.get_loc(f"{mujoco_to_mocap[name]}_X")
            for name in site_names
        ]

        marker_pos = np.zeros((T, len(site_ids), 3))
        for k, name in enumerate(site_names):
            base = mujoco_to_mocap[name]
            marker_pos[:, k, 0] = mocap_data_avr[f"{base}_X"].to_numpy()
            marker_pos[:, k, 1] = mocap_data_avr[f"{base}_Y"].to_numpy()
            marker_pos[:, k, 2] = mocap_data_avr[f"{base}_Z"].to_numpy()

        site_pos = np.zeros((T, len(site_ids), 3))
        for frame in range(mocap_data.shape[0]):
            data.qpos[:] = qpos[frame]
            mj.mj_forward(model, data)
            for k, sid in enumerate(site_ids):
                site_pos[frame, k] = data.site_xpos[sid]

        # qpos_traj shape: (T, nq)
        # marker_pos shape: (T, n_markers, 3)

        # errors
        err = site_pos - marker_pos  # (T, n_markers, 3) per marker, per axis
        err = err * 100  # m to cm

        err2 = np.sum(err ** 2, axis=-1)  # (T, n_markers)   errx2 + erry2 + errz2 dist per marker
        dist = np.sqrt(err2)
        all_dist.append(dist)

        trial_site[i, :] = np.mean(err2, axis=0)

        trial_site_median[i, :] = np.median(dist, axis=0)

        trial_site_p95[i, :] = np.percentile(dist, 95, axis=0)

        # if np.sqrt(trial_site[i, -1]) > 10:
        #     outlier = outlier +1
        #     print('outlier trial mean: ', subj_trial)
        # if np.sqrt(trial_site_median[i, -1]) > 10:
        #     print('outlier trial median: ', subj_trial)
        # print('value last marker mean: ', np.sqrt(trial_site[i, -1]))
        # print('value last marker median: ', np.sqrt(trial_site_median[i, -1]))

        i = i+1

    # print('outlier num: ', outlier)

    all_dist = np.concatenate(all_dist, axis=0)

    mse_per_marker_global = np.mean(trial_site, axis=0)
    rmse_per_marker = np.sqrt(mse_per_marker_global)
    print('RMSE per marker: ', rmse_per_marker)
    print('Mean RMSE: ', np.mean(rmse_per_marker))

    # median_per_marker = np.median(trial_site_median, axis=0)
    # print('Median per marker: ', median_per_marker)
    # print('Median marker: ', np.median(median_per_marker))
    #
    # p95_per_marker = np.percentile(trial_site_p95, 95, axis=0)
    # print('P95 per marker: ', p95_per_marker)
    # print('P95: ', np.percentile(p95_per_marker, 95))

    median_per_marker_all_dist = np.median(all_dist, axis=0)
    print('Median per marker all dist: ', median_per_marker_all_dist)
    print('Median marker: ', np.median(median_per_marker_all_dist))

    p95_per_marker_all_dist = np.percentile(all_dist, 95, axis=0)
    print('P95 per marker all dist: ', p95_per_marker_all_dist)
    print('P95 P95: ', np.percentile(p95_per_marker_all_dist, 95))
    print('Mean P95: ', np.mean(p95_per_marker_all_dist))

    # median_per_trial_global = np.median(trial_site_median, axis=1)
    # median_per_trial = np.sqrt(median_per_trial_global)
    # print('Median trial: ', np.median(median_per_trial))
    #
    # rmse_global = np.sqrt(np.mean(trial_site))
    # print('Globalno RMSE, uproseceno prvo po svim frejmovima, dobijena onda srednja vrednost za svaki marker za svaki trial pa uproseceno po markerima i trailovima i korenovano: ', rmse_global)
    #
    # rmse_per_trial = np.sqrt(np.mean(trial_site, axis=1))
    # rmse_per_trial_global = np.mean(rmse_per_trial)
    # print('RMSE uproseceno po markerima, korenovano, pa uproseceno po trailovima: ', rmse_per_trial_global)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mean error")

    parser.add_argument(
        "--qpos_path",
        type=Path,
        help="qpos CSV"
    )

    args = parser.parse_args()

    qpos_path = args.qpos_path

    mean_err(qpos_path)
