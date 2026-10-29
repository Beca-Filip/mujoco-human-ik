# qpos_plots.py
"""
This module provides utilities for loading, processing, and visualizing
joint kinematics (qpos) data.

"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


COM = 2  # root z component
T_START = 10  # stabile frame
FS = 300    # Hz
HIP_LIMIT_UP = 100   # limits from MuJoCo xml models in degrees
HIP_LIMIT_LOW = -30
KNEE_LIMIT_UP = 0
KNEE_LIMIT_LOW = -150
ANKLE_LIMIT_UP = 30
ANKLE_LIMIT_LOW = -20


def normalize(signal):
    signal = np.array(signal)

    t_original = np.linspace(0, 1, len(signal))

    t_new = np.linspace(0, 1, 101)

    f = interp1d(t_original, signal, kind='linear')

    signal_resampled = f(t_new)

    return np.asarray(signal_resampled)


def qpos_plot(qpos_path_dir: Path):
    """
    Processes and visualizes joint angle trajectories from multiple qpos CSV files.

    This function:
    - Loads all CSV files from a directory
    - Extracts hip, knee, and ankle joint angles
    - Detects movement phase based on center of mass (COM) trajectory
    - Aggregates statistics across trials (mean, min, max, std)
    - Plots joint angle envelopes with anatomical limits

    Parameters
    ----------
    qpos_path_dir : Path
        Path to directory containing CSV files with qpos data.
        Each CSV must include columns:
        - 'left_hip_z', 'right_hip_z'
        - 'left_knee_z', 'right_knee_z'
        - 'left_ankle_z', 'right_ankle_z'
        - COM position as column index COM (default index 2)

    Notes:
    - Assumes sampling frequency of 300 Hz (FS constant)
    - Assumes consistent column naming across all CSV files
    - Movement segmentation is based on COM peak detection
    - Output angles are converted to degrees before plotting
    """

    csv_files = list(qpos_path_dir.glob("*.csv"))
    n_files = len(csv_files)
    #n_files = 65
    # print(n_files) #=134
    hip = np.zeros([101, n_files])
    knee = np.zeros([101, n_files])
    ankle = np.zeros([101, n_files])
    i = 0

    for file in sorted(os.listdir(qpos_path_dir)):
        if i > n_files-1:
            continue
        if not file.endswith(".csv"):
            continue

        qpos_csv_path = os.path.join(qpos_path_dir, file)
        print(f"Processing: {file}")

        qpos = pd.read_csv(qpos_csv_path)
        T = qpos.shape[0]
        time_vector = np.arange(0, T)

        l_hip = qpos.loc[:, 'left_hip_z']
        r_hip = qpos.loc[:, 'right_hip_z']
        l_knee = qpos.loc[:, 'left_knee_z']
        r_knee = qpos.loc[:, 'right_knee_z']
        l_ankle = qpos.loc[:, 'left_ankle_z']
        r_ankle = qpos.loc[:, 'right_ankle_z']

        # plt.figure()
        # plt.plot(time_vector, l_hip, label='l_hip')
        # plt.plot(time_vector, r_hip, label='r_hip')
        # plt.plot(time_vector, l_knee, label='l_knee')
        # plt.plot(time_vector, r_knee, label='r_knee')
        # plt.plot(time_vector, l_ankle, label='l_ankle')
        # plt.plot(time_vector, r_ankle, label='r_ankle')
        # plt.legend()
        # plt.title('Joint angles')
        # plt.xlabel('frames')
        # plt.show()

        com = qpos.iloc[:, COM]

        stable_start_frame = T_START
        start_height = com[stable_start_frame]
        z = com[stable_start_frame:].to_numpy()

        # max
        z_max = max(z)
        z_max_id = np.argmax(z)
        z_smooth = savgol_filter(z, window_length=11, polyorder=3)

        # min
        min_before_max = min(z_smooth[:z_max_id])
        min_after_max = min(z_smooth[z_max_id:])
        z_min_before_id = np.where(z_smooth == min_before_max)[0][0]
        z_min_after_id = np.where(z_smooth == min_after_max)[0][0]

        # z_before_max = z_smooth[:z_max_id]
        # dz_before_max = np.gradient(z_before_max)
        #
        # signs_b_max = np.sign(dz_before_max)
        # sign_changes_b_max = np.diff(signs_b_max)
        # indices_b_max = np.where(sign_changes_b_max != 0)[0]
        # if len(indices_b_max) > 0:
        #     last_change_idx_b_max = indices_b_max[-1]
        # else:
        #     last_change_idx_b_max = 0
        # min_before_max = z_before_max[last_change_idx_b_max]
        # z_min_before_id = np.where(z_smooth == min_before_max)[0][0]
        #
        # z_after_max = z_smooth[z_max_id:]
        # dz_after_max = np.gradient(z_after_max)
        #
        # signs_a_max = np.sign(dz_after_max)
        # sign_changes_a_max = np.diff(signs_a_max)
        # indices_a_max = np.where(sign_changes_a_max != 0)[0]
        # if len(indices_a_max) > 0:
        #     last_change_idx_a_max = indices_b_max[0]
        # else:
        #     last_change_idx_a_max = 0
        # min_after_max = z_after_max[last_change_idx_a_max]
        # z_min_after_id = np.where(z_smooth == min_after_max)[0][0]

        # start
        z_before = z_smooth[:z_min_before_id]
        dz_before = np.gradient(z_before)

        signs_b = np.sign(dz_before)
        sign_changes_b = np.diff(signs_b)
        indices_b = np.where(sign_changes_b != 0)[0]
        if len(indices_b) > 0:
            last_change_idx = indices_b[-1]
        else:
            last_change_idx = 0
        start_val = z_before[last_change_idx]
        start_idx = np.where(z_smooth == start_val)[0][0]

        # end
        z_after = z_smooth[z_min_after_id:]
        if z_min_after_id == (len(z_smooth) - 1):
            first_change_idx = 0
        else:

            start_val_90 = 0.9 * start_val
            start_val_110 = 1.1 * start_val
            indices_a = np.where((z_after >= start_val_90) & (z_after <= start_val_110))[0]
            if len(indices_a) > 0:
                first_change_idx = indices_a[0]
            else:
                first_change_idx = z_after.shape[0] - 1

            # dz_after = np.gradient(z_after)
            # signs_a = np.sign(dz_after)
            # sign_changes_a = np.diff(signs_a)
            # indices_a = np.where(sign_changes_a != 0)[0]
            # if len(indices_a) > 0:
            #     first_change_idx = indices_a[0]
            # else:
            #     first_change_idx = dz_after.shape[0] - 1

        end_val = z_after[first_change_idx]
        end_idx = np.where(z_smooth == end_val)[0][0]

        if z_max_id + FS < len(z_smooth):
            end_idx = z_max_id + FS
        else:
            end_idx = len(z_smooth) -1

        # plt.figure()
        # plt.plot(z_smooth, label="z")
        # plt.axvline(start_idx, color='g', label='start')
        # plt.axvline(z_min_before_id, color='r', label='min')
        # plt.axvline(z_min_after_id, color='r', label='min')
        # plt.axvline(end_idx, color='b', label='end')
        # plt.axvline(z_max_id, color='r', label='max')
        # plt.legend()
        # plt.title('com')
        # plt.show()


        # qpos segmentation
        start_idx = start_idx + stable_start_frame
        end_idx = end_idx + stable_start_frame
        seg = np.arange(start_idx, end_idx)
        l_hip_seg = l_hip[seg].to_numpy()
        l_knee_seg = l_knee[seg].to_numpy()
        l_ankle_seg = l_ankle[seg].to_numpy()
        r_hip_seg = r_hip[seg].to_numpy()
        r_knee_seg = r_knee[seg].to_numpy()
        r_ankle_seg = r_ankle[seg].to_numpy()
        time_vector_seg = time_vector[seg]

        l_hip_seg_norm = normalize(l_hip_seg)
        r_hip_seg_norm = normalize(r_hip_seg)
        l_knee_seg_norm = normalize(l_knee_seg)
        r_knee_seg_norm = normalize(r_knee_seg)
        l_ankle_seg_norm = normalize(l_ankle_seg)
        r_ankle_seg_norm = normalize(r_ankle_seg)
        n_samples = len(l_hip_seg_norm)
        time_vector_norm = np.linspace(0, 1, n_samples)

        hip[:,i] = l_hip_seg_norm
        knee[:,i] = l_knee_seg_norm
        ankle[:,i] = l_ankle_seg_norm
        i = i+1

        # plt.figure()
        # plt.plot(time_vector_norm, l_hip_seg_norm, label='l_hip')
        # plt.plot(time_vector_norm, r_hip_seg_norm, label='r_hip')
        # plt.plot(time_vector_norm, l_knee_seg_norm, label='l_knee')
        # plt.plot(time_vector_norm, r_knee_seg_norm, label='r_knee')
        # plt.plot(time_vector_norm, l_ankle_seg_norm, label='l_ankle')
        # plt.plot(time_vector_norm, r_ankle_seg_norm, label='r_ankle')
        # plt.legend()
        # plt.title('Joint angles')
        # plt.xlabel('normalized time')
        # plt.show()

    hip = np.degrees(hip)
    knee = np.degrees(knee)
    ankle = np.degrees(ankle)

    mean_hip = np.mean(hip, axis=1)
    max_hip = np.max(hip, axis=1)
    min_hip = np.min(hip, axis=1)
    std_hip = np.std(hip, axis=1)

    mean_knee = np.mean(knee, axis=1)
    max_knee = np.max(knee, axis=1)
    min_knee = np.min(knee, axis=1)
    std_knee = np.std(knee, axis=1)

    mean_ankle = np.mean(ankle, axis=1)
    max_ankle = np.max(ankle, axis=1)
    min_ankle = np.min(ankle, axis=1)
    std_ankle = np.std(ankle, axis=1)

    time_samples = mean_hip.shape[0]
    time_ax = np.linspace(0, 1, time_samples)

    upper_hip = mean_hip + 2 * std_hip
    lower_hip = mean_hip - 2 * std_hip
    upper_knee = mean_knee + 2 * std_knee
    lower_knee = mean_knee - 2 * std_knee
    upper_ankle = mean_ankle + 2 * std_ankle
    lower_ankle = mean_ankle - 2 * std_ankle

    upper_hip = max_hip
    lower_hip = min_hip
    upper_knee = max_knee
    lower_knee = min_knee
    upper_ankle = max_ankle
    lower_ankle = min_ankle

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)

    # HIP
    ax[0].plot(time_ax, mean_hip, label='hip', color='#90be6d')
    ax[0].axhline(HIP_LIMIT_UP, label='upper limit', color='red', alpha=0.7, linestyle='--')
    ax[0].axhline(HIP_LIMIT_LOW, label='lower limit', color='blue', alpha=0.7, linestyle='--')
    ax[0].fill_between(time_ax, lower_hip, upper_hip, alpha=0.3, color='#90be6d')
    # ax[0].plot(time_ax, max_hip, color='#90be6d', linestyle=':')
    # ax[0].plot(time_ax, min_hip, color='#90be6d', linestyle=':')
    ax[0].legend(loc='lower right')
    ax[0].set_xlim(0, 1)

    # KNEE
    ax[1].plot(time_ax, mean_knee, label='knee', color='#f4a261')
    ax[1].axhline(KNEE_LIMIT_UP, label='upper limit', color='red', alpha=0.7, linestyle='--')
    ax[1].axhline(KNEE_LIMIT_LOW, label='lower limit', color='blue', alpha=0.7, linestyle='--')
    ax[1].fill_between(time_ax, lower_knee, upper_knee, alpha=0.3, color='#f4a261')
    # ax[1].plot(time_ax, max_knee, color='#f4a261', linestyle=':')
    # ax[1].plot(time_ax, min_knee, color='#f4a261', linestyle=':')
    ax[1].set_ylabel("Angles [deg]")
    ax[1].legend(loc='lower right')
    ax[1].set_xlim(0, 1)

    # ANKLE
    ax[2].plot(time_ax, mean_ankle, label='ankle', color='#8ecae6')
    ax[2].axhline(ANKLE_LIMIT_UP, label='upper limit', color='red', alpha=0.7, linestyle='--')
    ax[2].axhline(ANKLE_LIMIT_LOW, label='lower limit', color='blue', alpha=0.7, linestyle='--')
    ax[2].fill_between(time_ax, lower_ankle, upper_ankle, alpha=0.3, color='#8ecae6')
    # ax[2].plot(time_ax, max_ankle, color='#8ecae6', linestyle=':')
    # ax[2].plot(time_ax, min_ankle, color='#8ecae6', linestyle=':')
    ax[2].set_xlabel("Normalized time")
    ax[2].legend(loc='lower right')
    ax[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qpos plots")

    parser.add_argument(
        "--qpos_path",
        type=Path,
        help="qpos CSV"
    )

    args = parser.parse_args()

    qpos_path = args.qpos_path

    qpos_plot(qpos_path)



