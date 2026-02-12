import numpy
from mocap import *
from utils import *
from visualization import *
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

MOCAP_GLITCH_DISTANCE_THRESHOLD = 0.05 # in meters (corresponding to marker velocities of MOCAP_GLITCH_DISTANCE_THRESHOLD / dt)
MIN_DETECTED_GLITCH_DURATION = 7 # in number of samples

def is_repeating(a: np.ndarray, seq_len: int) -> np.ndarray[np.bool]:
    flag = np.ones_like(a, dtype=np.bool)

    for i in range(seq_len // 2, len(a) - seq_len // 2):
        x = a[i]
        for j in range(1, seq_len // 2 + 1):
            if x != a[i-j] or x != a[i+j]:
                flag[i] = False
                break
    flag[:seq_len // 2 + 1] = False
    flag[-seq_len // 2:] = False
    return flag

# ---------------- Load mocap data ----------------
mocap_data = load_mocap_data(data_path="data/03_1_1_pos.tsv")
joint_names = get_names(mocap_data)

n_markers = int(mocap_data.shape[1] / 3)
n_samples = int(mocap_data.shape[0])
dt = 1. / 300.
time_vector = np.arange(0, n_samples * dt, step=dt, )

# ---------------- Apply offsets ------------------
mocap_data = apply_offsets(mocap_data)

# ---------------- Convert mm to m ----------------
mocap_data = mm_to_meters(mocap_data)

# --------------- Plot MoCap data -----------------
# Compute shared axis limits once
x_min, x_max, y_min, y_max, z_min, z_max = compute_axis_limits(mocap_data)
x_limits = (x_min, x_max)
y_limits = (y_min, y_max)
z_limits = (z_min, z_max)

# # Non-clean data
# fig, ax = plt.subplots(figsize=(19.2, 10.8), nrows=3, ncols=1)
# for i in range(n_markers):
#     ax[0].plot(time_vector, mocap_data.iloc[:, 3*i], linestyle='--', label=mocap_data.columns[3*i])
#     ax[0].legend()

#     ax[1].plot(time_vector, mocap_data.iloc[:, 3*i+1], linestyle='--', label=mocap_data.columns[3*i+1])
#     ax[1].legend()

#     ax[2].plot(time_vector, mocap_data.iloc[:, 3*i+2], linestyle='--', label=mocap_data.columns[3*i+2])
#     ax[2].legend()


# Clean data
clean_mocap_data = mocap_data.copy(deep=True)
for col in clean_mocap_data.columns:

    data = clean_mocap_data[col].to_numpy()

    # corrupted_indices = data == 0
    # if not np.any(corrupted_indices):
    #     continue

    # Detect glitches as indices between sharp transitions
    diff_data = np.diff(data, n=1, axis=0)
    diff_data_bool = abs(diff_data) > MOCAP_GLITCH_DISTANCE_THRESHOLD
    
    if not np.any(diff_data_bool):
        continue

    diff_data_bool_cumsum = np.cumsum(diff_data_bool.astype(np.int64))
    diff_data_bool_1 = np.mod(diff_data_bool_cumsum, 2).astype(np.bool)
    diff_data_bool_2 = np.logical_not(diff_data_bool_1)

    diff_data_bool_1 = np.append(diff_data_bool_1, diff_data_bool_1[-1])
    diff_data_bool_2 = np.append(diff_data_bool_2, diff_data_bool_2[-1])

    # Detect glitches as indices with repeating values (occluded or untracked markers)
    n_consecutive = MIN_DETECTED_GLITCH_DURATION
    repeating_values_bool = is_repeating(data, seq_len=n_consecutive)
    
    # Find where they correlate the best and choose that as glitch indices
    corr_1 = np.sum(diff_data_bool_1 * repeating_values_bool)
    corr_2 = np.sum(diff_data_bool_2 * repeating_values_bool)

    if corr_1 > corr_2:
        corrupted_indices = diff_data_bool_1
    else:
        corrupted_indices = diff_data_bool_2

    if not np.any(corrupted_indices):
        continue
       
    non_corrupted_indices = np.logical_not(corrupted_indices)
    non_corrupted_data = data[non_corrupted_indices]
    non_corrupted_time_vecotr = time_vector[non_corrupted_indices]

    fill_values = (non_corrupted_data[0], non_corrupted_data[-1]) 
    interpolant = interp1d(non_corrupted_time_vecotr, non_corrupted_data, kind='cubic', fill_value=fill_values, bounds_error=False)
    clean_data = interpolant(time_vector)
    
    fig, ax = plt.subplots(figsize=(19.2, 10.8), nrows=4, ncols=1)
    ax[0].plot(time_vector, data, linestyle='--')
    ax[0].plot(time_vector, clean_data)
    
    ax[1].plot(time_vector[:-1], diff_data, linestyle='-.')
    
    ax[2].plot(time_vector, corrupted_indices, linestyle='-.')

    ax[3].plot(time_vector, repeating_values_bool, linestyle='-.')

    plt.show()

    # fill_values = (non_corrupted_data[0], non_corrupted_data[-1]) 
    # interpolant = interp1d(non_corrupted_time_vecotr, non_corrupted_data, kind='cubic', fill_value=fill_values, bounds_error=False)
    # clean_data = interpolant(time_vector)

    # fig, ax = plt.subplots(figsize=(19.2, 10.8), nrows=1, ncols=1)
    # ax.plot(time_vector, data, linestyle='--')
    # ax.plot(time_vector, clean_data)
    # plt.show()

    clean_mocap_data[col] = clean_data


# fig, ax = plt.subplots(figsize=(19.2, 10.8), nrows=3, ncols=1)
# for i in range(n_markers):
#     ax[0].plot(time_vector, clean_mocap_data.iloc[:, 3*i], linestyle='--', label=clean_mocap_data.columns[3*i])
#     ax[0].legend()

#     ax[1].plot(time_vector, clean_mocap_data.iloc[:, 3*i+1], linestyle='--', label=clean_mocap_data.columns[3*i+1])
#     ax[1].legend()

#     ax[2].plot(time_vector, clean_mocap_data.iloc[:, 3*i+2], linestyle='--', label=clean_mocap_data.columns[3*i+2])
#     ax[2].legend()

# plt.show()

print()