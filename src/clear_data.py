# clear_data.py
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

MOCAP_GLITCH_DISTANCE_THRESHOLD = 0.05  # in meters (corresponding to marker velocities of MOCAP_GLITCH_DISTANCE_THRESHOLD / dt)
MIN_DETECTED_GLITCH_DURATION = 7  # in number of samples
DT_DEFAULT = 1.0 / 300.0


# ============================================================
# -------------------- CLEAR DATA ----------------------------
# ============================================================


def plot_mocap_data_per_axes(time_vector, mocap_data):
    n_markers = int(mocap_data.shape[1] / 3)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), nrows=3, ncols=1)
    for i in range(n_markers):
        ax[0].plot(time_vector, mocap_data.iloc[:, 3*i], linestyle='--', label=mocap_data.columns[3*i])
        ax[0].legend()

        ax[1].plot(time_vector, mocap_data.iloc[:, 3*i+1], linestyle='--', label=mocap_data.columns[3*i+1])
        ax[1].legend()

        ax[2].plot(time_vector, mocap_data.iloc[:, 3*i+2], linestyle='--', label=mocap_data.columns[3*i+2])
        ax[2].legend()
        ax[0].set_title("Mocap data per axes")


def plot_cleaned_data(data, cleaned_data, time_vector, corrupted, title):
    fig, ax = plt.subplots(figsize=(19.2, 10.8), nrows=2, ncols=1)
    ax[0].plot(time_vector, data, linestyle='--', label='Original')
    ax[0].plot(time_vector, cleaned_data, label='Cleaned')
    ax[0].legend()
    ax[1].plot(time_vector, corrupted, linestyle='-.', label='Corrupted')
    ax[0].set_title(title)
    plt.show()

def is_repeating(a: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Detect indices where the value is repeated for a given window length.
    This is typically caused by occluded or untracked mocap markers.
    """
    flag = np.ones_like(a, dtype=bool)
    half = seq_len // 2

    for i in range(half, len(a) - half):
        x = a[i]
        for j in range(1, half + 1):
            if x != a[i - j] or x != a[i + j]:
                flag[i] = False
                break

    # Edge samples cannot be reliably evaluated
    flag[:half + 1] = False
    flag[-half:] = False

    return flag


def expand_boolean_mask(mask: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Expand True values in a boolean mask by k samples on both sides.
    Useful to slightly enlarge detected glitch regions.
    """
    out = mask.copy()
    ones = np.where(mask == 1)[0]

    for i in ones:
        start = max(0, i - k)
        end = min(len(mask), i + k + 1)
        out[start:end] = 1

    return out


def detect_glitches_1d(
    data: np.ndarray,
    distance_threshold: float,
    min_repeating_len: int
) -> np.ndarray:
    """
    Detect corrupted indices in a 1D mocap signal.

    Glitches are detected using:
    1) Large frame-to-frame jumps
    2) Repeating (flat) values

    The method chooses the segmentation ("inside" vs "outside" jumps)
    that best correlates with repeating values.
    """
    diff_data = np.diff(data)
    diff_data_bool = np.abs(diff_data) > distance_threshold
    print(sum(diff_data_bool))

    if not np.any(diff_data_bool):
        return np.zeros(len(data), dtype=int)

    # Segment signal into alternating regions (inside / outside jumps)
    cumsum = np.cumsum(diff_data_bool.astype(np.int64))
    region_1 = (cumsum % 2).astype(bool)
    region_2 = np.logical_not(region_1)

    # Match signal length
    region_1 = np.append(region_1, region_1[-1])
    region_2 = np.append(region_2, region_2[-1])

    # Detect repeating values
    repeating_mask = is_repeating(data, min_repeating_len)

    # Choose region that best correlates with repeating values
    corr_1 = np.sum(region_1 * repeating_mask)
    corr_2 = np.sum(region_2 * repeating_mask)

    corrupted = region_1 if corr_1 > corr_2 else region_2

    return expand_boolean_mask(corrupted.astype(int), k=2)


def combine_xyz_glitches(mask_x, mask_y, mask_z) -> np.ndarray:
    """
    Combine X/Y/Z glitch masks.
    A sample is considered corrupted if at least 2 axes are corrupted.
    """
    min_len = min(len(mask_x), len(mask_y), len(mask_z))
    combined = (mask_x[:min_len] + mask_y[:min_len] + mask_z[:min_len]) >= 2
    return combined.astype(int)


def interpolate_corrupted_segments(
    time_vector: np.ndarray,
    data: np.ndarray,
    corrupted_mask: np.ndarray
) -> np.ndarray:
    """
    Interpolate corrupted samples using PCHIP interpolation.
    """
    clean_data = data.copy()
    non_corrupted_mask = np.logical_not(corrupted_mask)

    if not np.any(non_corrupted_mask):
        return clean_data

    non_corrupted_time_vector = time_vector[non_corrupted_mask]
    non_corrupted_data = data[non_corrupted_mask]

    if corrupted_mask[0] == 0:
        interpolator = PchipInterpolator(non_corrupted_time_vector, non_corrupted_data, extrapolate=True)
        clean_data = interpolator(time_vector)
    else:
        # Fallback for leading corrupted samples
        clean_data[time_vector < non_corrupted_time_vector[0]] = non_corrupted_data[1]
        clean_data[time_vector > non_corrupted_time_vector[-1]] = non_corrupted_data[-1]

    return clean_data


def clean_mocap_data(mocap_data, dt: float = DT_DEFAULT):
    """
    Clean mocap marker trajectories by detecting and interpolating glitches.
    """
    n_markers = int(mocap_data.shape[1] / 3)
    n_samples = int(mocap_data.shape[0])

    time_vector = np.arange(0, n_samples * dt, step=dt)

    plot_mocap_data_per_axes(time_vector, mocap_data)
    cleaned = mocap_data.copy(deep=True)

    for col in mocap_data.columns:
        if not col.endswith("_X"):
            continue

        idx = mocap_data.columns.get_loc(col)
        marker_name = col[:-2]

        data_X = mocap_data.loc[:, marker_name + '_X'].to_numpy()
        data_Y = mocap_data.loc[:, marker_name + '_Y'].to_numpy()
        data_Z = mocap_data.loc[:, marker_name + '_Z'].to_numpy()

        mask_X = detect_glitches_1d(
            data_X,
            MOCAP_GLITCH_DISTANCE_THRESHOLD,
            MIN_DETECTED_GLITCH_DURATION
        )
        mask_Y = detect_glitches_1d(
            data_Y,
            MOCAP_GLITCH_DISTANCE_THRESHOLD,
            MIN_DETECTED_GLITCH_DURATION
        )
        mask_Z = detect_glitches_1d(
            data_Z,
            MOCAP_GLITCH_DISTANCE_THRESHOLD,
            MIN_DETECTED_GLITCH_DURATION
        )

        corrupted = combine_xyz_glitches(mask_X, mask_Y, mask_Z)

        if not np.any(corrupted):
            continue

        cleaned.iloc[:, idx] = interpolate_corrupted_segments(time_vector, data_X, corrupted)
        cleaned.iloc[:, idx + 1] = interpolate_corrupted_segments(time_vector, data_Y, corrupted)
        cleaned.iloc[:, idx + 2] = interpolate_corrupted_segments(time_vector, data_Z, corrupted)

        plot_cleaned_data(data_X, cleaned.iloc[:, idx], time_vector, corrupted, (marker_name+'_X'))
        plot_cleaned_data(data_Y, cleaned.iloc[:, idx+1], time_vector, corrupted, (marker_name+'_Y'))
        plot_cleaned_data(data_Z, cleaned.iloc[:, idx+2], time_vector, corrupted, (marker_name+'_Z'))

    plot_mocap_data_per_axes(time_vector, cleaned)
    return cleaned

