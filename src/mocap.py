# mocap.py
import numpy as np
import pandas as pd
import mujoco as mj
from utils import read_data

# ============================================================
# --------------- DATA LOADING & PREPROCESSING ---------------
# ============================================================


def load_mocap_data(
    data_path: str,
    separator: str = "\t",
    header_row: int = 5,
    data_start_row: int = 8,
) -> pd.DataFrame:
    """
    Load motion capture data from a TSV file.
    """
    return read_data(data_path, separator, data_start_row, header_row)


def apply_offsets(
    data: pd.DataFrame,
    offset_x: float = 200.0,
    offset_y: float = 600.0
) -> pd.DataFrame:
    """
    Apply XY offsets to recenter mocap coordinates.
    """
    for col in data.columns:
        if col.endswith("_X"):
            data.loc[:, col] -= offset_x
        elif col.endswith("_Y"):
            data.loc[:, col] -= offset_y
    return data


def mm_to_meters(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert millimeters to meters.
    """
    return data / 1000.0


def compute_axis_scaling_factors(
    model: mj.MjModel,
    data: mj.MjData,
    mocap_data: pd.DataFrame,
    marker_names: list[str],
    frame_index: int = 0
) -> tuple[float, float]:
    """
    Compute average Y and Z scaling factors by comparing mocap marker
    positions with corresponding MuJoCo site positions.
    """

    y_scales = []
    z_scales = []

    mj.mj_forward(model, data)

    for marker in marker_names:
        # Mocap marker position
        marker_pos = mocap_data.loc[
            frame_index,
            [f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]
        ].to_numpy()
        print("Marker pos (mocap):", marker, marker_pos)

        # MuJoCo site position
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, marker)
        site_pos = data.site_xpos[site_id]
        print("Site pos (sim):", marker, site_pos)

        # Compute per-axis scaling
        y_scales.append(site_pos[1] / marker_pos[1])
        z_scales.append(site_pos[2] / marker_pos[2])

    return np.mean(y_scales), np.mean(z_scales)


def apply_axis_scaling(
    mocap_data: pd.DataFrame,
    y_scale: float,
    z_scale: float
) -> pd.DataFrame:
    """
    Scale Y and Z axes of mocap data to match MuJoCo model proportions.
    """

    for col_name in mocap_data.columns:
        if col_name.endswith("_Y"):
            mocap_data.loc[:, col_name] *= y_scale
        elif col_name.endswith("_Z"):
            mocap_data.loc[:, col_name] *= z_scale

    return mocap_data
