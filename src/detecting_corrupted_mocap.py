# detecting_corrupted_mocap.py
import numpy as np
import pandas as pd
from collections import defaultdict


def get_marker_array(df, marker_name):
    return df[[f"{marker_name}_X", f"{marker_name}_Y", f"{marker_name}_Z"]].values

def filter_markers_from_pairs(df: pd.DataFrame, dev_threshold=0.3,
                              length_threshold=None,
                              expected_lengths=None):

    """
    Detect corrupted MoCap data by checking deviations in distances between marker pairs on the same segment.
    """

    pairs = [("l_metatarsal_pos", "l_ankle_pos"),
        ("l_ankle_pos", "l_knee_pos"),
        ("l_knee_pos", "l_hip_pos"),
        ("r_metatarsal_pos", "r_ankle_pos"),
        ("r_ankle_pos", "r_knee_pos"),
        ("r_knee_pos", "r_hip_pos"),
        ("l_wrist_pos", "l_elbow_pos"),
        ("l_elbow_pos", "l_shoulder_pos"),
        ("r_wrist_pos", "r_elbow_pos"),
        ("r_elbow_pos", "r_shoulder_pos"),
        ("l_hip_pos", "r_hip_pos"),
        ("l_shoulder_pos", "r_shoulder_pos"),]

    # problems = []

    problems = defaultdict(lambda: {
        "frames": [],
        "max_deviation": 0.0,
        "median": 0.0
    })

    for (m1, m2) in pairs:
        p1 = get_marker_array(df, m1)
        p2 = get_marker_array(df, m2)

        dist = np.linalg.norm(p1 - p2, axis=1)

        median_dist = np.median(dist)

        dev = np.abs(dist - median_dist) / (median_dist + 1e-8)
        bad_idx = np.where(dev > dev_threshold)[0]

        if expected_lengths is not None and (m1, m2) in expected_lengths:
            expected = expected_lengths[(m1, m2)]
            length_dev = np.abs(median_dist - expected) / expected
            bad_idx = np.where(length_dev > length_threshold)[0]

        # for i in bad_idx:
        #     problems.append({
        #         "frame": int(i),
        #         "pair": (m1, m2),
        #         "distance": float(dist[i]),
        #         "median": float(median_dist),
        #         "deviation": float(dev[i])
        #     })

        if len(bad_idx) == 0:
            continue

        pair = (m1, m2)

        for i in bad_idx:
            problems[pair]["frames"].append(int(i))

            current_dev = float(dev[i])
            if current_dev > problems[pair]["max_deviation"]:
                problems[pair]["max_deviation"] = current_dev

        problems[pair]["median"] = float(median_dist)

    corr_flag = len(problems) > 0

    return problems, corr_flag
