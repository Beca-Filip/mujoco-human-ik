import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw
import os
from utils import read_data, get_names


def main():
    data_path = "03_1_1_pos.tsv"
    separator = "\t"
    header_row = 5
    data_start = 8
    data = read_data(data_path, separator, data_start, header_row)
    print(data)
    #print(data.iloc[1, :])

    joint_names = get_names(data)
    #print(joint_names)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for joint in joint_names:
        x = data[f'{joint}_X'][1000]
        y = data[f'{joint}_Y'][1000]
        z = data[f'{joint}_Z'][1000]
        ax.scatter(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect("equal")
    ax.view_init(elev=0, azim=0)
    plt.show()

# =======================================
    # novi deo
    shift_y = 600
    shift_x = 200
    for col in data.columns:
        if col.endswith("Y"):
            data.loc[:, col] = data.loc[:, col] - shift_y
        if col.endswith("X"):
            data.loc[:, col] = data.loc[:, col] - shift_x
    #print(data)

    data = data/1000.0
    #print(data)

    model_mj = mj.MjModel.from_xml_path("human_marina.xml")
    data_mj = mj.MjData(model_mj)
    mj.mj_forward(model_mj, data_mj)

    frame_idx = 0
    site = "r_knee_pos"
    marker_pos = data.loc[frame_idx, [site+"_X", site+"_Y", site+"_Z"]].to_numpy()
    print("Marker pos (mocap):", marker_pos)

    site_id = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, site)
    site_pos_sim = data_mj.site_xpos[site_id]
    print("Site pos (sim):", site_pos_sim)
    for i in range(model_mj.nsite):
        name = mj.mj_id2name(model_mj, mj.mjtObj.mjOBJ_SITE, i)
        print(i, name)



if __name__ == "__main__":
    main()

