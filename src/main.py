import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
from scipy.optimize import least_squares
from mujoco.glfw import glfw
import os
from utils import read_data, get_names


def main():
    data_path = "data/03_1_1_pos.tsv"
    separator = "\t"
    header_row = 5
    data_start = 8
    data = read_data(data_path, separator, data_start, header_row)
    print(data)
    # print(data.iloc[1, :])

    joint_names = get_names(data)
    # print(joint_names)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for joint in joint_names:
        x = data[f'{joint}_X'][900]
        y = data[f'{joint}_Y'][900]
        z = data[f'{joint}_Z'][900]
        ax.scatter(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect("equal")
    ax.view_init(elev=0, azim=0)
    plt.show()

    offset_y = 600
    offset_x = 200
    for col in data.columns:
        if col.endswith("Y"):
            data.loc[:, col] = data.loc[:, col] - offset_y
        if col.endswith("X"):
            data.loc[:, col] = data.loc[:, col] - offset_x
    # print(data)

    data = data / 1000.0
    # print(data)

    model_mj = mj.MjModel.from_xml_path("human_marina.xml")
    data_mj = mj.MjData(model_mj)
    mj.mj_forward(model_mj, data_mj)

    print("Broj qpos:", model_mj.nq)
    print("Broj site-ova:", model_mj.nsite)
    for j in range(model_mj.njnt):
        joint = model_mj.jnt(j)
        print(j, joint.name, joint.type, joint.qposadr)

    frame_idx = 0
    site = "r_knee_pos"
    marker_pos = data.loc[frame_idx, [site + "_X", site + "_Y", site + "_Z"]].to_numpy()
    print("Marker pos (mocap):", marker_pos)

    site_id = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, site)
    site_pos_sim = data_mj.site_xpos[site_id]
    print("Site pos (sim):", site_pos_sim)

    # ==============================================
    markers = ['l_shoulder_pos', 'r_shoulder_pos', 'l_hip_pos', 'r_hip_pos', 'l_knee_pos', 'r_knee_pos']

    y_list = []
    z_list = []
    for marker in markers:
        frame_idx = 0
        marker_pos = data.loc[frame_idx, [marker + "_X", marker + "_Y", marker + "_Z"]].to_numpy()
        print("Marker pos (mocap):", marker, marker_pos)
        site_id = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, marker)
        site_pos_sim = data_mj.site_xpos[site_id]
        print("Site pos (sim):", marker, site_pos_sim)

        y_scale = site_pos_sim[1] / marker_pos[1]
        y_list.append(y_scale)

        z_scale = site_pos_sim[2] / marker_pos[2]
        z_list.append(z_scale)

    y_res = np.mean(y_list)
    z_res = np.mean(z_list)

    # print(y_res, z_res)

    for col in data.columns:
        if col.endswith("Y"):
            data.loc[:, col] = data.loc[:, col] * y_res
        if col.endswith("Z"):
            data.loc[:, col] = data.loc[:, col] * z_res

    ''' 
        for marker in markers:
            frame_idx = 0
            marker_pos = data.loc[frame_idx, [marker + "_X", marker + "_Y", marker + "_Z"]].to_numpy()
            print("Marker pos (mocap):", marker, marker_pos)
            site_id = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, marker)
            site_pos_sim = data_mj.site_xpos[site_id]
            print("Site pos (sim):", marker, site_pos_sim)
    '''

    site_shoulder = "l_shoulder_pos"
    site_ankle = "r_ankle_pos"
    site_hip = "r_hip_pos"

    site_id_sh = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, site_shoulder)
    site_id_an = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, site_ankle)
    site_id_hi = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, site_hip)

    # Jacobian
    Jp_sh = np.zeros((3, model_mj.nv))
    Jr_sh = np.zeros((3, model_mj.nv))

    Jp_an = np.zeros((3, model_mj.nv))
    Jr_an = np.zeros((3, model_mj.nv))

    Jp_hi = np.zeros((3, model_mj.nv))
    Jr_hi = np.zeros((3, model_mj.nv))

    marker_id_sh = data.columns.get_loc(site_shoulder + '_X')
    marker_id_an = data.columns.get_loc(site_ankle + '_X')
    marker_id_hi = data.columns.get_loc(site_hip + '_X')
    print(marker_id_sh, marker_id_an, marker_id_hi)


    def mocap_l_shoulder(t):
        return data.iloc[t, marker_id_sh:marker_id_sh + 3].to_numpy()

    def mocap_r_ankle(t):
        return data.iloc[t, marker_id_an:marker_id_an + 3].to_numpy()

    def mocap_r_hip(t):
        return data.iloc[t, marker_id_hi:marker_id_hi + 3].to_numpy()


    def ik_step_multi(model, data, target_sh, target_an, target_hi, step_size=0.5, damping=1e-4):
        mj.mj_forward(model, data)

        e_sh = target_sh - data.site_xpos[site_id_sh]
        e_an = target_an - data.site_xpos[site_id_an]
        e_hi = target_hi - data.site_xpos[site_id_hi]

        # Jacobian
        mj.mj_jacSite(model, data, Jp_sh, Jr_sh, site_id_sh)
        mj.mj_jacSite(model, data, Jp_an, Jr_an, site_id_an)
        mj.mj_jacSite(model, data, Jp_hi, Jr_hi, site_id_hi)

        e = np.concatenate([e_sh, e_an, e_hi], axis=0)
        J = np.vstack([Jp_sh, Jp_an, Jp_hi])

        # Damped least squares
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(
            JJt + damping * np.eye(9),
            e
        )

        mj.mj_integratePos(model, data.qpos, dq * step_size, 1)

        return np.linalg.norm(e)

    def solve_one_frame_multi(model, data, target_sh, target_an, target_hi, n_iter=20, tol=1e-4):
        for i in range(n_iter):
            err = ik_step_multi(model, data, target_sh, target_an, target_hi)
            if err < tol:
                break

    data_mj.qpos[:] = 0
    mj.mj_forward(model_mj, data_mj)

    t = 0
    target_sh = mocap_l_shoulder(t)
    target_an = mocap_r_ankle(t)
    target_hi = mocap_r_hip(t)

    solve_one_frame_multi(model_mj, data_mj, target_sh, target_an, target_hi)

    print("Shoulder target:", target_sh)
    print("Shoulder model :", data_mj.site_xpos[site_id_sh])

    print("Ankle target:", target_an)
    print("Ankle model :", data_mj.site_xpos[site_id_an])

    print("Hip target:", target_hi)
    print("Hip model :", data_mj.site_xpos[site_id_hi])

    T_test = 2000
    qpos_traj = np.zeros((T_test, model_mj.nq))

    data_mj.qpos[:] = 0
    mj.mj_forward(model_mj, data_mj)

    for t in range(T_test):
        print("Frejm", t)
        target_sh = mocap_l_shoulder(t)
        target_an = mocap_r_ankle(t)
        target_hi = mocap_r_hip(t)

        solve_one_frame_multi(model_mj, data_mj, target_sh, target_an, target_hi)
        qpos_traj[t] = data_mj.qpos.copy()

    print("Shoulder target:", target_sh)
    print("Shoulder model :", data_mj.site_xpos[site_id_sh])

    print("Ankle target:", target_an)
    print("Ankle model :", data_mj.site_xpos[site_id_an])

    print("Hip target:", target_hi)
    print("Hip model :", data_mj.site_xpos[site_id_hi])

    import time
    import mujoco.viewer

    dt = model_mj.opt.timestep
    data_mj = mj.MjData(model_mj)
    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        for t in range(len(qpos_traj)):
            data_mj.qpos[:] = qpos_traj[t]
            mj.mj_forward(model_mj, data_mj)

            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
