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

    x_min = data.filter(regex=r'_X$').min().min()
    x_max = data.filter(regex=r'_X$').max().max()

    y_min = data.filter(regex=r'_Y$').min().min()
    y_max = data.filter(regex=r'_Y$').max().max()

    z_min = 0.0
    z_max = data.filter(regex=r'_Z$').max().max()
    z_max = z_max * 5/4



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
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    #ax.set_aspect("equal")
    ax.view_init(elev=0, azim=0)
    plt.show()

    for joint in joint_names:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = data[f'{joint}_X']
        y = data[f'{joint}_Y']
        z = data[f'{joint}_Z']
        ax.scatter(x, y, z)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=0, azim=0)
        plt.title(joint)
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
    print("Broj DOF:", model_mj.nv)
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

    site_names = [
        "l_elbow_pos",
        "r_elbow_pos",
        "l_ankle_pos",
        "r_ankle_pos",
        "l_knee_pos",
        "r_knee_pos",
    ]

    site_ids = [
        mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    ]

    N = len(site_ids)

    # Jacobian
    J_pos = [np.zeros((3, model_mj.nv)) for _ in range(N)]
    J_rot = [np.zeros((3, model_mj.nv)) for _ in range(N)]

    marker_ids = [
        data.columns.get_loc(name + '_X')
        for name in site_names
    ]

    print(marker_ids)

    def enforce_joint_limits(model, data):
        for j in range(model.njnt):
            if not model.jnt_limited[j]:
                continue

            jtype = model.jnt_type[j]
            qadr = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]

            if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)

    def ik_step_nsite(model, data, site_ids, targets, weights=None, step_size=0.2, damping=1e-3):
        mj.mj_forward(model, data)

        N = len(site_ids)
        if weights is None:
            weights = np.ones(N)

        e_list = []
        J_list = []

        for i, site_id in enumerate(site_ids):
            ei = targets[i] - data.site_xpos[site_id]
            ei *= weights[i]
            e_list.append(ei)

            # Jacobian
            mj.mj_jacSite(model, data, J_pos[i], J_rot[i], site_id)
            J_list.append(weights[i] * J_pos[i])

        e = np.concatenate(e_list, axis=0)  # (3N,)
        J = np.vstack(J_list)  # (3N, nv)

        # damped least squares
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(
            JJt + damping * np.eye(3 * N),
            e
        )

        mj.mj_integratePos(model, data.qpos, dq * step_size, 1)

        enforce_joint_limits(model, data)

        return np.linalg.norm(e)

    def solve_one_frame_nsite(model, data, site_ids, targets, weights=None, n_iter=30, tol=1e-4):
        for _ in range(n_iter):
            err = ik_step_nsite(
                model, data,
                site_ids, targets,
                weights
            )
            if err < tol:
                break

    data_mj.qpos[:] = 0
    mj.mj_forward(model_mj, data_mj)

    weights = [
        1.0,  # lakat
        1.0,
        10.0,  # stopalo
        10.0,
        2.0,  # koleno
        2.0,
    ]

    t = 0
    targets = [
        data.iloc[t, id:id+3].to_numpy()
        for id in marker_ids
    ]

    solve_one_frame_nsite(model_mj, data_mj, site_ids, targets, weights)

    print("Targets:", targets)
    print("Model :", data_mj.site_xpos[site_ids])


    T_test = 2000
    qpos_traj = np.zeros((T_test, model_mj.nq))

    data_mj.qpos[:] = 0
    mj.mj_forward(model_mj, data_mj)

    for t in range(T_test):
        print("Frejm", t)
        targets = [
            data.iloc[t, id:id + 3].to_numpy()
            for id in marker_ids
        ]

        solve_one_frame_nsite(model_mj, data_mj, site_ids, targets, weights)
        qpos_traj[t] = data_mj.qpos.copy()

    print("Targets:", targets)
    print("Model :", data_mj.site_xpos[site_ids])

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
