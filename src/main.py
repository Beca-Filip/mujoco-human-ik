import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
from scipy.optimize import least_squares
from mujoco.glfw import glfw
import os
from utils import read_data, get_names, site_position


def main():
    data_path = "data/03_1_1_pos.tsv"
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
    #print(data)

    data = data/1000.0
    #print(data)

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
    marker_pos = data.loc[frame_idx, [site+"_X", site+"_Y", site+"_Z"]].to_numpy()
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

    #print(y_res, z_res)

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

    site_name = "l_shoulder_pos"
    site_id = mj.mj_name2id(model_mj, mj.mjtObj.mjOBJ_SITE, site_name)
    # print("Site id:", site_id)  ---13

    marker_id = data.columns.get_loc(site_name + '_X')
    print(marker_id)

    '''
    torso_marker_ids = [data.columns.get_loc('l_shoulder_pos_X'), data.columns.get_loc('r_shoulder_pos_X')]
    def estimate_root_from_markers(t):
        offset_z_root = -0.06
        pts = []
        for id in torso_marker_ids:
            pts.append(data.iloc[t, id:id + 3].to_numpy())
        mean_shoulder = np.mean(pts, axis=0)
        mean_shoulder[2] = mean_shoulder[2] + offset_z_root
        return mean_shoulder
    
    body_id = model_mj.body("torso").id
    root_pos = data_mj.xpos[body_id]

    print(root_pos)
    print(estimate_root_from_markers(0))
    '''

    def mocap_l_shoulder(t):
        return data.iloc[t, marker_id:marker_id + 3].to_numpy()

    def error(qpos, target):
        return site_position(model_mj, qpos, site_id) - target

    def numeric_jacobian(qpos, eps=1e-6):
        base = site_position(model_mj, qpos, site_id)
        J = np.zeros((3, model_mj.nq))

        for i in range(model_mj.nq):
            dq = np.zeros_like(qpos)
            dq[i] = eps

            plus = site_position(model_mj, qpos + dq, site_id)
            minus = site_position(model_mj, qpos - dq, site_id)

            J[:, i] = (plus - minus) / (2 * eps)

        return J

    def ik_step(qpos, target, alpha=0.5):
        e = error(qpos, target)
        J = numeric_jacobian(qpos)

        dq, *_ = np.linalg.lstsq(J, -e, rcond=None)


        return qpos + alpha * dq

    def solve_one_frame(target, q_init, n_iter=10):
        q = q_init.copy()
        for _ in range(n_iter):
            q = ik_step(q, target)
        return q

    q0 = np.zeros(model_mj.nq)
    t = 0

    target = mocap_l_shoulder(t)
    q_sol = solve_one_frame(target, q0)

    print("Target:", target)
    print("Model :", site_position(model_mj, q_sol, site_id))

    T = data.shape[0]
    T_test = 50
    qpos_traj = np.zeros((T, model_mj.nq))
    q_current = np.zeros(model_mj.nq)
    for t in range(T_test):
        print('Frejm', t)
        target = mocap_l_shoulder(t)
        q_current = solve_one_frame(target, q_current, n_iter=10)
        qpos_traj[t] = q_current

    print("Target:", target)
    print("Model :", site_position(model_mj, q_current, site_id))

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

