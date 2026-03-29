# id.py
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import argparse
from pathlib import Path

# ============================================================
# -------------------- INVERSE DYNAMICS ----------------------
# ============================================================


def lowpass_filter(data: np.ndarray, fs: float, wn: float = 10.0, N: int = 4):
    nyq = 0.5 * fs
    wn_norm = wn / nyq
    b, a = butter(N, wn_norm, btype='low')
    return filtfilt(b, a, data, axis=0)

# def bandpass_filter(data: np.ndarray, fs: float, N: int = 3):
#     nyq = 0.5 * fs
#     lowcut = 0.5
#     highcut = 12
#     low = lowcut / nyq
#     high = highcut / nyq
#
#     b, a = butter(N, [low, high], btype='band')
#
#     return filtfilt(b, a, data)


def fourier_transform(signal, fs, signal_name):
    N = len(signal)
    signal_detrended = signal - np.mean(signal)
    fft_signal = fft(signal_detrended)
    fft_freq = fftfreq(N, 1 / fs)
    positive = fft_freq >= 0

    # plt.figure()
    # plt.plot(fft_freq[positive], np.abs(fft_signal[positive]))
    # plt.xlabel("Frequencies (Hz)")
    # plt.ylabel("Amplitude")
    # plt.xlim([0, 50])
    # plt.title("FFT " + signal_name)
    # plt.show()


def inverse_dynamics(model_path: Path, qpos_path: Path, subj_trail: str, fs: float = 300.0, N: int = 3,
                     wn: float = 10.0, save_plot_flag: bool = False, gravity: float = 9.81):
    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    qpos = pd.read_csv(qpos_path)
    qpos = qpos.to_numpy()
    if subj_trail is None:
        subj_trail = str(qpos_path)[5:11]
    T = qpos.shape[0]
    nv = model.nv
    dt = 1/fs

    qvel = np.zeros((T, nv))
    qacc = np.zeros((T, nv))
    tau = np.zeros((T, nv))

    qpos_filtered = qpos.copy()

    qpos_filtered[:, 0:3] = lowpass_filter(qpos[:, 0:3], fs, wn, N)
    qpos_filtered[:, 7:] = lowpass_filter(qpos[:, 7:], fs, wn, N)

    for i in range(6, nv):
        #print(i)
        joint_id = model.dof_jntid[i]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        #print(joint_name)
        #
        # plt.figure()
        # plt.plot(np.arange(T), qpos[:, i + 1])
        # plt.title('qpos ' + joint_name)
        # plt.show()

        fourier_transform(qpos[:, i+1], fs, joint_name)

    for t in range(1, T-1):
        mj.mj_differentiatePos(model, qvel[t, :], 2*dt, qpos_filtered[t-1, :], qpos_filtered[t+1, :])

    qvel[0, :] = qvel[1, :]

    qvel_filtered = qvel.copy()

    qvel_filtered[:, 0:3] = lowpass_filter(qvel[:, 0:3], fs, wn, N)
    qvel_filtered[:, 6:] = lowpass_filter(qvel[:, 6:], fs, wn, N)

    qacc[1:T-1] = (qvel_filtered[2:] - qvel_filtered[:-2]) / (2*dt)

    qacc[0] = qacc[1]
    qacc[T-1] = qacc[T-2]

    total_grf_left = np.zeros((T, 3))
    total_grf_right = np.zeros((T, 3))
    for t in range(T):
        data.qpos[:] = qpos_filtered[t, :]
        data.qvel[:] = qvel_filtered[t, :]
        data.qacc[:] = qacc[t, :]

        mj.mj_inverse(model, data)
        tau[t, :] = data.qfrc_inverse.copy()

        # ------------- Ground reaction force ---------------

        left_foot_body_id = model.body("left_foot").id # model.body_name2id("left_foot")
        right_foot_body_id = model.body("right_foot").id # model.body_name2id("right_foot")
        ground_body_id = model.body("world").id # model.body_name2id("world")
        # print(t)
        # print(data.ncon)

        for i in range(data.ncon):
            c = data.contact[i]

            body1 = model.geom_bodyid[c.geom1]
            body2 = model.geom_bodyid[c.geom2]

            bodies = {body1, body2}
            # print(bodies)

            if bodies == {left_foot_body_id, ground_body_id}:
                force_left = np.zeros(6)
                mj.mj_contactForce(model, data, i, force_left)

                R = c.frame.reshape(3, 3)
                f_world_left = R.T @ force_left[:3]

                total_grf_left[t,:] += f_world_left

            if bodies == {right_foot_body_id, ground_body_id}:
                force_right = np.zeros(6)
                mj.mj_contactForce(model, data, i, force_right)

                R = c.frame.reshape(3, 3)
                f_world_right = R.T @ force_right[:3]

                total_grf_right[t,:] += f_world_right


    # print("SUM GRF:", total_grf_left[600] + total_grf_right[600])
    # mass = sum(model.body_mass)
    # print("Expected:", mass * gravity)

    time_vector = np.arange(0, T) * dt

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('GRF')
    ax[0].plot(time_vector, total_grf_left[:,0], label='grf_x_left')
    ax[0].plot(time_vector, total_grf_left[:,1], label='grf_y_left')
    ax[0].plot(time_vector, total_grf_left[:,2], label='grf_z_left')
    ax[0].legend()
    ax[1].plot(time_vector, total_grf_right[:, 0], label='grf_x_right')
    ax[1].plot(time_vector, total_grf_right[:, 1], label='grf_y_right')
    ax[1].plot(time_vector, total_grf_right[:, 2], label='grf_z_right')
    ax[1].legend()
    if save_plot_flag:
        save_path_GRF = "GRF_" + subj_trail + ".png"
        plt.savefig(save_path_GRF)
        print(f"GRF plot saved to {save_path_GRF}")
    plt.show()

    # for i in range(6, nv):
    #     joint_id = model.dof_jntid[i]
    #     joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    #
    #     fig, ax = plt.subplots(nrows=3, ncols=1)
    #     fig.suptitle(joint_name)
    #     ax[0].plot(time_vector, qpos_filtered[:, i+1], linestyle='--', label='qpos')
    #     ax[0].plot(time_vector, qvel_filtered[:,i], label='qvel')
    #     ax[0].legend()
    #     ax[1].plot(time_vector, qacc[:, i], label='qacc')
    #     ax[1].legend()
    #     ax[2].plot(time_vector, tau[:, i], label='tau')
    #     ax[2].legend()
    #     plt.show()



    for i in range(6, nv):
        joint_id = model.dof_jntid[i]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)

        plt.plot(time_vector, qpos_filtered[:, i+1], label=joint_name)
        plt.legend()
        plt.title('Joint angles')

    if save_plot_flag:
        save_path_qpos = "joint_angles_" + subj_trail + ".png"
        plt.savefig(save_path_qpos)
        print(f"Joint angles plot saved to {save_path_qpos}")
    plt.show()

    for i in range(6, nv):
        joint_id = model.dof_jntid[i]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)

        plt.plot(time_vector, tau[:, i], label=joint_name)
        plt.xlim([1.5, time_vector[-1]])
        plt.legend()
        plt.title('Joint moments')

    if save_plot_flag:
        save_path_tau = "joint_moments_" + subj_trail + ".png"
        plt.savefig(save_path_tau)
        print(f"Joint moments plot saved to {save_path_tau}")
    plt.show()


def parse_id():
    parser = argparse.ArgumentParser(description="Inverse dynamics")

    parser.add_argument(
        "--input_xml",
        type=Path,
        default=None,
        help="Path to MuJoCo XML Model."
    )

    parser.add_argument(
        "--input_qpos",
        type=Path,
        default=None,
        help="qpos CSV path"
    )

    parser.add_argument(
        "--subj_trail",
        type=str,
        default=None,
        help="subject trail"
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=300.0,
        help="MoCap sampling frequency"
    )

    parser.add_argument(
        "--filter_order",
        type=int,
        default=3,
        help="Filter order"
    )

    parser.add_argument(
        "--filter_frequency",
        type=float,
        default=10.0,
        help="filter frequency"
    )

    parser.add_argument(
        "--save_plots",
        type=bool,
        default=False,
        help="Do you want to save plots?"
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="gravity"
    )

    args = parser.parse_args()

    qpos_path = args.input_qpos
    model_path = args.input_xml
    subj_trail = args.subj_trail
    fs = args.fs
    N = args.filter_order
    wn = args.filter_frequency
    save_plots = args.save_plots
    gravity = args.gravity

    inverse_dynamics(model_path=model_path, qpos_path=qpos_path, subj_trail= subj_trail,
                     fs=fs, N=N, wn=wn, save_plot_flag=save_plots, gravity=gravity)


if __name__ == "__main__":
    parse_id()
