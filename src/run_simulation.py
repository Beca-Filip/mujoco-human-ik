# run_simulation.py
from visualization import simulation_qpos_trajectory
import mujoco as mj
import pandas as pd
import argparse
from pathlib import Path


def run_simulation(mj_model_path: Path, qpos_path: Path, pause_flag: bool):
    model = mj.MjModel.from_xml_path(str(mj_model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    q_pos = pd.read_csv(str(qpos_path))
    simulation_qpos_trajectory(model, q_pos.to_numpy(), pause_flag=pause_flag)


def parse_simulation():
    parser = argparse.ArgumentParser(description="Simulation in MuJoCo viewer")

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
        "--pause_simulation",
        type=bool,
        default=False,
        help="Pause simulation"
    )

    args = parser.parse_args()

    qpos_path = args.input_qpos
    model_path = args.input_xml
    pause_sim = args.pause_simulation

    run_simulation(mj_model_path=model_path, qpos_path=qpos_path, pause_flag=pause_sim)


if __name__ == "__main__":
    parse_simulation()


