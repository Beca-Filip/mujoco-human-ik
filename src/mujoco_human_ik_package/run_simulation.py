# run_simulation.py
"""
This module provides a MuJoCo-based simulation using a predefined XML model
and a sequence of qpos states stored in a CSV file.

1. Loads a MuJoCo model from an XML file
2. Loads qpos (joint configuration) trajectory from a CSV file
3. Runs a forward simulation step to initialize the model state
4. Visualizes the trajectory using simulation_qpos_trajectory
"""
from .visualization import simulation_qpos_trajectory
import mujoco as mj
import pandas as pd
import argparse
from pathlib import Path


def run_simulation(mj_model_path: Path, qpos_path: Path, pause_flag: bool = False):
    """
    Parameters
    ----------
    mj_model_path: Path
    Path to the MuJoCo XML model file.

    qpos_path: Path
    Path to a CSV file containing qpos trajectories.
    Each row represents one timestep, and each column corresponds
    to a joint position value.

    pause_flag: bool, default=False
    If False, the simulation runs continuously.
    """
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


