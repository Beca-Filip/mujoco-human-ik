from visualization import simulation_qpos_trajectory
import mujoco as mj
import pandas as pd

def pom(mj_model_path = 'c:/Users/SVIKI/PycharmProjects/mujoco_human/human_whole.xml', qpos_path = 'c:/Users/SVIKI/PycharmProjects/mujoco_human/qpos_03_1_1.csv'):
    model = mj.MjModel.from_xml_path(str(mj_model_path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    q_pos = pd.read_csv(str(qpos_path))
    simulation_qpos_trajectory(model, q_pos.to_numpy())

pom()

