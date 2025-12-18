import pandas as pd
import mujoco as mj

def read_data(file_path, sep, data_start, header_row):
    data = pd.read_csv(file_path, sep=sep, skiprows=data_start, header=None)
    header = pd.read_csv(file_path, sep=sep, header=None, nrows=1, skiprows=header_row)
    header = header.iloc[:, 1:]
    header.columns = range(header.shape[1])
    data.columns = header.iloc[0, :]
    return data


def get_names(data):
    columns = list(data.columns)
    names = []
    for c in columns:
        name = c[:-2]
        if name not in names:
            names.append(name)
    return names


def site_position(model, qpos, site_id):
    data = mj.MjData(model)
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    return data.site(site_id).xpos.copy()

