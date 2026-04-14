import pandas as pd


def read_data(file_path, sep: str = "\t", data_start: int = 5, header_row: int = 8):
    try:
        data = pd.read_csv(file_path, sep=sep, skiprows=data_start, header=None)

    except pd.errors.EmptyDataError:
        print("CSV file is completely empty")
        return None

    no_marker_flag = False
    no_col = data.shape[1]
    if no_col != 42:
        print("Incorrect number of markers")
        no_marker_flag = True
        return no_marker_flag
    data = data.dropna(axis=1, how='all')
    header = pd.read_csv(file_path, sep=sep, header=None, nrows=1, skiprows=header_row)
    header = header.dropna(axis=1, how='all')
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
