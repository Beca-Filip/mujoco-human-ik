# checking_marker_names.py
from utils import read_data
import os
from pathlib import Path
import argparse


def checking_marker_names(mocap_dir_path, header_row: int=5):
    rename_map = {
        "l_metatarsa_pos_X": "l_metatarsal_pos_X",
        "l_metatarsa_pos_Y": "l_metatarsal_pos_Y",
        "l_metatarsa_pos_Z": "l_metatarsal_pos_Z",
        "r_metatarsa_pos_X": "r_metatarsal_pos_X",
        "r_metatarsa_pos_Y": "r_metatarsal_pos_Y",
        "r_metatarsa_pos_Z": "r_metatarsal_pos_Z",

        "l_metatarsis_pos_X": "l_metatarsal_pos_X",
        "l_metatarsis_pos_Y": "l_metatarsal_pos_Y",
        "l_metatarsis_pos_Z": "l_metatarsal_pos_Z",
        "r_metatarsis_pos_X": "r_metatarsal_pos_X",
        "r_metatarsis_pos_Y": "r_metatarsal_pos_Y",
        "r_metatarsis_pos_Z": "r_metatarsal_pos_Z",
    }

    for file in sorted(os.listdir(mocap_dir_path)):
        print(file)
        if not file.endswith(".tsv"):
            continue
        mocap_file = os.path.join(mocap_dir_path, file)

        with open(mocap_file, "r") as f:
            lines = f.readlines()

        if not lines:
            print("CSV file is completely empty")
            continue

        header = lines[header_row]

        changed = False
        for old, new in rename_map.items():
            if old in header:
                header = header.replace(old, new)
                changed = True

        if changed:
            lines[header_row] = header
            with open(mocap_file, "w") as f:
                f.writelines(lines)
            print(f"Fixed: {mocap_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checking marker names")

    parser.add_argument(
        "--mocap_dir",
        type=Path,
        help="MoCap TSV"
    )

    args = parser.parse_args()

    mocap_path = args.mocap_dir
    checking_marker_names(mocap_path)