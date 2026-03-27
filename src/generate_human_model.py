import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def generate_human_model(filename : str, mass : float, height : float, sex : str, alpha : float = 1.0):
    length_scaling = 1 # scaling coefficient - will be changed depending on subject height
    male_height = 1.77
    female_height = 1.61
    if sex == "male":
        file_name = "anthropometric_table_male.csv"
        shoulder_hip_table = "male_shoulder_hip.csv"
        length_scaling = float(length_scaling / male_height)
        rgba_in = f"0.2 0.4 0.8 {alpha}"
    elif sex == "female":
        file_name = "anthropometric_table_female.csv"
        shoulder_hip_table = "female_shoulder_hip.csv"
        length_scaling = float(length_scaling / female_height)
        rgba_in = f"0.9 0.4 0.6 {alpha}"
    length_scaling = length_scaling*float(height)
    df = pd.read_csv(file_name)

    # First column is segment names, second column is lengths
    segments = df.iloc[:, 0]
    lengths = df.iloc[:, 1] * length_scaling   # scale values
    widths = df.iloc[:, 12] * length_scaling   # scale values
    segment_mass_percentages = df.iloc[:, 2]
    segment_com_x = df.iloc[:, 3]
    segment_com_y = df.iloc[:, 4]
    segment_com_z = df.iloc[:, 5]
    x_rad_gyr_sqr = (df.iloc[:, 6]/100)**2
    y_rad_gyr_sqr = (df.iloc[:, 7]/100)**2
    z_rad_gyr_sqr = (df.iloc[:, 8]/100)**2
    xy_inert_prod = df.iloc[:, 9].apply(complex)
    xz_inert_prod = df.iloc[:, 10].apply(complex)
    yz_inert_prod = df.iloc[:, 11].apply(complex)
    joint_limit_negative_x = df.iloc[:, 13]
    joint_limit_positive_x = df.iloc[:, 14]
    joint_limit_negative_y = df.iloc[:, 15]
    joint_limit_positive_y = df.iloc[:, 16]
    joint_limit_negative_z = df.iloc[:, 17]
    joint_limit_positive_z = df.iloc[:, 18]

    #squaring of the complex inertial products and converting result to float
    xy_inert_prod_sqr = (xy_inert_prod/100)**2
    xz_inert_prod_sqr = (xz_inert_prod/100)**2
    yz_inert_prod_sqr = (yz_inert_prod/100)**2
    xy_inert_prod_float = xy_inert_prod_sqr.apply(lambda x: x.real)
    xz_inert_prod_float = xz_inert_prod_sqr.apply(lambda x: x.real)
    yz_inert_prod_float = yz_inert_prod_sqr.apply(lambda x: x.real)

    # Calculate segment masses based on mass percentage and total user mass
    segment_masses = (segment_mass_percentages / 100) * float(mass)
    #print(segment_masses)

    # fullinertia elements
    I11 = segment_masses * (lengths**2) * x_rad_gyr_sqr
    I22 = segment_masses * (lengths**2) * y_rad_gyr_sqr
    I33 = segment_masses * (lengths**2) * z_rad_gyr_sqr
    I12 = segment_masses * (lengths**2) * xy_inert_prod_float
    I13 = segment_masses * (lengths**2) * xz_inert_prod_float
    I23 = segment_masses * (lengths**2) * yz_inert_prod_float
    # print(segment_masses)
    # print(lengths)
    # print(x_rad_gyr_sqr)
    # print(I11)
    # print(I33)

    # Create dicts
    lengths_dict = dict(zip(segments, lengths))
    widths_dict = dict(zip(segments, widths))
    mass_dict = dict(zip(segments, segment_masses))
    segment_com_x_dict = dict(zip(segments, segment_com_x))
    segment_com_y_dict = dict(zip(segments, segment_com_y))
    segment_com_z_dict = dict(zip(segments, segment_com_z))
    joint_limit_negative_x_dict = dict(zip(segments, joint_limit_negative_x))
    joint_limit_positive_x_dict = dict(zip(segments, joint_limit_positive_x))
    joint_limit_negative_y_dict = dict(zip(segments, joint_limit_negative_y))
    joint_limit_positive_y_dict = dict(zip(segments, joint_limit_positive_y))
    joint_limit_negative_z_dict = dict(zip(segments, joint_limit_negative_z))
    joint_limit_positive_z_dict = dict(zip(segments, joint_limit_positive_z))
    I11_dict = dict(zip(segments, I11))
    I22_dict = dict(zip(segments, I22))
    I33_dict = dict(zip(segments, I33))
    I12_dict = dict(zip(segments, I12))
    I13_dict = dict(zip(segments, I13))
    I23_dict = dict(zip(segments, I23))

    # Local center of mass position dicts
    com_pos_x_dict = {}
    for i in lengths_dict:
        length_x = lengths_dict[i]
        fraction_x = float(segment_com_x_dict[i]) / 100
        # length percentage -> actual length -> local offset
        position_x = fraction_x * length_x
        com_pos_x_dict[i] = position_x

    com_pos_y_dict = {}
    for i in lengths_dict:
        length_y = lengths_dict[i]
        fraction_y = float(segment_com_y_dict[i]) / 100
        # length percentage -> actual length -> local offset
        position_y = fraction_y * length_y
        com_pos_y_dict[i] = position_y

    com_pos_z_dict = {}
    for i in lengths_dict:
        length_z = lengths_dict[i]
        fraction_z = float(segment_com_z_dict[i]) / 100
        # length percentage -> actual length -> local offset
        position_z = fraction_z * length_z
        com_pos_z_dict[i] = position_z

    # Site positions loading and dicts
    df_sites = pd.read_csv("site_positions.csv")

    site_names = df_sites.iloc[:, 0]
    site_segments = df_sites.iloc[:, 1]
    site_specific = df_sites.iloc[:, 5]

    site_x = df_sites.iloc[:, 2]
    site_y = df_sites.iloc[:, 3]
    site_z = df_sites.iloc[:, 4]

    # Site segments length dict
    site_body_dict = dict(zip(site_names, site_segments))
    # Site specific dict
    site_specific_dict = dict(zip(site_names, site_specific))

    # Site coordinates dicts
    site_x_dict = dict(zip(site_names, site_x))
    site_y_dict = dict(zip(site_names, site_y))
    site_z_dict = dict(zip(site_names, site_z))

    for site in site_names:
        segment_name = site_body_dict[site]
        segment_length = lengths_dict[segment_name]
        segment_width = widths_dict[segment_name]

        fraction_x = float(site_x_dict[site]) / 100
        fraction_y = float(site_y_dict[site]) / 100
        fraction_z = float(site_z_dict[site]) / 100

        site_x_dict[site] = fraction_x * segment_length
        site_y_dict[site] = fraction_y * segment_length
        site_z_dict[site] = fraction_z * segment_width

    # Shoulder and hip joint positions loading and dicts
    df = pd.read_csv(shoulder_hip_table)
    special_joint_name = df.iloc[:, 0]
    special_joint_pos_x = df.iloc[:, 1] * length_scaling
    special_joint_pos_y = df.iloc[:, 2] * length_scaling
    special_joint_pos_z = df.iloc[:, 3] * length_scaling

    special_joint_pos_x_dict = dict(zip(special_joint_name, special_joint_pos_x))
    special_joint_pos_y_dict = dict(zip(special_joint_name, special_joint_pos_y))
    special_joint_pos_z_dict = dict(zip(special_joint_name, special_joint_pos_z))

    # print("lengths ", lengths_dict)
    # print("masses ", mass_dict)
    # print("x ", site_x_dict)
    # print("y ", site_y_dict)
    # print("z ", site_z_dict)

    mujoco = ET.Element("mujoco", model=filename.replace(".xml", ""))
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "map", force="0.1", zfar="30")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", offwidth="2560", offheight="1440", elevation="-20", azimuth="120")
    contact = ET.SubElement(mujoco, "contact")
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="1 1 .1", rgba=".8 .8 .8 1")
    ET.SubElement(worldbody, "light", diffuse=".8 .8 .8", pos="0 0 3", dir="0 0 -1")
    ET.SubElement(mujoco, "option", gravity="0 0 -9.81")

    # Calculate thorax generation height, abdomen length is not factored in as hip joint position is given from the start of the abdomen segment
    thorax_gen_height = widths_dict["Foot"]/2 + lengths_dict["Shank"] + lengths_dict["Thigh"] + special_joint_pos_y_dict['Hip'] + lengths_dict['Abdomen'] + lengths_dict['Thorax']

    # Thorax as root segment
    thorax = ET.SubElement(worldbody, "body", name="thorax", pos=f"0 0 {thorax_gen_height}", euler="90 0 0")
    ET.SubElement(thorax, "joint", type="free", pos="0 0 0")
    ET.SubElement(thorax, "geom", type="box", size=f"{lengths_dict['Thorax']/4} {lengths_dict['Thorax']/2} {widths_dict['Thorax']/2}", pos=f"0 -{lengths_dict['Thorax']/2} 0", euler="0 0 0", rgba=rgba_in)
    #ET.SubElement(thorax, "geom", type="capsule", size=f"{lengths_dict['Thorax']/4} {widths_dict['Thorax']/2-lengths_dict['Thorax']/4}", pos=f"0 -{lengths_dict['Thorax']/4} 0", euler="0 0 0", rgba=rgba_in)
    #ET.SubElement(thorax, "geom", type="capsule", size=f"{lengths_dict['Thorax']/4} {widths_dict['Thorax']/2-lengths_dict['Thorax']/4}", pos=f"0 -{lengths_dict['Thorax']*3/4} 0", euler="0 0 0", rgba=rgba_in)
    ET.SubElement(thorax, "inertial", mass=f"{mass_dict['Thorax']}", pos=f"{com_pos_x_dict['Thorax']} {com_pos_y_dict['Thorax']} {com_pos_z_dict['Thorax']}", fullinertia=f"{I11_dict['Thorax']} {I22_dict['Thorax']} {I33_dict['Thorax']} {I12_dict['Thorax']} {I13_dict['Thorax']} {I23_dict['Thorax']}")

    # Head
    head = ET.SubElement(thorax, "body", name="head", pos="0 0 0")
    ET.SubElement(contact, "exclude", body1="thorax", body2="head")
    ET.SubElement(head, "joint", name="head_x", type="hinge", axis="1 0 0", pos="0 0 0", range=f"{-joint_limit_negative_x_dict['Head with Neck']} {joint_limit_positive_x_dict['Head with Neck']}")
    ET.SubElement(head, "joint", name="head_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Head with Neck']} {joint_limit_positive_y_dict['Head with Neck']}")
    ET.SubElement(head, "joint", name="head_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Head with Neck']} {joint_limit_positive_z_dict['Head with Neck']}")
    ET.SubElement(head, "geom", type="sphere", size=f"{lengths_dict['Head with Neck']/2}", pos=f"0 {lengths_dict['Head with Neck']/2} 0", euler="0 0 0", rgba=rgba_in)
    ET.SubElement(head, "inertial", mass=f"{mass_dict['Head with Neck']}", pos=f"{com_pos_x_dict['Head with Neck']} {com_pos_y_dict['Head with Neck']} {com_pos_z_dict['Head with Neck']}", fullinertia = f"{I11_dict['Head with Neck']} {I22_dict['Head with Neck']} {I33_dict['Head with Neck']} {I12_dict['Head with Neck']} {I13_dict['Head with Neck']} {I23_dict['Head with Neck']}")

    # Abdomen
    abdomen = ET.SubElement(thorax, "body", name="abdomen", pos=f"0 {-lengths_dict['Thorax']} 0")
    ET.SubElement(contact, "exclude", body1="thorax", body2="abdomen")
    ET.SubElement(abdomen, "joint", name="abdomen_x", type="hinge", axis="1 0 0", pos="0 0 0", range=f"{-joint_limit_negative_x_dict['Abdomen']} {joint_limit_positive_x_dict['Abdomen']}")
    ET.SubElement(abdomen, "joint", name="abdomen_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Abdomen']} {joint_limit_positive_y_dict['Abdomen']}")
    ET.SubElement(abdomen, "joint", name="abdomen_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Abdomen']} {joint_limit_positive_z_dict['Abdomen']}")
    ET.SubElement(abdomen, "geom", type="capsule", size=f"{lengths_dict['Abdomen']/2} {widths_dict['Abdomen']/2-lengths_dict['Abdomen']/4}", pos=f"0 -{lengths_dict['Abdomen']/2} 0", euler="0 0 0", rgba=rgba_in)
    ET.SubElement(abdomen, "inertial", mass=f"{mass_dict['Abdomen']}", pos=f"{com_pos_x_dict['Abdomen']} {com_pos_y_dict['Abdomen']} {com_pos_z_dict['Abdomen']}", fullinertia=f"{I11_dict['Abdomen']} {I22_dict['Abdomen']} {I33_dict['Abdomen']} {I12_dict['Abdomen']} {I13_dict['Abdomen']} {I23_dict['Abdomen']}")

    # Pelvis
    pelvis = ET.SubElement(abdomen, "body", name="pelvis", pos=f"0 {-lengths_dict['Abdomen']} 0")
    ET.SubElement(contact, "exclude", body1="abdomen", body2="pelvis")
    ET.SubElement(pelvis, "joint", name="pelvis_x", type="hinge", axis="1 0 0", pos="0 0 0", range=f"{-joint_limit_negative_x_dict['Pelvis']} {joint_limit_positive_x_dict['Pelvis']}")
    ET.SubElement(pelvis, "joint", name="pelvis_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Pelvis']} {joint_limit_positive_y_dict['Pelvis']}")
    ET.SubElement(pelvis, "joint", name="pelvis_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Pelvis']} {joint_limit_positive_z_dict['Pelvis']}")
    ET.SubElement(pelvis, "geom", type="capsule", size=f"{lengths_dict['Pelvis']/2} {widths_dict['Pelvis']/2-lengths_dict['Pelvis']/4}", pos=f"0 -{lengths_dict['Pelvis']/2} 0", euler="0 0 0", rgba=rgba_in)
    ET.SubElement(pelvis, "inertial", mass=f"{mass_dict['Pelvis']}", pos=f"{com_pos_x_dict['Pelvis']} {com_pos_y_dict['Pelvis']} {com_pos_z_dict['Pelvis']}", fullinertia=f"{I11_dict['Pelvis']} {I22_dict['Pelvis']} {I33_dict['Pelvis']} {I12_dict['Pelvis']} {I13_dict['Pelvis']} {I23_dict['Pelvis']}")

    # Left thigh
    left_thigh = ET.SubElement(pelvis, "body", name="left_thigh", pos=f"{special_joint_pos_x_dict['Hip']} {-special_joint_pos_y_dict['Hip']} {-special_joint_pos_z_dict['Hip']}")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="left_thigh")
    ET.SubElement(left_thigh, "joint", name="left_hip_x", type="hinge", axis="1 0 0", pos="0 0 0", range=f"{-joint_limit_negative_x_dict['Thigh']} {joint_limit_positive_x_dict['Thigh']}")
    ET.SubElement(left_thigh, "joint", name="left_hip_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Thigh']} {joint_limit_positive_y_dict['Thigh']}")
    ET.SubElement(left_thigh, "joint", name="left_hip_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Thigh']} {joint_limit_positive_z_dict['Thigh']}")
    ET.SubElement(left_thigh, "geom", type="capsule", size=f"{widths_dict['Thigh']/2} {lengths_dict['Thigh']/2-widths_dict['Thigh']/4}", pos = f"0 {-lengths_dict['Thigh']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(left_thigh, "inertial", mass=f"{mass_dict['Thigh']}", pos=f"{com_pos_x_dict['Thigh']} {com_pos_y_dict['Thigh']} {-com_pos_z_dict['Thigh']}", fullinertia=f"{I11_dict['Thigh']} {I22_dict['Thigh']} {I33_dict['Thigh']} {I12_dict['Thigh']} {-I13_dict['Thigh']} {-I23_dict['Thigh']}")

    # Right thigh
    right_thigh = ET.SubElement(pelvis, "body", name="right_thigh", pos=f"{special_joint_pos_x_dict['Hip']} {-special_joint_pos_y_dict['Hip']} {special_joint_pos_z_dict['Hip']}")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="right_thigh")
    ET.SubElement(right_thigh, "joint", name="right_hip_x", type="hinge", axis="1 0 0", pos="0 0 0", range=f"{-joint_limit_positive_x_dict['Thigh']} {joint_limit_negative_x_dict['Thigh']}")
    ET.SubElement(right_thigh, "joint", name="right_hip_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Thigh']} {joint_limit_positive_y_dict['Thigh']}")
    ET.SubElement(right_thigh, "joint", name="right_hip_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Thigh']} {joint_limit_positive_z_dict['Thigh']}")
    ET.SubElement(right_thigh, "geom", type="capsule", size=f"{widths_dict['Thigh']/2} {lengths_dict['Thigh']/2-widths_dict['Thigh']/4}", pos = f"0 {-lengths_dict['Thigh']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(right_thigh, "inertial", mass=f"{mass_dict['Thigh']}", pos=f"{com_pos_x_dict['Thigh']} {com_pos_y_dict['Thigh']} {com_pos_z_dict['Thigh']}", fullinertia=f"{I11_dict['Thigh']} {I22_dict['Thigh']} {I33_dict['Thigh']} {I12_dict['Thigh']} {I13_dict['Thigh']} {I23_dict['Thigh']}")

    # Left shank
    left_shank = ET.SubElement(left_thigh, "body", name="left_shank", pos=f"0 {-lengths_dict['Thigh']} 0")
    ET.SubElement(contact, "exclude", body1="left_thigh", body2="left_shank")
    ET.SubElement(left_shank, "joint", name="left_knee_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Shank']} {joint_limit_positive_z_dict['Shank']}")
    ET.SubElement(left_shank, "geom", type="capsule", size=f"{widths_dict['Shank']/2} {lengths_dict['Shank']/2-widths_dict['Shank']/4}", pos = f"0 {-lengths_dict['Shank']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(left_shank, "inertial", mass=f"{mass_dict['Shank']}", pos=f"{com_pos_x_dict['Shank']} {com_pos_y_dict['Shank']} {-com_pos_z_dict['Shank']}", fullinertia=f"{I11_dict['Shank']} {I22_dict['Shank']} {I33_dict['Shank']} {I12_dict['Shank']} {-I13_dict['Shank']} {-I23_dict['Shank']}")

    # Right shank
    right_shank = ET.SubElement(right_thigh, "body", name="right_shank", pos=f"0 {-lengths_dict['Thigh']} 0")
    ET.SubElement(contact, "exclude", body1="right_thigh", body2="right_shank")
    ET.SubElement(right_shank, "joint", name="right_knee_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Shank']} {joint_limit_positive_z_dict['Shank']}")
    ET.SubElement(right_shank, "geom", type="capsule", size=f"{widths_dict['Shank']/2} {lengths_dict['Shank']/2-widths_dict['Shank']/4}", pos = f"0 {-lengths_dict['Shank']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(right_shank, "inertial", mass=f"{mass_dict['Shank']}", pos=f"{com_pos_x_dict['Shank']} {com_pos_y_dict['Shank']} {com_pos_z_dict['Shank']}", fullinertia=f"{I11_dict['Shank']} {I22_dict['Shank']} {I33_dict['Shank']} {I12_dict['Shank']} {I13_dict['Shank']} {I23_dict['Shank']}")

    # Left foot
    left_foot = ET.SubElement(left_shank, "body", name="left_foot", pos=f"0 {-lengths_dict['Shank']} 0")
    ET.SubElement(contact, "exclude", body1="left_shank", body2="left_foot")
    ET.SubElement(left_foot, "joint", name="left_ankle_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Foot']} {joint_limit_positive_y_dict['Foot']}")
    ET.SubElement(left_foot, "joint", name="left_ankle_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Foot']} {joint_limit_positive_z_dict['Foot']}")
    ET.SubElement(left_foot, "geom", type="capsule", size=f"{widths_dict['Foot']/2} {lengths_dict['Foot']/2-widths_dict['Foot']/4}", pos = f"{lengths_dict['Foot']/2} 0 0", euler="0 90 0", rgba=rgba_in)
    ET.SubElement(left_foot, "inertial", mass=f"{mass_dict['Foot']}", pos=f"{com_pos_x_dict['Foot']} {com_pos_y_dict['Foot']} {-com_pos_z_dict['Foot']}", fullinertia=f"{I11_dict['Foot']} {I22_dict['Foot']} {I33_dict['Foot']} {I12_dict['Foot']} {-I13_dict['Foot']} {-I23_dict['Foot']}")

    # Right foot
    right_foot = ET.SubElement(right_shank, "body", name="right_foot", pos=f"0 {-lengths_dict['Shank']} 0")
    ET.SubElement(contact, "exclude", body1="right_shank", body2="right_foot")
    ET.SubElement(right_foot, "joint", name="right_ankle_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Foot']} {joint_limit_positive_y_dict['Foot']}")
    ET.SubElement(right_foot, "joint", name="right_ankle_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Foot']} {joint_limit_positive_z_dict['Foot']}")
    ET.SubElement(right_foot, "geom", type="capsule", size=f"{widths_dict['Foot']/2} {lengths_dict['Foot']/2-widths_dict['Foot']/4}", pos = f"{lengths_dict['Foot']/2} 0 0", euler="0 90 0", rgba=rgba_in)
    ET.SubElement(right_foot, "inertial", mass=f"{mass_dict['Foot']}", pos=f"{com_pos_x_dict['Foot']} {com_pos_y_dict['Foot']} {com_pos_z_dict['Foot']}", fullinertia=f"{I11_dict['Foot']} {I22_dict['Foot']} {I33_dict['Foot']} {I12_dict['Foot']} {I13_dict['Foot']} {I23_dict['Foot']}")

    # Left upper arm
    left_upper_arm = ET.SubElement(thorax, "body", name="left_upper_arm", pos=f"{special_joint_pos_x_dict['Shoulder']} {-special_joint_pos_y_dict['Shoulder']} {-special_joint_pos_z_dict['Shoulder']}")
    ET.SubElement(contact, "exclude", body1="thorax", body2="left_upper_arm")
    ET.SubElement(left_upper_arm, "joint", name="left_shoulder_x", type="hinge", axis="1 0 0", pos=f"0 0 0", range=f"{-joint_limit_negative_x_dict['Upper Arm']} {joint_limit_positive_x_dict['Upper Arm']}")
    ET.SubElement(left_upper_arm, "joint", name="left_shoulder_y", type="hinge", axis="0 1 0", pos=f"0 0 0", range=f"{-joint_limit_negative_y_dict['Upper Arm']} {joint_limit_positive_y_dict['Upper Arm']}")
    ET.SubElement(left_upper_arm, "joint", name="left_shoulder_z", type="hinge", axis="0 0 1", pos=f"0 0 0", range=f"{-joint_limit_negative_z_dict['Upper Arm']} {joint_limit_positive_z_dict['Upper Arm']}")
    ET.SubElement(left_upper_arm, "geom", type="capsule", size=f"{widths_dict['Upper Arm']/2} {lengths_dict['Upper Arm']/2-widths_dict['Upper Arm']/4}", pos = f"0 {-lengths_dict['Upper Arm']/2 + height*0.0208} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(left_upper_arm, "inertial", mass=f"{mass_dict['Upper Arm']}", pos=f"{com_pos_x_dict['Upper Arm']} {com_pos_y_dict['Upper Arm']} {-com_pos_z_dict['Upper Arm']}", fullinertia=f"{I11_dict['Upper Arm']} {I22_dict['Upper Arm']} {I33_dict['Upper Arm']} {I12_dict['Upper Arm']} {-I13_dict['Upper Arm']} {-I23_dict['Upper Arm']}")

    # Right upper arm
    right_upper_arm = ET.SubElement(thorax, "body", name="right_upper_arm", pos=f"{special_joint_pos_x_dict['Shoulder']} {-special_joint_pos_y_dict['Shoulder']} {special_joint_pos_z_dict['Shoulder']}")
    ET.SubElement(contact, "exclude", body1="thorax", body2="right_upper_arm")
    ET.SubElement(right_upper_arm, "joint", name="right_shoulder_x", type="hinge", axis="1 0 0", pos=f"0 0 0", range=f"{-joint_limit_positive_x_dict['Upper Arm']} {joint_limit_negative_x_dict['Upper Arm']}")
    ET.SubElement(right_upper_arm, "joint", name="right_shoulder_y", type="hinge", axis="0 1 0", pos=f"0 0 0", range=f"{-joint_limit_negative_y_dict['Upper Arm']} {joint_limit_positive_y_dict['Upper Arm']}")
    ET.SubElement(right_upper_arm, "joint", name="right_shoulder_z", type="hinge", axis="0 0 1", pos=f"0 0 0", range=f"{-joint_limit_negative_z_dict['Upper Arm']} {joint_limit_positive_z_dict['Upper Arm']}")
    ET.SubElement(right_upper_arm, "geom", type="capsule", size=f"{widths_dict['Upper Arm']/2} {lengths_dict['Upper Arm']/2-widths_dict['Upper Arm']/4}", pos = f"0 {-lengths_dict['Upper Arm']/2 + height*0.0208} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(right_upper_arm, "inertial", mass=f"{mass_dict['Upper Arm']}", pos=f"{com_pos_x_dict['Upper Arm']} {com_pos_y_dict['Upper Arm']} {com_pos_z_dict['Upper Arm']}", fullinertia=f"{I11_dict['Upper Arm']} {I22_dict['Upper Arm']} {I33_dict['Upper Arm']} {I12_dict['Upper Arm']} {I13_dict['Upper Arm']} {I23_dict['Upper Arm']}")

    # Left forearm
    left_forearm = ET.SubElement(left_upper_arm, "body", name="left_forearm", pos=f"0 {-lengths_dict['Upper Arm']} 0")
    ET.SubElement(contact, "exclude", body1="left_upper_arm", body2="left_forearm")
    ET.SubElement(left_forearm, "joint", name="left_elbow_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Forearm']} {joint_limit_positive_z_dict['Forearm']}")
    ET.SubElement(left_forearm, "geom", type="capsule", size=f"{widths_dict['Forearm']/2} {lengths_dict['Forearm']/2-widths_dict['Forearm']/4}", pos=f"0 {-lengths_dict['Forearm']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(left_forearm, "inertial", mass=f"{mass_dict['Forearm']}", pos=f"{com_pos_x_dict['Forearm']} {com_pos_y_dict['Forearm']} {-com_pos_z_dict['Forearm']}", fullinertia=f"{I11_dict['Forearm']} {I22_dict['Forearm']} {I33_dict['Forearm']} {I12_dict['Forearm']} {-I13_dict['Forearm']} {-I23_dict['Forearm']}")

    # Right forearm
    right_forearm = ET.SubElement(right_upper_arm, "body", name="right_forearm", pos=f"0 {-lengths_dict['Upper Arm']} 0")
    ET.SubElement(contact, "exclude", body1="right_upper_arm", body2="right_forearm")
    ET.SubElement(right_forearm, "joint", name="right_elbow_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Forearm']} {joint_limit_positive_z_dict['Forearm']}")
    ET.SubElement(right_forearm, "geom", type="capsule", size=f"{widths_dict['Forearm']/2} {lengths_dict['Forearm']/2-widths_dict['Forearm']/4}", pos=f"0 {-lengths_dict['Forearm']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(right_forearm, "inertial", mass=f"{mass_dict['Forearm']}", pos=f"{com_pos_x_dict['Forearm']} {com_pos_y_dict['Forearm']} {com_pos_z_dict['Forearm']}", fullinertia=f"{I11_dict['Forearm']} {I22_dict['Forearm']} {I33_dict['Forearm']} {I12_dict['Forearm']} {I13_dict['Forearm']} {I23_dict['Forearm']}")

    # Left hand
    left_hand = ET.SubElement(left_forearm, "body", name="left_hand", pos=f"0 {-lengths_dict['Forearm']} 0")
    ET.SubElement(contact, "exclude", body1="left_forearm", body2="left_hand")
    ET.SubElement(left_hand, "joint", name="left_wrist_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Hand']} {joint_limit_positive_y_dict['Hand']}")
    ET.SubElement(left_hand, "joint", name="left_wrist_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Hand']} {joint_limit_positive_z_dict['Hand']}")
    ET.SubElement(left_hand, "geom", type="capsule", size=f"{widths_dict['Hand']/2} {lengths_dict['Hand']/2-widths_dict['Hand']/4}", pos=f"0 {-lengths_dict['Hand']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(left_hand, "inertial", mass=f"{mass_dict['Hand']}", pos=f"{com_pos_x_dict['Hand']} {com_pos_y_dict['Hand']} {-com_pos_z_dict['Hand']}", fullinertia=f"{I11_dict['Hand']} {I22_dict['Hand']} {I33_dict['Hand']} {I12_dict['Hand']} {-I13_dict['Hand']} {-I23_dict['Hand']}")

    # Right hand
    right_hand = ET.SubElement(right_forearm, "body", name="right_hand", pos=f"0 {-lengths_dict['Forearm']} 0")
    ET.SubElement(contact, "exclude", body1="right_forearm", body2="right_hand")
    ET.SubElement(right_hand, "joint", name="right_wrist_y", type="hinge", axis="0 1 0", pos="0 0 0", range=f"{-joint_limit_negative_y_dict['Hand']} {joint_limit_positive_y_dict['Hand']}")
    ET.SubElement(right_hand, "joint", name="right_wrist_z", type="hinge", axis="0 0 1", pos="0 0 0", range=f"{-joint_limit_negative_z_dict['Hand']} {joint_limit_positive_z_dict['Hand']}")
    ET.SubElement(right_hand, "geom", type="capsule", size=f"{widths_dict['Hand']/2} {lengths_dict['Hand']/2-widths_dict['Hand']/4}", pos=f"0 {-lengths_dict['Hand']/2} 0", euler="-90 0 0", rgba=rgba_in)
    ET.SubElement(right_hand, "inertial", mass=f"{mass_dict['Hand']}", pos=f"{com_pos_x_dict['Hand']} {com_pos_y_dict['Hand']} {com_pos_z_dict['Hand']}", fullinertia=f"{I11_dict['Hand']} {I22_dict['Hand']} {I33_dict['Hand']} {I12_dict['Hand']} {I13_dict['Hand']} {I23_dict['Hand']}")

    # Keyframe for t-pose and a-pose
    qpos_vals_t_pose = [
        0, 0, thorax_gen_height, 0.7071, 0.7071, 0, 0, #thorax
        0, 0, 0, #head
        0, 0, 0, #abdomen
        0, 0, 0, #pelvis
        0, 0, 0, #left thigh
        0, #left shank
        0, 0, #left foot
        0, 0, 0, #right thigh
        0, #right shank
        0, 0, #right foot
        1.57079632679, 0, 0, #left upper arm
        0, #left forearm
        0, 0, #left hand
        -1.57079632679, 0, 0, #right upper arm
        0, #right forearm
        0, 0 #right hand
    ]
    qpos_vals_a_pose = [
        0, 0, thorax_gen_height, 0.7071, 0.7071, 0, 0, #thorax
        0, 0, 0, #head
        0, 0, 0, #abdomen
        0, 0, 0, #pelvis
        0, 0, 0, #left thigh
        0, #left shank
        0, 0, #left foot
        0, 0, 0, #right thigh
        0, #right shank
        0, 0, #right foot
        0.75, 0, 0, #left upper arm
        0, #left forearm
        0, 0, #left hand
        -0.75, 0, 0, #right upper arm
        0, #right forearm
        0, 0 #right hand
    ]
    keyframe = ET.SubElement(mujoco, "keyframe")
    ET.SubElement(keyframe, "key", name="t-pose", qpos=" ".join(map(str, qpos_vals_t_pose)))
    ET.SubElement(keyframe, "key", name="a-pose", qpos=" ".join(map(str, qpos_vals_a_pose)))

    variable_name_dict = {
        "thorax": thorax, 
        "head": head, 
        "abdomen": abdomen,
        "pelvis": pelvis,
        "left_thigh": left_thigh,
        "right_thigh": right_thigh,
        "left_shank": left_shank,
        "right_shank": right_shank,
        "left_foot": left_foot,
        "right_foot": right_foot,
        "left_upper_arm": left_upper_arm,
        "right_upper_arm": right_upper_arm,
        "left_forearm": left_forearm,
        "right_forearm": right_forearm,
        "left_hand": left_hand,
        "right_hand": right_hand
    }
    for site in site_names:
        specific_body_part = site_specific_dict[site]
        ET.SubElement(variable_name_dict[specific_body_part], "site", name=site, pos=f"{site_x_dict[site]} {site_y_dict[site]} {site_z_dict[site]}", size="0.02", rgba="1 0 0 1")

    # TODO: Limit joints
    # TODO: How to determine segment widths? Should we also scale by width?
    # TODO: How does collision work for overlapping segments?
    # TODO: Add mass etc...
    # TODO: Potential issue with thigh body position in relation to pelvis

    # Save to file
    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a human MJCF based on sex, height, and mass, according to Dumas (2018)."
    )

    parser.add_argument(
        "-o", "--output",
        default="human.xml",
        help="Output XML filename (default: human.xml)"
    )

    parser.add_argument(
        "-m", "--mass",
        required=True,
        type=float,
        help="Mass of the human who's MJCF we want to generate in kilogram"
    )

    parser.add_argument(
        "-t", "--tall",
        required=True,
        type=float,
        help="Height of the human who's MJCF we want to generate in m"
    )

    parser.add_argument(
        "-s", "--sex",
        required=True,
        choices=["male", "female"],
        help="Sex of the human who's MJCF we want to generate ('male' or 'female')"
    )

    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=1.0,
        help="The alpha value of the generated mesh (0: transparent, 1: opaque)"
    )

    args = parser.parse_args()

    generate_human_model(filename=args.output, mass=args.mass, height=args.tall, sex=args.sex, alpha=args.alpha)
