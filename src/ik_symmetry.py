# ik.py
import numpy as np
import mujoco as mj

# ============================================================
# -------------------- INVERSE KINEMATICS --------------------
# ============================================================


def joint_dofs(model: mj.MjModel, ):
    dof_to_joint = {}
    joint_to_dof = {}

    for j in range(model.njnt):
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)

        dof_start = model.jnt_dofadr[j]
        joint_type = model.jnt_type[j]

        if joint_type == mj.mjtJoint.mjJNT_FREE:
            dof_num = 6
        elif joint_type == mj.mjtJoint.mjJNT_BALL:
            dof_num = 3
        else:
            dof_num = 1

        dof_ids = list(range(dof_start, dof_start + dof_num))

        joint_to_dof[joint_name] = dof_ids

        for d in dof_ids:
            dof_to_joint[d] = joint_name
    return joint_to_dof, dof_to_joint


def enforce_joint_limits(model: mj.MjModel, data: mj.MjData) -> None:
    """
    Clamp joint positions to their physical limits.
    """

    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        if model.jnt_type[joint_id] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        q_adr = model.jnt_qposadr[joint_id]
        q_min, q_max = model.jnt_range[joint_id]
        eps = 1e-4
        data.qpos[q_adr] = np.clip(data.qpos[q_adr], q_min + eps, q_max - eps)


def ik_step_multi_site(
    model: mj.MjModel,
    data: mj.MjData,
    site_ids: list[int],
    target_positions: list[np.ndarray],
    qpos_reference: np.ndarray,
    site_weights: list[float] | None = None,
    step_size: float = 0.1,
    damping: float = 5e-1,
    regularization: float = 5e-3,
    beta: float = 1e-3
) -> float:
    """
    Perform a single damped least-squares IK step for multiple sites.
    """

    mj.mj_forward(model, data)

    num_sites = len(site_ids)
    if site_weights is None:
        site_weights = np.ones(num_sites)

    # J = Jacobian
    J_pos = [np.zeros((3, model.nv)) for _ in range(num_sites)]
    J_rot = [np.zeros((3, model.nv)) for _ in range(num_sites)]

    error_blocks = []
    J_blocks = []

    for i, site_id in enumerate(site_ids):
        # Position error
        position_error = target_positions[i] - data.site_xpos[site_id]
        position_error *= site_weights[i]
        error_blocks.append(position_error)

        # Site Jacobian
        mj.mj_jacSite(
            model,
            data,
            J_pos[i],
            J_rot[i],
            site_id
        )
        J_blocks.append(site_weights[i] * J_pos[i])

    # Stack all site errors and Jacobians
    error_vector = np.concatenate(error_blocks, axis=0)
    J = np.vstack(J_blocks)

    # Symmetry
    joint_name_to_dof, dof_id_to_joint = joint_dofs(model)
    symmetry_pairs = [
        (joint_name_to_dof['left_hip_z'], joint_name_to_dof['right_hip_z']),
        (joint_name_to_dof['left_knee_z'], joint_name_to_dof['right_knee_z']),
        (joint_name_to_dof['left_ankle_z'], joint_name_to_dof['right_ankle_z']),
        (joint_name_to_dof['left_shoulder_z'], joint_name_to_dof['right_shoulder_z']),
        (joint_name_to_dof['left_elbow_z'], joint_name_to_dof['right_elbow_z']),
        (joint_name_to_dof['left_wrist_z'], joint_name_to_dof['right_wrist_z'])
    ]

    anti_symmetry_pairs = [
        (joint_name_to_dof['left_hip_x'], joint_name_to_dof['right_hip_x']),
        (joint_name_to_dof['left_hip_y'], joint_name_to_dof['right_hip_y']),
        (joint_name_to_dof['left_ankle_y'], joint_name_to_dof['right_ankle_y']),
        (joint_name_to_dof['left_shoulder_x'], joint_name_to_dof['right_shoulder_x']),
        (joint_name_to_dof['left_shoulder_y'], joint_name_to_dof['right_shoulder_y']),
        (joint_name_to_dof['left_wrist_y'], joint_name_to_dof['right_wrist_y'])
    ]

    J_sym = np.zeros((len(symmetry_pairs), model.nv))

    for i, (left, right) in enumerate(symmetry_pairs):
        J_sym[i, left] = 1
        J_sym[i, right] = -1

    J_anti_sym = np.zeros((len(anti_symmetry_pairs), model.nv))

    for i, (left, right) in enumerate(anti_symmetry_pairs):
        J_anti_sym[i, left] = 1
        J_anti_sym[i, right] = 1

    error_sym = np.zeros(len(symmetry_pairs) + len(anti_symmetry_pairs))
    symmetry_weight = 10

    J = np.vstack([
        J,
        symmetry_weight * J_sym,
        symmetry_weight * J_anti_sym
    ])

    error_vector = np.concatenate([
        error_vector,
        symmetry_weight * error_sym
    ])

    # Penalize deviation from reference posture
    dq_prior = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq_prior, 1, data.qpos, qpos_reference)

    # Soft constraints
    W_lim = np.zeros(model.nv)

    for joint_id in range(model.njnt):

        if not model.jnt_limited[joint_id]:
            continue

        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue

        q_adr = model.jnt_qposadr[joint_id]
        v_adr = model.jnt_dofadr[joint_id]
        q = data.qpos[q_adr]
        q_min, q_max = model.jnt_range[joint_id]

        dist_min = q - q_min
        dist_max = q_max - q

        eps = 1e-4
        if dist_min < dist_max:
            W_lim[v_adr] = 1.0 / (dist_min ** 2 + eps)
        else:
            W_lim[v_adr] = 1.0 / (dist_max ** 2 + eps)

    W_lim = np.diag(W_lim)

    # Damped least-squares system
    Jt_J = J.T @ J

    system_matrix = (
            Jt_J
            + damping * np.eye(model.nv)
            + regularization * np.eye(model.nv)
            + beta * W_lim
    )
    rhs = J.T @ error_vector - regularization * dq_prior

    dq = np.linalg.solve(system_matrix, rhs)

    # Integrate joint update
    mj.mj_integratePos(model, data.qpos, dq * step_size, 1)

    enforce_joint_limits(model, data)

    mj.mj_forward(model, data)

    left_heel_site = model.site("under_maleollus_left").id
    right_heel_site = model.site("under_maleollus_right").id
    left_toe_site = model.site("under_metatarsal_left").id
    right_toe_site = model.site("under_metatarsal_right").id

    left_heel_z = data.site_xpos[left_heel_site][2]
    right_heel_z = data.site_xpos[right_heel_site][2]
    left_toe_z = data.site_xpos[left_toe_site][2]
    right_toe_z = data.site_xpos[right_toe_site][2]

    ground_height = 0.0
    correction = 0
    eps = 1e-6

    if (left_heel_z < ground_height - eps or right_heel_z < ground_height - eps or
            left_toe_z < ground_height - eps or right_toe_z < ground_height - eps):
        correction = max(correction, ground_height - min(left_heel_z, left_toe_z, right_heel_z, right_toe_z))

    data.qpos[2] += correction

    # left_foot_bodyid = model.body("left_foot").id
    # right_foot_bodyid = model.body("right_foot").id
    #
    # radius_left = 0
    # radius_right = 0
    # for i in range(model.ngeom):
    #     if model.geom_bodyid[i] == left_foot_bodyid:
    #         if model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE:
    #             radius_left = model.geom_size[i][0]
    #     if model.geom_bodyid[i] == right_foot_bodyid:
    #         if model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE:
    #             radius_right = model.geom_size[i][0]
    #
    # left_meta_site = model.site("metatarsal_fifth_left").id
    # right_meta_site = model.site("metatarsal_fifth_right").id
    #
    # if radius_left != 0 and radius_right != 0:
    #     left_z = data.site_xpos[left_meta_site][2] - 0.55 * radius_left
    #     right_z = data.site_xpos[right_meta_site][2] - 0.55 * radius_right
    #
    #     ground_height = 0.0
    #     correction = 0
    #     eps = 1e-6
    #
    #     if left_z < ground_height - eps:
    #         correction = max(correction, ground_height-left_z)
    #
    #     if right_z < ground_height - eps:
    #         correction = max(correction, ground_height-right_z)
    #
    #     data.qpos[2] += correction
    #
    # else:
    #     print('Foot geom is not capsule')

    return np.linalg.norm(error_vector)


def solve_ik_for_frame(
    model: mj.MjModel,
    data: mj.MjData,
    site_ids: list[int],
    target_positions: list[np.ndarray],
    site_weights: list[float] | None = None,
    max_iterations: int = 20,
    tolerance: float = 1e-4
) -> None:
    """
    Solve IK for a single frame using iterative refinement.
    """

    mj.mj_forward(model, data)
    qpos_reference = data.qpos.copy()

    for _ in range(max_iterations):
        error_norm = ik_step_multi_site(
            model,
            data,
            site_ids,
            target_positions,
            qpos_reference,
            site_weights
        )
        if error_norm < tolerance:
            break