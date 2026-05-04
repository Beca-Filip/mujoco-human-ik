# ik.py
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares

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


def joint_constraints(model: mj.MjModel, data: mj.MjData):
    # Joint constraints
    qmin = np.full(model.nq, -np.inf)
    qmax = np.full(model.nq, np.inf)

    for j in range(model.njnt):
        if model.jnt_type[j] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        qadr = model.jnt_qposadr[j]

        lo, hi = model.jnt_range[j]
        qmin[qadr] = lo
        qmax[qadr] = hi

    dq_min = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq_min, 1, data.qpos, qmin)
    dq_max = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq_max, 1, data.qpos, qmax)

    dq_min[0:6] = -10000
    dq_max[0:6] = 10000

    bounds = (dq_min, dq_max)
    return bounds

def solve_ik_for_frame(
    model: mj.MjModel,
    data: mj.MjData,
    site_ids: list[int],
    target_positions: list[np.ndarray],
    site_weights: list[float] | None = None,
    symmetry_weight: int = 10,
    regularization: float = 5e-2,
    step_size: float = 0.1
) -> None:
    """
    Solve IK for a single frame.
    """

    mj.mj_forward(model, data)
    qpos_reference = data.qpos.copy()

    if site_weights is None:
        site_weights = np.ones(len(site_ids))

    num_sites = len(site_ids)

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

    def residual(dq):
        # data.qpos[:] = q
        # mj.mj_integratePos(model, data.qpos, dq * step_size, 1)
        # mj.mj_forward(model, data)

        residuals = []
        error_blocks = []

        # J = Jacobian
        J_pos = [np.zeros((3, model.nv)) for _ in range(num_sites)]
        J_rot = [np.zeros((3, model.nv)) for _ in range(num_sites)]

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

        error_vector = np.concatenate(error_blocks, axis=0)
        J = np.vstack(J_blocks)

        residuals.append(J @ dq - error_vector)

        # Penalize deviation from reference posture
        dq_ref = np.zeros(model.nv)
        mj.mj_differentiatePos(model, dq_ref, 1, data.qpos, qpos_reference)
        residuals.append(np.sqrt(regularization) * (dq - dq_ref))

        # Symmetry
        for i, (left, right) in enumerate(symmetry_pairs):
            dq_ls = dq[left]
            dq_rs = dq[right]

            residuals.append(symmetry_weight * (dq_ls - dq_rs))

        for i, (left, right) in enumerate(anti_symmetry_pairs):
            dq_la = dq[left]
            dq_ra = dq[right]

            residuals.append(symmetry_weight * (dq_la + dq_ra))

        residual = np.concatenate(residuals)
        return residual

    def jacobian(dq):

        # data.qpos[:] = q
        # mj.mj_integratePos(model, data.qpos, dq * step_size, 1)
        # mj.mj_forward(model, data)

        J_pos = [np.zeros((3, model.nv)) for _ in range(num_sites)]
        J_rot = [np.zeros((3, model.nv)) for _ in range(num_sites)]
        J_blocks = []

        for i, site_id in enumerate(site_ids):

            # Site Jacobian
            mj.mj_jacSite(
                model,
                data,
                J_pos[i],
                J_rot[i],
                site_id
            )
            J_blocks.append(site_weights[i] * J_pos[i])

        J = np.vstack(J_blocks)

        # prior regularization
        J = np.vstack([
            J,
            np.sqrt(regularization) * np.eye(model.nv)
        ])

        # symmetry rows
        J_sym = np.zeros((len(symmetry_pairs), model.nv))
        for i, (left, right) in enumerate(symmetry_pairs):
            J_sym[i, left] = 1
            J_sym[i, right] = -1

        J_anti_sym = np.zeros((len(anti_symmetry_pairs), model.nv))
        for i, (left, right) in enumerate(anti_symmetry_pairs):
            J_anti_sym[i, left] = 1
            J_anti_sym[i, right] = 1

        J = np.vstack([
            J,
            symmetry_weight * J_sym,
            symmetry_weight * J_anti_sym
        ])

        return np.vstack(J)

    dq0 = np.zeros(model.nv)
    bounds = joint_constraints(model, data)

    result = least_squares(
        residual,
        x0=dq0,
        bounds=bounds,
        method='trf',
        jac=jacobian
    )

    dq = result.x
    mj.mj_integratePos(model, data.qpos, dq * step_size, 1)
    mj.mj_forward(model, data)

    # foot vs ground
    left_foot_bodyid = model.body("left_foot").id
    right_foot_bodyid = model.body("right_foot").id

    radius_left = 0
    radius_right = 0
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == left_foot_bodyid:
            if model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE:
                radius_left = model.geom_size[i][0]
        if model.geom_bodyid[i] == right_foot_bodyid:
            if model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE:
                radius_right = model.geom_size[i][0]

    left_meta_site = model.site("metatarsal_fifth_left").id
    right_meta_site = model.site("metatarsal_fifth_right").id

    if radius_left != 0 and radius_right != 0:
        left_z = data.site_xpos[left_meta_site][2] - 0.97 * radius_left
        right_z = data.site_xpos[right_meta_site][2] - 0.97 * radius_right

        ground_height = 0.0
        correction = 0
        eps = 1e-6

        if left_z < ground_height - eps:
            correction = max(correction, ground_height - left_z)

        if right_z < ground_height - eps:
            correction = max(correction, ground_height - right_z)

        data.qpos[2] += correction

    else:
        print('Foot geom is not capsule')