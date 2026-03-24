# ik.py
import numpy as np
import mujoco as mj

# ============================================================
# -------------------- INVERSE KINEMATICS --------------------
# ============================================================

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