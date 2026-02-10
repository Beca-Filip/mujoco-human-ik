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

        joint_type = model.jnt_type[joint_id]
        qpos_address = model.jnt_qposadr[joint_id]
        qpos_min, qpos_max = model.jnt_range[joint_id]

        if joint_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            data.qpos[qpos_address] = np.clip(
                data.qpos[qpos_address],
                qpos_min,
                qpos_max
            )


def ik_step_multi_site(
    model: mj.MjModel,
    data: mj.MjData,
    site_ids: list[int],
    target_positions: list[np.ndarray],
    qpos_reference: np.ndarray,
    site_weights: list[float] | None = None,
    step_size: float = 0.1,
    damping: float = 1e-3,
    regularization: float = 5e-3
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

    # Damped least-squares system
    Jt_J = J.T @ J
    system_matrix = (
        Jt_J
        + damping * np.eye(model.nv)
        + regularization * np.eye(model.nv)
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

