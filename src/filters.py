# filters.py
import numpy as np

# ============================================================
# -------------------- TARGET FILTERING ----------------------
# ============================================================


def filter_marker_targets(
    current_targets: np.ndarray,
    previous_targets: np.ndarray,
    max_delta: float = 0.01,
    smoothing: float = 0.7
) -> np.ndarray:
    """
    Temporal filtering and step clamping.
    """

    # Exponential smoothing
    filtered = previous_targets + smoothing * (current_targets - previous_targets)

    # Hard safety clamp
    delta = filtered - previous_targets
    dist = np.linalg.norm(delta, axis=1)
    mask = dist > max_delta
    delta[mask] *= (max_delta / dist[mask])[:, None]

    return previous_targets + delta
