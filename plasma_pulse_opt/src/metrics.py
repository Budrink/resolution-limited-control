"""
Metrics for plasma control with resolution-scale ℓ.
Single objective J with configurable weights (lower is better).
"""

import numpy as np
from typing import Dict, Any, Optional

W_MIN_FRAC_DEFAULT = 0.9
FRAC_TOL_DEFAULT = 0.05
J_FAIL_PENALTY_DEFAULT = 75


def compute_metrics(
    t: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    ell: np.ndarray,
    P: np.ndarray,
    W_target: float = 1.0,
    W_min_frac: float = 0.9,
    W_max_frac: float = 1.2,
    ell0: float = 0.0,
) -> Dict[str, float]:
    """
    Compute metrics including ℓ-based resolution excess.
    """
    W_min = W_min_frac * W_target
    W_max = W_max_frac * W_target
    n = len(t)
    if n == 0:
        return {
            "avg_power": 0.0, "tracking_error": 0.0,
            "time_below": 0.0, "time_above": 0.0,
            "barrier_uptime": 0.0, "mean_ell_excess": 0.0, "var_penalty": 0.0,
        }

    avg_power = float(np.mean(P))
    tracking_error = float(np.mean(np.abs(W - W_target)))
    time_below = float(np.mean(W < W_min))
    time_above = float(np.mean(W > W_max))
    barrier_uptime = float(np.mean(B))
    mean_ell_excess = float(np.mean(np.maximum(0.0, ell - ell0)))

    _epsilon = 1e-6
    _var_cap = 100.0
    var_P = float(np.var(P))
    var_penalty = min(1.0 / (var_P + _epsilon), _var_cap)

    switching_effort = float(np.sum(np.abs(np.diff(P)))) if n > 1 else 0.0

    return {
        "avg_power": avg_power,
        "tracking_error": tracking_error,
        "time_below": time_below,
        "time_above": time_above,
        "barrier_uptime": barrier_uptime,
        "mean_ell_excess": mean_ell_excess,
        "var_penalty": var_penalty,
        "switching_effort": switching_effort,
    }


DEFAULT_WEIGHTS = {
    "w1": 1.0,      # avg_power
    "w2": 2.0,      # tracking_error
    "w3": 1.5,      # time_below
    "w4": 1.0,      # time_above
    "w5": 1.0,      # mean_ell_excess (was wash_violation)
    "w_var": 0.1,   # var_penalty
    "w_switch": 0.0, # switching effort
}


def objective_J(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    frac_tol: float = FRAC_TOL_DEFAULT,
    J_fail_penalty: float = J_FAIL_PENALTY_DEFAULT,
) -> float:
    """
    J = w1*avg_power + w2*tracking_error + w3*time_below + w4*time_above
        + w5*mean_ell_excess + w_var*var_penalty + w_switch*switching_effort.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    eff_frac_tol = w.get("frac_tol", frac_tol)
    eff_J_fail = w.get("J_fail_penalty", J_fail_penalty)
    w1 = w.get("w1", DEFAULT_WEIGHTS["w1"])
    w2 = w.get("w2", DEFAULT_WEIGHTS["w2"])
    w3 = w.get("w3", DEFAULT_WEIGHTS["w3"])
    w4 = w.get("w4", DEFAULT_WEIGHTS["w4"])
    w5 = w.get("w5", DEFAULT_WEIGHTS["w5"])
    w_var = w.get("w_var", DEFAULT_WEIGHTS["w_var"])
    w_switch = w.get("w_switch", DEFAULT_WEIGHTS["w_switch"])
    J = (
        w1 * metrics["avg_power"]
        + w2 * metrics["tracking_error"]
        + w3 * metrics["time_below"]
        + w4 * metrics["time_above"]
        + w5 * metrics.get("mean_ell_excess", 0.0)
        + w_var * metrics.get("var_penalty", 0.0)
        + w_switch * metrics.get("switching_effort", 0.0)
    )
    frac_below = metrics.get("time_below", 0.0)
    if frac_below > eff_frac_tol:
        J += eff_J_fail
    return J
