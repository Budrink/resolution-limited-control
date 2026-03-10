"""
Stability classification and two-tier feasibility.
Accepts optional ℓ array but does not use it for classification
(stability is determined by W and B behaviour).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


def check_two_tier(
    t: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    W_target: float = 1.0,
    window: float = 10.0,
    transient_cut: float = 20.0,
    W_floor_frac: float = 0.85,
    W_hard_frac: float = 0.60,
    MAE_max_frac: float = 0.10,
    eps_mean: Optional[float] = None,
    eps_std: Optional[float] = None,
) -> Tuple[bool, bool, bool, float]:
    if eps_mean is None:
        eps_mean = 0.02 * W_target
    if eps_std is None:
        eps_std = 0.02 * W_target
    W_floor = W_floor_frac * W_target
    W_hard = W_hard_frac * W_target
    MAE_max = MAE_max_frac * W_target
    n = len(t)
    if n < 10:
        return False, False, False, np.nan
    t_end = t[-1]
    if t_end - transient_cut < window * 0.5:
        return False, False, False, np.nan
    mask_last = (t >= t_end - window) & (t <= t_end)
    W_last = W[mask_last]
    if len(W_last) < 5:
        return False, False, False, np.nan
    mean_W_last = float(np.mean(W_last))
    min_W_last = float(np.min(W_last))
    tracking_MAE_last = float(np.mean(np.abs(W_last - W_target)))
    stab = classify_stability(t, W, B, W_target=W_target, transient_cut=transient_cut,
                              window=window, eps_mean=eps_mean, eps_std=eps_std)
    stable = stab["stable"]
    operational_ok = (mean_W_last >= W_floor) and (min_W_last >= W_hard) and stable
    quality_ok = tracking_MAE_last <= MAE_max
    return operational_ok, quality_ok, stable, tracking_MAE_last


def classify_stability(
    t: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    W_target: float = 1.0,
    transient_cut: float = 20.0,
    window: float = 10.0,
    eps_mean: Optional[float] = None,
    eps_std: Optional[float] = None,
) -> Dict[str, Any]:
    if eps_mean is None:
        eps_mean = 0.02 * W_target
    if eps_std is None:
        eps_std = 0.02 * W_target
    n = len(t)
    if n < 10:
        return {"stable": False, "mode": "unstable"}
    t_end = t[-1]
    t_start_post = transient_cut
    if t_end - t_start_post < window * 1.5:
        return {"stable": False, "mode": "unstable"}
    mask_last = (t >= t_end - window) & (t <= t_end)
    mask_prev = (t >= t_end - 2 * window) & (t < t_end - window)
    W_last = W[mask_last]
    W_prev = W[mask_prev]
    B_last = B[mask_last]
    if len(W_last) < 5 or len(W_prev) < 5:
        return {"stable": False, "mode": "unstable"}
    std_last = float(np.std(W_last))
    mean_last = float(np.mean(W_last))
    mean_prev = float(np.mean(W_prev))
    std_prev = float(np.std(W_prev))
    B_ok = float(np.min(B_last)) >= -0.01 and float(np.max(B_last)) <= 1.01
    if not B_ok:
        return {"stable": False, "mode": "unstable"}
    if std_last <= eps_std:
        return {"stable": True, "mode": "steady"}
    if (
        abs(mean_last - mean_prev) <= eps_mean
        and abs(std_last - std_prev) <= eps_std
    ):
        return {"stable": True, "mode": "cycle"}
    return {"stable": False, "mode": "unstable"}
