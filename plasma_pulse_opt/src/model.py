"""
0D plasma confinement model with resolution-scale degradation.

State variables:
  W   — plasma energy (>= 0)
  B   — transport barrier ∈ [0, 1]
  ℓ   — resolution scale (>= 0): grows with heating power, relaxes on τ_ℓ

The resolution scale ℓ mediates two degradation channels:
  1. Barrier washout:  dB/dt  -= γ_wash · (ℓ − ℓ₀)²
  2. Confinement loss:  τ_E   /= (1 + λ_wash · (ℓ − ℓ₀))
  3. Observation noise: σ     = σ₀ + β_ℓ · (ℓ − ℓ₀)

All units dimensionless; W_target = 1.
"""

import numpy as np
from typing import Tuple


def tauE(B: float, ell: float, ell0: float, tau0: float, kappa: float,
         lambda_wash: float) -> float:
    """Confinement time: improves with B, worsens with excess ℓ."""
    excess = max(0.0, ell - ell0)
    return tau0 * (1.0 + kappa * B) / (1.0 + lambda_wash * excess)


def hysteresis_mode(W: float, P: float, M_prev: int,
                    W_on: float, W_off: float, P_on: float, P_off: float) -> int:
    """
    P-dominant hysteresis for L→H transition.
    Barrier target ON if P >= P_on, OFF if P <= P_off.
    """
    if P >= P_on:
        return 1
    if P <= P_off:
        return 0
    return M_prev


def rhs(W: float, B: float, ell: float, P: float, M: int,
        eta: float, beta: float, tau0: float, tauB: float,
        kappa: float, gamma_wash: float, lambda_wash: float,
        alpha_ell: float, tau_ell: float, ell0: float,
        W_on: float, W_off: float, P_on: float, P_off: float,
        P_hold: float, tau_decay: float) -> Tuple[float, float, float, int]:
    """
    Right-hand side for (W, B, ℓ) and updated mode M.
    Returns (dW_dt, dB_dt, dell_dt, M_new).
    """
    B_star = float(M)
    conf = tauE(B, ell, ell0, tau0, kappa, lambda_wash)
    if conf <= 0:
        conf = 1e-12
    dW_dt = eta * P - W / conf - beta * (W ** 2)

    excess = max(0.0, ell - ell0)
    dB_dt = (B_star - B) / tauB - gamma_wash * (excess ** 2)

    if P < P_hold and tau_decay > 0:
        dB_dt -= B / tau_decay

    dell_dt = alpha_ell * P - (ell - ell0) / tau_ell

    M_new = hysteresis_mode(W, P, M, W_on, W_off, P_on, P_off)
    return dW_dt, dB_dt, dell_dt, M_new


def clip_B(B: float) -> float:
    return float(np.clip(B, 0.0, 1.0))


def clip_W(W: float) -> float:
    return float(max(0.0, W))


def clip_ell(ell: float) -> float:
    return float(max(0.0, ell))
