"""
Fixed-step RK4 integration for (W, B, ℓ) with hysteresis mode M.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional

from .model import rhs, clip_B, clip_W, clip_ell, hysteresis_mode
from .controllers import PIController, get_controller_P


def choose_dt(tau0: float, tauB: float, tau_ell: float = 1.0,
              dt_max: float = 0.01, dt_min: float = 1e-4) -> float:
    dt = min(dt_max, tau0 / 200.0, tauB / 200.0, tau_ell / 200.0)
    return float(np.clip(dt, dt_min, dt_max))


def step_rk4(W: float, B: float, ell: float, M: int, P: float,
             dt: float, params: Dict[str, Any]) -> Tuple[float, float, float, int]:
    """One RK4 step for (W, B, ℓ); M updated after step."""
    eta = params["eta"]
    beta = params["beta"]
    tau0 = params["tau0"]
    tauB = params["tauB"]
    kappa = params["kappa"]
    gamma_wash = params["gamma_wash"]
    lambda_wash = params["lambda_wash"]
    alpha_ell = params["alpha_ell"]
    tau_ell = params["tau_ell"]
    ell0 = params["ell0"]
    W_on = params["W_on"]
    W_off = params["W_off"]
    P_on = params["P_on"]
    P_off = params["P_off"]
    P_hold = params["P_hold"]
    tau_decay = params["tau_decay"]

    def f(w, b, el, m):
        dW, dB, dE, _ = rhs(w, b, el, P, m, eta, beta, tau0, tauB, kappa,
                             gamma_wash, lambda_wash, alpha_ell, tau_ell, ell0,
                             W_on, W_off, P_on, P_off, P_hold, tau_decay)
        return dW, dB, dE

    k1W, k1B, k1E = f(W, B, ell, M)
    k2W, k2B, k2E = f(W + 0.5*dt*k1W, B + 0.5*dt*k1B, ell + 0.5*dt*k1E, M)
    k3W, k3B, k3E = f(W + 0.5*dt*k2W, B + 0.5*dt*k2B, ell + 0.5*dt*k2E, M)
    k4W, k4B, k4E = f(W + dt*k3W, B + dt*k3B, ell + dt*k3E, M)

    W_new = W + (dt / 6.0) * (k1W + 2*k2W + 2*k3W + k4W)
    B_new = B + (dt / 6.0) * (k1B + 2*k2B + 2*k3B + k4B)
    ell_new = ell + (dt / 6.0) * (k1E + 2*k2E + 2*k3E + k4E)

    W_new = clip_W(W_new)
    B_new = clip_B(B_new)
    ell_new = clip_ell(ell_new)

    M_new = hysteresis_mode(W_new, P, M,
                            params["W_on"], params["W_off"],
                            params["P_on"], params["P_off"])

    if np.isnan(W_new) or np.isnan(B_new) or np.isnan(ell_new):
        W_new, B_new, ell_new = clip_W(W), clip_B(B), clip_ell(ell)
    return W_new, B_new, ell_new, M_new


def run_simulation(
    controller: Callable,
    T_total: float,
    dt: Optional[float],
    params: Dict[str, Any],
    W0: float = 0.5,
    B0: float = 0.0,
    ell0_init: Optional[float] = None,
    M0: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the plasma model.
    Returns (t, W, B, ell, P_actual).

    Observation noise on W is ℓ-dependent: σ = σ_base + β_ℓ · max(0, ℓ − ℓ₀).
    """
    tau0 = params["tau0"]
    tauB = params["tauB"]
    tau_ell_val = params.get("tau_ell", 1.0)
    ell0_eq = params.get("ell0", 0.0)
    beta_ell = params.get("beta_ell", 0.0)

    if dt is None:
        dt = choose_dt(tau0, tauB, tau_ell_val)
    dt = float(dt)
    n = int(round(T_total / dt))
    dt = T_total / n

    tau_actuator = float(params.get("tau_actuator", 0.0))
    sigma_W_base = float(params.get("sigma_W_base", 0.0))
    has_noise = sigma_W_base > 0 or beta_ell > 0
    p_elm = float(params.get("p_elm", 0.0))
    elm_crash_frac = float(params.get("elm_crash_frac", 0.3))
    elm_B_threshold = float(params.get("elm_B_threshold", 0.7))
    has_elm = p_elm > 0 and elm_crash_frac > 0
    P_max = float(params.get("P_max", 1.0))
    rng = np.random.default_rng(int(params.get("noise_seed", 42)))

    t = np.zeros(n + 1)
    W = np.zeros(n + 1)
    B = np.zeros(n + 1)
    ell = np.zeros(n + 1)
    P_arr = np.zeros(n + 1)

    W[0] = clip_W(W0)
    B[0] = clip_B(B0)
    ell[0] = ell0_init if ell0_init is not None else ell0_eq
    t[0] = 0.0
    ctrl_state = None
    P_cmd, ctrl_state = get_controller_P(controller, 0.0, W[0], B[0], ctrl_state, dt)
    P_actual = float(P_cmd)
    P_arr[0] = P_actual

    M = M0
    for i in range(n):
        ti = t[i]
        Wi, Bi, elli = W[i], B[i], ell[i]

        if has_noise:
            excess = max(0.0, elli - ell0_eq)
            sigma = sigma_W_base + beta_ell * excess
            W_obs = Wi + rng.normal(0.0, max(sigma, 1e-12))
        else:
            W_obs = Wi
        B_obs = Bi

        P_cmd, ctrl_state = get_controller_P(controller, ti, W_obs, B_obs, ctrl_state, dt)

        if tau_actuator > 0:
            P_actual += dt * (float(P_cmd) - P_actual) / tau_actuator
            P_actual = float(np.clip(P_actual, 0.0, P_max))
        else:
            P_actual = float(P_cmd)

        W[i+1], B[i+1], ell[i+1], M = step_rk4(Wi, Bi, elli, M, P_actual, dt, params)
        t[i+1] = ti + dt
        P_arr[i+1] = P_actual

        if has_elm and B[i+1] > elm_B_threshold:
            if rng.random() < p_elm * dt:
                B[i+1] *= (1.0 - elm_crash_frac)

        if np.isnan(W[i+1]) or np.isnan(B[i+1]) or np.isnan(ell[i+1]):
            W[i+1], B[i+1], ell[i+1] = W[i], B[i], ell[i]
        W[i+1] = clip_W(W[i+1])
        B[i+1] = clip_B(B[i+1])
        ell[i+1] = clip_ell(ell[i+1])

    return t, W, B, ell, P_arr
