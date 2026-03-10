"""
Broad physics parameter space with log/uniform sampling.
Builds params dict for simulate from a sample.

Resolution-scale parameters (alpha_ell, tau_ell, beta_ell) replace
the old direct-washout threshold P_wash_frac and power-noise sigma_W_power.
"""

import numpy as np
from typing import Dict, Any, List


def _log_uniform(low: float, high: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def _log_uniform_with_zero(low: float, high: float, size: int, rng: np.random.Generator, zero_frac: float = 0.2) -> np.ndarray:
    out = _log_uniform(low, high, size, rng)
    n_zero = int(zero_frac * size)
    if n_zero > 0:
        out[rng.choice(size, size=n_zero, replace=False)] = 0.0
    return out


PHYSICS_KEYS = [
    "tau0", "tauB", "kappa", "eta", "beta",
    "gamma_wash", "lambda_wash",
    "P_hold_frac", "tau_decay_frac",
    "W_on", "W_off", "P_on_frac", "P_off_frac",
    "alpha_ell", "tau_ell_frac", "beta_ell",
]


def sample_physics(
    N: int,
    seed: int,
    P_max: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Sample N physics regimes.
    Keys: tau0, tauB, kappa, eta, beta, gamma_wash, lambda_wash,
          P_hold_frac, tau_decay_frac, W_on, W_off, P_on_frac, P_off_frac,
          alpha_ell, tau_ell_frac, beta_ell.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(N):
        tau0 = float(_log_uniform(0.05, 5.0, 1, rng)[0])
        tauB = float(_log_uniform(0.1, 20.0, 1, rng)[0])
        if rng.random() < 0.2:
            kappa = 0.0
        else:
            kappa = float(_log_uniform(0.3, 30.0, 1, rng)[0])
        eta = float(rng.uniform(0.2, 1.0))
        beta = float(_log_uniform_with_zero(1e-6, 0.05, 1, rng, zero_frac=0.3)[0])
        gamma_wash = float(rng.uniform(0.0, 1.0))
        lambda_wash = float(rng.uniform(0.0, 3.0))
        P_hold_frac = float(rng.uniform(0.05, 0.7))
        tau_decay_frac = float(_log_uniform(0.05, 1.0, 1, rng)[0])
        W_on = float(rng.uniform(0.8, 1.2))
        W_off = float(rng.uniform(0.5, 1.0))
        if W_off >= W_on:
            W_off = max(0.5, W_on - 0.05)
        P_on_frac = float(rng.uniform(0.4, 1.0))
        P_off_frac = float(rng.uniform(0.1, 0.8))
        if P_off_frac >= P_on_frac:
            P_off_frac = max(0.1, P_on_frac - 0.05)

        alpha_ell = float(rng.uniform(0.2, 3.0))
        tau_ell_frac = float(_log_uniform(0.1, 5.0, 1, rng)[0])
        beta_ell = float(rng.uniform(0.01, 0.3))

        samples.append({
            "tau0": tau0, "tauB": tauB, "kappa": kappa, "eta": eta, "beta": beta,
            "gamma_wash": gamma_wash, "lambda_wash": lambda_wash,
            "P_hold_frac": P_hold_frac, "tau_decay_frac": tau_decay_frac,
            "W_on": W_on, "W_off": W_off, "P_on_frac": P_on_frac, "P_off_frac": P_off_frac,
            "alpha_ell": alpha_ell, "tau_ell_frac": tau_ell_frac, "beta_ell": beta_ell,
        })
    return samples


def params_from_sample(sample: Dict[str, Any], P_max: float = 1.0) -> Dict[str, Any]:
    """Build full params dict for simulate() from a physics sample."""
    tau0 = sample["tau0"]
    tauB = sample["tauB"]
    P_on = sample["P_on_frac"] * P_max
    P_off = sample["P_off_frac"] * P_max
    P_hold = sample["P_hold_frac"] * P_max
    tau_decay = sample["tau_decay_frac"] * tauB
    tau_ell = sample["tau_ell_frac"] * tauB
    return {
        "eta": sample["eta"],
        "beta": sample["beta"],
        "tau0": tau0,
        "tauB": tauB,
        "kappa": sample["kappa"],
        "gamma_wash": sample["gamma_wash"],
        "lambda_wash": sample["lambda_wash"],
        "alpha_ell": sample["alpha_ell"],
        "tau_ell": tau_ell,
        "ell0": 0.0,
        "beta_ell": sample["beta_ell"],
        "W_on": sample["W_on"],
        "W_off": sample["W_off"],
        "P_on": P_on,
        "P_off": P_off,
        "P_max": P_max,
        "P_hold": P_hold,
        "tau_decay": tau_decay,
        # Degradation handles (default off)
        "tau_actuator": sample.get("tau_actuator", 0.0),
        "sigma_W_base": sample.get("sigma_W_base", 0.0),
        "p_elm": sample.get("p_elm", 0.0),
        "elm_crash_frac": sample.get("elm_crash_frac", 0.3),
        "elm_B_threshold": sample.get("elm_B_threshold", 0.7),
        "noise_seed": sample.get("noise_seed", 42),
    }
