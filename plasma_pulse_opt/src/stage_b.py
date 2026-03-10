"""
Stage B: within feasible+stable regimes, search for pulsing advantage vs constant/PI.
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Tuple

from .physics_sampling import params_from_sample, PHYSICS_KEYS
from .simulate import run_simulation, choose_dt
from .controllers import make_constant, make_pulse_train, make_kick_hold, PIController
from .metrics import compute_metrics, objective_J
from .stability import classify_stability


def _run_stage_b_one(args: Tuple) -> Dict[str, Any]:
    (
        sample, P_max, W_target, T_total, W_min_frac, frac_tol, seed,
        transient_cut, window, eps_mean, eps_std, enable_pi, n_const, n_pulse_coarse,
        K_refine, win_margin, weights, degradation,
    ) = args
    merged_sample = {**sample, **degradation}
    merged_sample["noise_seed"] = seed
    params = params_from_sample(merged_sample, P_max)
    if seed is not None:
        np.random.seed(seed)
    dt = choose_dt(params["tau0"], params["tauB"], params.get("tau_ell", 1.0))
    ell0 = params.get("ell0", 0.0)

    def run_and_score(controller, name: str) -> Tuple[float, Dict, bool, bool]:
        t, W, B, ell, P = run_simulation(controller, T_total, dt, params, W0=0.5, B0=0.0, M0=0)
        metrics = compute_metrics(t, W, B, ell, P, W_target=W_target, W_min_frac=W_min_frac, ell0=ell0)
        stab = classify_stability(t, W, B, W_target=W_target, transient_cut=transient_cut,
                                  window=window, eps_mean=eps_mean, eps_std=eps_std)
        J = objective_J(metrics, weights)
        feasible = metrics["time_below"] <= frac_tol
        stable = stab["stable"]
        return J, metrics, feasible, stable

    best_baseline_J = 1e9
    best_baseline_metrics = None
    best_baseline_name = None
    for P_frac in np.linspace(0.1, 1.0, n_const):
        ctrl = make_constant(P_frac * P_max, P_max)
        J, metrics, feasible, stable = run_and_score(ctrl, f"const_{P_frac:.2f}")
        if feasible and stable and J < best_baseline_J:
            best_baseline_J = J
            best_baseline_metrics = metrics
            best_baseline_name = f"const_{P_frac:.2f}"

    if enable_pi:
        for Kp in np.linspace(0.2, 1.0, 6):
            for Ki in np.linspace(0.05, 0.4, 6):
                ctrl = PIController(W_target, Kp, Ki, 0.3 * P_max, P_max)
                J, metrics, feasible, stable = run_and_score(ctrl, "pi")
                if feasible and stable and J < best_baseline_J:
                    best_baseline_J = J
                    best_baseline_metrics = metrics
                    best_baseline_name = "pi"

    if best_baseline_metrics is None:
        return {
            **sample,
            "best_baseline_name": "none",
            "best_baseline_J": np.nan,
            "best_pulse_J": np.nan,
            "pulsing_wins": False,
            "improvement": 0.0,
            "feasible_stable_baseline": False,
        }

    periods = [0.5, 1.0, 2.0, 3.0]
    duties = [0.05, 0.1, 0.2, 0.3]
    P_bases = [0.1, 0.2, 0.3]
    DeltaPs = [0.5, 0.7, 0.9]
    candidates = []
    for period in periods:
        for duty in duties:
            for pbase in P_bases:
                for dp in DeltaPs:
                    if pbase + dp > 1.01:
                        continue
                    ctrl = make_pulse_train(pbase * P_max, dp * P_max, period, duty, P_max)
                    J, metrics, feasible, stable = run_and_score(ctrl, "pulse")
                    if feasible and stable:
                        candidates.append((J, period, duty, pbase, dp, metrics))
    candidates.sort(key=lambda x: x[0])
    top_K = candidates[:min(K_refine, len(candidates))]

    for (J0, period, duty, pbase, dp, _) in list(top_K):
        for p_mult in [0.7, 1.0, 1.4]:
            for d_delta in [-0.05, 0, 0.05]:
                p_new = max(0.2, period * p_mult)
                d_new = np.clip(duty + d_delta, 0.02, 0.5)
                ctrl = make_pulse_train(pbase * P_max, dp * P_max, p_new, d_new, P_max)
                J, metrics, feasible, stable = run_and_score(ctrl, "pulse")
                if feasible and stable:
                    candidates.append((J, p_new, d_new, pbase, dp, metrics))
    candidates.sort(key=lambda x: x[0])
    best_pulse_J = candidates[0][0] if candidates else np.nan
    best_pulse_metrics = candidates[0][5] if candidates else None
    best_pulse_period = candidates[0][1] if candidates else np.nan
    best_pulse_duty = candidates[0][2] if candidates else np.nan
    best_pulse_pbase = candidates[0][3] if candidates else np.nan
    best_pulse_dp = candidates[0][4] if candidates else np.nan

    J_threshold = best_baseline_J * (1 - win_margin)
    pulsing_wins = bool(candidates and best_pulse_J <= J_threshold)
    improvement = (best_baseline_J - best_pulse_J) / best_baseline_J if best_baseline_J > 0 and candidates else 0.0

    row = {
        **sample,
        "best_baseline_name": best_baseline_name,
        "best_baseline_J": best_baseline_J,
        "best_baseline_avg_power": best_baseline_metrics["avg_power"],
        "best_baseline_tracking_error": best_baseline_metrics["tracking_error"],
        "best_pulse_J": best_pulse_J,
        "best_pulse_avg_power": best_pulse_metrics["avg_power"] if best_pulse_metrics else np.nan,
        "best_pulse_tracking_error": best_pulse_metrics["tracking_error"] if best_pulse_metrics else np.nan,
        "best_pulse_period": best_pulse_period,
        "best_pulse_duty": best_pulse_duty,
        "best_pulse_P_base_frac": best_pulse_pbase,
        "best_pulse_DeltaP_frac": best_pulse_dp,
        "pulsing_wins": pulsing_wins,
        "improvement": improvement,
        "feasible_stable_baseline": True,
    }
    return row


def run_stage_b(
    physics_df: pd.DataFrame,
    seed: int,
    P_max: float = 1.0,
    W_target: float = 1.0,
    T_total: float = 60.0,
    W_min_frac: float = 0.9,
    frac_tol: float = 0.05,
    transient_cut: float = 20.0,
    window: float = 10.0,
    enable_pi: bool = False,
    n_const: int = 20,
    n_pulse_coarse: int = 50,
    K_refine: int = 20,
    win_margin: float = 0.02,
    weights: Optional[Dict[str, float]] = None,
    outdir: str = "results",
    n_workers: Optional[int] = None,
    degradation: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if degradation is None:
        degradation = {}
    df_f = physics_df[physics_df["feasible_and_stable"]].copy()
    if df_f.empty:
        print("Stage B: no feasible+stable samples, skipping.")
        return pd.DataFrame()
    samples = df_f.to_dict("records")
    samples_clean = []
    for r in samples:
        s = {k: r[k] for k in PHYSICS_KEYS if k in r}
        samples_clean.append(s)
    eps_mean = 0.02 * W_target
    eps_std = 0.02 * W_target
    tasks = [
        (
            s, P_max, W_target, T_total, W_min_frac, frac_tol, seed + i,
            transient_cut, window, eps_mean, eps_std, enable_pi, n_const, n_pulse_coarse,
            K_refine, win_margin, weights, degradation,
        )
        for i, s in enumerate(samples_clean)
    ]
    workers = n_workers if n_workers is not None else min(32, (os.cpu_count() or 4))
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(tasks), desc="Stage B", unit="sample")
    except ImportError:
        pbar = None
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row in ex.map(_run_stage_b_one, tasks, chunksize=max(1, len(tasks) // (workers * 2))):
            rows.append(row)
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()
    df_b = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df_b.to_csv(os.path.join(outdir, "stage_b_results.csv"), index=False)
    n_wins = df_b["pulsing_wins"].sum()
    print(f"Stage B: pulsing wins in {n_wins}/{len(df_b)} feasible+stable samples")
    return df_b
