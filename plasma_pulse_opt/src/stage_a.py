"""
Stage A: discover feasible + stable physics regimes.
Random sampling, existence controllers, pruning, output physics_samples.csv.
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Tuple

from .physics_sampling import sample_physics, params_from_sample, PHYSICS_KEYS
from .simulate import run_simulation, choose_dt
from .controllers import make_constant, make_kick_hold, make_pulse_train
from .metrics import compute_metrics
from .stability import classify_stability

EXISTENCE_CONTROLLERS = [
    {"name": "const_Pmax", "type": "constant", "P_frac": 1.0},
    {"name": "const_08", "type": "constant", "P_frac": 0.8},
    {"name": "kick_hold_1", "type": "kick_hold", "P_kick": 1.0, "t_kick": 0.2, "P_hold": 0.4, "period": 2.0},
    {"name": "kick_hold_2", "type": "kick_hold", "P_kick": 1.0, "t_kick": 0.1, "P_hold": 0.3, "period": 1.0},
    {"name": "pulse_1", "type": "pulse", "P_base": 0.2, "DeltaP": 0.8, "period": 1.0, "duty": 0.1},
]


def _make_existence_controller(spec: Dict[str, Any], P_max: float):
    if spec["type"] == "constant":
        return make_constant(spec["P_frac"] * P_max, P_max)
    if spec["type"] == "kick_hold":
        return make_kick_hold(
            spec["P_kick"] * P_max, spec["t_kick"], spec["P_hold"] * P_max, spec["period"], P_max
        )
    if spec["type"] == "pulse":
        return make_pulse_train(
            spec["P_base"] * P_max, spec["DeltaP"] * P_max, spec["period"], spec["duty"], P_max
        )
    return make_constant(P_max, P_max)


def _run_stage_a_one(args: Tuple) -> Dict[str, Any]:
    (
        sample, P_max, W_target, T_total, T_short, W_min_frac, frac_tol, seed,
        transient_cut, window, eps_mean, eps_std, alpha, prune_margin, degradation,
    ) = args
    W_min = W_min_frac * W_target
    merged_sample = {**sample, **degradation}
    merged_sample["noise_seed"] = seed
    params = params_from_sample(merged_sample, P_max)
    if seed is not None:
        np.random.seed(seed)
    dt = choose_dt(params["tau0"], params["tauB"], params.get("tau_ell", 1.0))
    penalty_unstable = 2.0 * W_target
    ell0 = params.get("ell0", 0.0)

    best_row = None
    best_score = -1e9
    feasible_any = False
    stable_any = False

    ctrl_pmax = make_constant(P_max, P_max)
    t_short, W_short, B_short, ell_short, _ = run_simulation(ctrl_pmax, T_short, dt, params, W0=0.5, B0=0.0, M0=0)
    mean_W_short = float(np.mean(W_short[-len(W_short) // 5:]))
    mean_B_short = float(np.mean(B_short[-len(B_short) // 5:]))
    if mean_W_short < W_min - prune_margin and mean_B_short < 0.2:
        frac_below_short = float(np.mean(W_short < W_min))
        row = {**sample, "feasible_any": False, "stable_any": False, "best_controller_name": "const_Pmax",
               "best_score": mean_W_short - alpha * P_max - penalty_unstable,
               "best_avg_power": P_max, "best_tracking_error": np.nan, "best_time_below": frac_below_short,
               "stability_mode": "unstable", "pruned": True}
        return row

    for spec in EXISTENCE_CONTROLLERS:
        ctrl = _make_existence_controller(spec, P_max)
        t, W, B, ell, P = run_simulation(ctrl, T_total, dt, params, W0=0.5, B0=0.0, M0=0)
        metrics = compute_metrics(t, W, B, ell, P, W_target=W_target, W_min_frac=W_min_frac, ell0=ell0)
        stab = classify_stability(t, W, B, W_target=W_target, transient_cut=transient_cut,
                                  window=window, eps_mean=eps_mean, eps_std=eps_std)
        feasible = metrics["time_below"] <= frac_tol
        stable = stab["stable"]
        if feasible:
            feasible_any = True
        if stable:
            stable_any = True
        penalty = 0.0 if stable else penalty_unstable
        mean_W_last = float(np.mean(W[-len(W) // 10:]))
        score = mean_W_last - alpha * metrics["avg_power"] - penalty
        if score > best_score:
            best_score = score
            best_row = {
                **sample,
                "feasible_any": feasible_any,
                "stable_any": stable_any,
                "best_controller_name": spec["name"],
                "best_score": score,
                "best_avg_power": metrics["avg_power"],
                "best_tracking_error": metrics["tracking_error"],
                "best_time_below": metrics["time_below"],
                "stability_mode": stab["mode"],
                "pruned": False,
            }
    if best_row is None:
        best_row = {**sample, "feasible_any": False, "stable_any": False, "best_controller_name": "none",
                    "best_score": -1e9, "stability_mode": "unstable", "pruned": False}
    return best_row


def run_stage_a(
    N_phys: int,
    seed: int,
    P_max: float = 1.0,
    W_target: float = 1.0,
    T_total: float = 60.0,
    T_short: float = 20.0,
    W_min_frac: float = 0.9,
    frac_tol: float = 0.05,
    transient_cut: float = 20.0,
    window: float = 10.0,
    eps_mean: Optional[float] = None,
    eps_std: Optional[float] = None,
    alpha: float = 0.3,
    prune_margin: float = 0.1,
    outdir: str = "results",
    n_workers: Optional[int] = None,
    degradation: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if eps_mean is None:
        eps_mean = 0.02 * W_target
    if eps_std is None:
        eps_std = 0.02 * W_target
    if degradation is None:
        degradation = {}
    samples = sample_physics(N_phys, seed, P_max)
    tasks = [
        (
            s, P_max, W_target, T_total, T_short, W_min_frac, frac_tol, seed + i,
            transient_cut, window, eps_mean, eps_std, alpha, prune_margin, degradation,
        )
        for i, s in enumerate(samples)
    ]
    workers = n_workers if n_workers is not None else min(32, (os.cpu_count() or 4))
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(tasks), desc="Stage A", unit="sample")
    except ImportError:
        pbar = None
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row in ex.map(_run_stage_a_one, tasks, chunksize=max(1, len(tasks) // (workers * 4))):
            rows.append(row)
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()
    df = pd.DataFrame(rows)
    df["feasible_and_stable"] = df["feasible_any"] & df["stable_any"]
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "physics_samples.csv"), index=False)
    n_fs = df["feasible_and_stable"].sum()
    print(f"Stage A: {n_fs}/{N_phys} feasible+stable")
    if n_fs == 0 and W_min_frac >= 0.85:
        if "best_time_below" in df.columns:
            relaxed_ok = (df["best_time_below"] <= 0.20).sum()
            print(f"  (Strict W_min_frac={W_min_frac} yields 0 feasible. With relaxed 0.8: ~{relaxed_ok} would pass.)")
    return df
