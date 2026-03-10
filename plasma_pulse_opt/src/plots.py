"""
Plots: feasibility map, pulsing advantage map, time-series comparisons.
Now includes resolution scale ℓ in time-series panels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from .physics_sampling import params_from_sample
from .simulate import run_simulation, choose_dt
from .controllers import make_constant, make_pulse_train, PIController
from .metrics import compute_metrics


def _bin_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    agg: str = "mean",
    n_bins: int = 20,
) -> tuple:
    x = df[x_col].values
    y = df[y_col].values
    v = df[value_col].values
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    Z = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (x >= x_edges[i]) & (x < x_edges[i + 1]) & (y >= y_edges[j]) & (y < y_edges[j + 1])
            if mask.sum() > 0:
                if agg == "mean":
                    Z[j, i] = np.nanmean(v[mask])
                elif agg == "any":
                    Z[j, i] = 1.0 if np.any(v[mask]) else 0.0
    return x_edges, y_edges, Z


def plot_feasibility_map(
    physics_df: pd.DataFrame,
    outpath: str,
    x_col: str = "tauB",
    y_col: str = "kappa",
    n_bins: int = 25,
) -> None:
    if physics_df.empty:
        return
    if "feasible_and_stable" not in physics_df.columns:
        return
    df = physics_df.copy()
    df["feasible_const"] = df["best_controller_name"].astype(str).str.startswith("const", na=False) & df["feasible_and_stable"]
    df["feasible_pulse_only"] = df["feasible_and_stable"] & (~df["feasible_const"])
    x_edges, y_edges, Z_fs = _bin_2d(df, x_col, y_col, "feasible_and_stable", agg="any", n_bins=n_bins)
    _, _, Z_const = _bin_2d(df, x_col, y_col, "feasible_const", agg="any", n_bins=n_bins)
    _, _, Z_pulse_only = _bin_2d(df, x_col, y_col, "feasible_pulse_only", agg="any", n_bins=n_bins)
    Z = np.full_like(Z_fs, np.nan)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z_fs[i, j]) or Z_fs[i, j] == 0:
                Z[i, j] = np.nan
            elif Z_pulse_only[i, j] == 1 and (Z_const[i, j] != 1 or np.isnan(Z_const[i, j])):
                Z[i, j] = 2
            else:
                Z[i, j] = 1
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = matplotlib.colors.ListedColormap(["#4472C4", "#70AD47"])
    cmap.set_bad(color="lightgray")
    im = ax.pcolormesh(x_edges, y_edges, Z, shading="flat", cmap=cmap, vmin=1, vmax=2)
    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2], label="")
    cbar.ax.set_yticklabels(["feasible (const)", "feasible (pulse only)"])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Feasibility map (grey=infeasible)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_pulsing_advantage_map(
    stage_b_df: pd.DataFrame,
    outpath: str,
    x_col: str = "tauB",
    y_col: str = "kappa",
    n_bins: int = 25,
) -> None:
    if stage_b_df.empty or "pulsing_wins" not in stage_b_df.columns:
        return
    x_edges, y_edges, Z = _bin_2d(stage_b_df, x_col, y_col, "pulsing_wins", agg="any", n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="lightgray")
    im = ax.pcolormesh(x_edges, y_edges, Z, shading="flat", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Pulsing wins (1=yes)")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Pulsing advantage (feasible samples only)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_top3_timeseries(
    stage_b_df: pd.DataFrame,
    outdir: str,
    P_max: float = 1.0,
    W_target: float = 1.0,
    T_total: float = 60.0,
    W_min_frac: float = 0.9,
    seed: Optional[int] = None,
    top_n: int = 3,
) -> None:
    """Best baseline vs best pulse, now with ℓ panel (4 rows)."""
    if stage_b_df.empty or "improvement" not in stage_b_df.columns:
        return
    df = stage_b_df[stage_b_df["feasible_stable_baseline"]].copy()
    df = df.nlargest(top_n, "improvement")
    if df.empty:
        return
    from .physics_sampling import PHYSICS_KEYS
    for rank, (_, row) in enumerate(df.iterrows()):
        sample = {k: row[k] for k in PHYSICS_KEYS if k in row.index}
        params = params_from_sample(sample, P_max)
        dt = choose_dt(params["tau0"], params["tauB"], params.get("tau_ell", 1.0))
        if seed is not None:
            np.random.seed(seed + rank)
        if str(row["best_baseline_name"]).startswith("const_"):
            p_frac = float(str(row["best_baseline_name"]).replace("const_", ""))
            ctrl_b = make_constant(p_frac * P_max, P_max)
        else:
            ctrl_b = PIController(W_target, 0.5, 0.2, 0.3 * P_max, P_max)
        t, W_b, B_b, ell_b, P_b = run_simulation(ctrl_b, T_total, dt, params, W0=0.5, B0=0.0, M0=0)
        ctrl_p = make_pulse_train(
            row["best_pulse_P_base_frac"] * P_max,
            row["best_pulse_DeltaP_frac"] * P_max,
            row["best_pulse_period"],
            row["best_pulse_duty"],
            P_max,
        )
        t, W_p, B_p, ell_p, P_p = run_simulation(ctrl_p, T_total, dt, params, W0=0.5, B0=0.0, M0=0)
        fig, axes = plt.subplots(4, 2, sharex="col", figsize=(10, 9))
        for col, (W, B, ell, P, label) in enumerate([
            (W_b, B_b, ell_b, P_b, "baseline"),
            (W_p, B_p, ell_p, P_p, "pulse"),
        ]):
            axes[0, col].plot(t, W, label="W")
            axes[0, col].axhline(W_target, color="gray", ls="--")
            axes[0, col].axhline(W_min_frac * W_target, color="red", ls=":")
            axes[0, col].set_ylabel("W")
            axes[0, col].set_title(label)
            axes[0, col].legend()
            axes[1, col].plot(t, B)
            axes[1, col].set_ylabel("B")
            axes[2, col].plot(t, ell, color="darkorange")
            axes[2, col].set_ylabel(r"$\ell$")
            axes[3, col].plot(t, P)
            axes[3, col].set_ylabel("P")
        axes[3, 0].set_xlabel("t")
        axes[3, 1].set_xlabel("t")
        fig.suptitle(f"Top {rank+1} improvement={row['improvement']:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"timeseries_top{rank+1}_baseline_vs_pulse.png"), dpi=150)
        plt.close()


def generate_autosearch_plots(
    physics_df: pd.DataFrame,
    stage_b_df: Optional[pd.DataFrame],
    outdir: str,
    P_max: float = 1.0,
    W_target: float = 1.0,
    T_total: float = 60.0,
    W_min_frac: float = 0.9,
    seed: Optional[int] = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    plot_feasibility_map(physics_df, os.path.join(outdir, "feasibility_map.png"))
    if stage_b_df is not None and not stage_b_df.empty:
        plot_pulsing_advantage_map(stage_b_df, os.path.join(outdir, "pulsing_advantage_map.png"))
        plot_top3_timeseries(stage_b_df, outdir, P_max=P_max, W_target=W_target,
                             T_total=T_total, W_min_frac=W_min_frac, seed=seed)
