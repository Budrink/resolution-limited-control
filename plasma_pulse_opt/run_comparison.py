#!/usr/bin/env python3
"""
Visual comparison: pulsed vs continuous control,
clean vs degraded, across hand-picked physics regimes.

Resolution-scale ℓ mediates both observation noise and barrier washout.
"""

import os, sys, time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_sampling import params_from_sample
from src.simulate import run_simulation, choose_dt
from src.controllers import make_constant, make_pulse_train, make_kick_hold
from src.metrics import compute_metrics

LOG = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Physics regimes (ℓ-mediated model) ───────────────────────────────────────
REGIMES = {
    "Pulse-friendly\n(slow barrier, fast ℓ relax)": {
        "tau0": 1.0, "tauB": 8.0, "kappa": 8.0, "eta": 0.6, "beta": 0.001,
        "gamma_wash": 0.4, "lambda_wash": 0.8,
        "P_hold_frac": 0.05, "tau_decay_frac": 0.8,
        "W_on": 0.9, "W_off": 0.6, "P_on_frac": 0.5, "P_off_frac": 0.08,
        "alpha_ell": 1.5, "tau_ell_frac": 0.3, "beta_ell": 0.1,
    },
    "Moderate barrier\n(balanced ℓ dynamics)": {
        "tau0": 1.0, "tauB": 6.0, "kappa": 8.0, "eta": 0.7, "beta": 0.002,
        "gamma_wash": 0.35, "lambda_wash": 0.6,
        "P_hold_frac": 0.06, "tau_decay_frac": 0.6,
        "W_on": 0.9, "W_off": 0.6, "P_on_frac": 0.45, "P_off_frac": 0.08,
        "alpha_ell": 1.0, "tau_ell_frac": 0.5, "beta_ell": 0.08,
    },
    "High washout\n(strong ℓ coupling)": {
        "tau0": 1.5, "tauB": 2.0, "kappa": 3.0, "eta": 0.7, "beta": 0.005,
        "gamma_wash": 1.5, "lambda_wash": 2.0,
        "P_hold_frac": 0.15, "tau_decay_frac": 0.15,
        "W_on": 1.1, "W_off": 0.8, "P_on_frac": 0.4, "P_off_frac": 0.25,
        "alpha_ell": 2.5, "tau_ell_frac": 0.2, "beta_ell": 0.2,
    },
    "Balanced\n(marginal regime)": {
        "tau0": 1.0, "tauB": 5.0, "kappa": 6.0, "eta": 0.55, "beta": 0.002,
        "gamma_wash": 0.5, "lambda_wash": 1.0,
        "P_hold_frac": 0.08, "tau_decay_frac": 0.5,
        "W_on": 0.9, "W_off": 0.6, "P_on_frac": 0.45, "P_off_frac": 0.10,
        "alpha_ell": 1.2, "tau_ell_frac": 0.4, "beta_ell": 0.12,
    },
}

DEGRADATION = {
    "tau_actuator": 0.05,
    "sigma_W_base": 0.02,
    "p_elm": 0.5,
    "elm_crash_frac": 0.3,
}

P_MAX = 1.0
T_TOTAL = 30.0
W_TARGET = 1.0
W_MIN_FRAC = 0.9
FEASIBILITY_TOL = 0.20

CONST_LEVELS = np.linspace(0.05, 1.0, 30)

PULSE_GRID = [
    (pbase, dp, period, duty)
    for pbase in [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
    for dp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for period in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    for duty in [0.05, 0.1, 0.2, 0.3, 0.5]
    if pbase + dp <= 1.01
]

KICK_GRID = [
    (p_kick, t_kick, p_hold, period)
    for p_kick in [0.5, 0.7, 0.8, 1.0]
    for t_kick in [0.05, 0.1, 0.2, 0.5]
    for p_hold in [0.03, 0.05, 0.1, 0.2]
    for period in [0.3, 0.5, 1.0, 2.0, 3.0]
]

N_CONST = len(CONST_LEVELS)
N_PULSE = len(PULSE_GRID)
N_KICK = len(KICK_GRID)
N_TOTAL = N_CONST + N_PULSE + N_KICK


def sweep(sample_dict, extra_params=None, tag=""):
    merged = {**sample_dict}
    if extra_params:
        merged.update(extra_params)
    params = params_from_sample(merged, P_MAX)
    dt = choose_dt(params["tau0"], params["tauB"], params.get("tau_ell", 1.0))
    n_steps = int(round(T_TOTAL / dt))
    ell0 = params.get("ell0", 0.0)

    LOG(f"    {tag} dt={dt:.5f}  steps={n_steps}  sims={N_TOTAL}")
    t0 = time.time()

    const_pts = []
    best_const = (None, 1e9)
    for P_frac in CONST_LEVELS:
        ctrl = make_constant(P_frac * P_MAX, P_MAX)
        t, W, B, ell, Parr = run_simulation(ctrl, T_TOTAL, dt, params)
        m = compute_metrics(t, W, B, ell, Parr, W_TARGET, W_MIN_FRAC, ell0=ell0)
        if m["time_below"] <= FEASIBILITY_TOL:
            const_pts.append((m["avg_power"], m["tracking_error"]))
            if m["avg_power"] < best_const[1]:
                best_const = ((t, W, B, ell, Parr, m), m["avg_power"])
    LOG(f"    {tag} const: {len(const_pts)}/{N_CONST} feasible  {time.time()-t0:.1f}s")

    pulse_pts = []
    best_pulse = (None, 1e9)
    t1 = time.time()
    for i, (pbase, dp, period, duty) in enumerate(PULSE_GRID):
        ctrl = make_pulse_train(pbase * P_MAX, dp * P_MAX, period, duty, P_MAX)
        t, W, B, ell, Parr = run_simulation(ctrl, T_TOTAL, dt, params)
        m = compute_metrics(t, W, B, ell, Parr, W_TARGET, W_MIN_FRAC, ell0=ell0)
        if m["time_below"] <= FEASIBILITY_TOL:
            pulse_pts.append((m["avg_power"], m["tracking_error"]))
            if m["avg_power"] < best_pulse[1]:
                best_pulse = ((t, W, B, ell, Parr, m), m["avg_power"])
        if (i + 1) % 200 == 0:
            LOG(f"      {tag} pulse {i+1}/{N_PULSE}  ({len(pulse_pts)} ok)  {time.time()-t1:.0f}s")
    LOG(f"    {tag} pulse: {len(pulse_pts)}/{N_PULSE} feasible  {time.time()-t1:.1f}s")

    kick_pts = []
    best_kick = (None, 1e9)
    t2 = time.time()
    for i, (p_kick, t_kick, p_hold, period) in enumerate(KICK_GRID):
        ctrl = make_kick_hold(p_kick * P_MAX, t_kick, p_hold * P_MAX, period, P_MAX)
        t, W, B, ell, Parr = run_simulation(ctrl, T_TOTAL, dt, params)
        m = compute_metrics(t, W, B, ell, Parr, W_TARGET, W_MIN_FRAC, ell0=ell0)
        if m["time_below"] <= FEASIBILITY_TOL:
            kick_pts.append((m["avg_power"], m["tracking_error"]))
            if m["avg_power"] < best_kick[1]:
                best_kick = ((t, W, B, ell, Parr, m), m["avg_power"])
        if (i + 1) % 200 == 0:
            LOG(f"      {tag} kick {i+1}/{N_KICK}  ({len(kick_pts)} ok)  {time.time()-t2:.0f}s")
    LOG(f"    {tag} kick: {len(kick_pts)}/{N_KICK} feasible  {time.time()-t2:.1f}s")

    elapsed = time.time() - t0
    LOG(f"    {tag} TOTAL {elapsed:.0f}s  feasible: c={len(const_pts)} p={len(pulse_pts)} k={len(kick_pts)}")

    return {
        "const_pts": const_pts, "pulse_pts": pulse_pts, "kick_pts": kick_pts,
        "best_const_ts": best_const[0],
        "best_pulse_ts": best_pulse[0],
        "best_kick_ts": best_kick[0],
    }


def _pareto_front(pts):
    if not pts:
        return []
    arr = np.array(sorted(pts, key=lambda x: x[0]))
    front = [arr[0]]
    for p in arr[1:]:
        if p[1] < front[-1][1]:
            front.append(p)
    return front


def main():
    t_start = time.time()
    outdir = "results/comparison"
    os.makedirs(outdir, exist_ok=True)
    regime_names = list(REGIMES.keys())
    n = len(regime_names)

    LOG(f"=== Comparison: {n} regimes x 2 conditions x {N_TOTAL} sims = {n*2*N_TOTAL} total ===")
    LOG(f"    const={N_CONST}  pulse={N_PULSE}  kick={N_KICK}  T={T_TOTAL}")

    cache = {}
    for j, name in enumerate(regime_names):
        sample = REGIMES[name]
        short = name.split("\n")[0]
        LOG(f"[{j+1}/{n}] ========== {short} ==========")
        for condition, extra in [("Clean", None), ("Degraded", DEGRADATION)]:
            tag = f"{short}/{condition}"
            d = sweep(sample, extra, tag)
            cache[(name, condition)] = d
        LOG(f"  [{j+1}/{n}] done. Elapsed: {time.time()-t_start:.0f}s\n")

    # ── Figure 1: Pareto clouds ──────────────────────────────────────────────
    LOG("Plotting Figure 1: Pareto clouds...")
    fig1, axes1 = plt.subplots(2, n, figsize=(5*n, 9), squeeze=False)
    fig1.suptitle("Pareto frontiers: avg power vs tracking error", fontsize=13, y=1.02)

    for j, name in enumerate(regime_names):
        for row_idx, condition in enumerate(["Clean", "Degraded"]):
            ax = axes1[row_idx, j]
            d = cache[(name, condition)]
            for pts, color, marker, label in [
                (d["const_pts"], "steelblue", "o", "Constant"),
                (d["pulse_pts"], "crimson", "s", "Pulse train"),
                (d["kick_pts"], "seagreen", "^", "Kick-hold"),
            ]:
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, c=color, s=12, alpha=0.35, marker=marker)
                    front = _pareto_front(pts)
                    if len(front) >= 2:
                        fx, fy = zip(*front)
                        ax.plot(fx, fy, color=color, lw=2, alpha=0.9, label=label)
                    elif front:
                        ax.scatter(*front[0], c=color, s=60, marker="*",
                                   edgecolors="black", linewidths=0.5, zorder=5, label=label)
            ax.set_xlabel("Avg power", fontsize=8)
            ax.set_ylabel("Tracking error", fontsize=8)
            ax.set_title(f"{name}\n({condition})", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlim(0, 1.05)
            ax.grid(True, alpha=0.2)

    fig1.tight_layout()
    p = os.path.join(outdir, "pareto_comparison.png")
    fig1.savefig(p, dpi=150, bbox_inches="tight")
    LOG(f"  Saved {p}")
    plt.close(fig1)

    # ── Figure 2: Time-series with ℓ panel ───────────────────────────────────
    LOG("Plotting Figure 2: time-series...")
    fig2, axes2 = plt.subplots(4, n, figsize=(5*n, 11), sharex="col", squeeze=False)
    fig2.suptitle("Best feasible: Constant (blue) vs Pulsed (red) vs Kick (green)",
                  fontsize=12, y=1.02)

    for j, name in enumerate(regime_names):
        d = cache[(name, "Clean")]
        has_any = False
        entries = [
            (d["best_const_ts"], "steelblue", "Const", "-"),
            (d["best_pulse_ts"], "crimson", "Pulse", "-"),
            (d["best_kick_ts"], "seagreen", "Kick", "-"),
        ]
        for data, color, label, ls in entries:
            if data is None:
                continue
            has_any = True
            tt, W, B, ell, Parr, m = data
            axes2[0, j].plot(tt, W, color=color, alpha=0.8, ls=ls, lw=1.2,
                             label=f"{label} (\u0050\u0304={m['avg_power']:.2f})")
            axes2[1, j].plot(tt, B, color=color, alpha=0.8, ls=ls, lw=1.2)
            axes2[2, j].plot(tt, ell, color=color, alpha=0.8, ls=ls, lw=1.2)
            axes2[3, j].plot(tt, Parr, color=color, alpha=0.8, ls=ls, lw=0.8)

        axes2[0, j].axhline(W_TARGET, ls="--", c="grey", lw=0.8, label="W_target")
        axes2[0, j].axhline(W_MIN_FRAC * W_TARGET, ls=":", c="orange", lw=0.8)
        axes2[0, j].set_title(name, fontsize=9)
        if has_any:
            axes2[0, j].legend(fontsize=6, loc="lower right")
        else:
            axes2[0, j].text(0.5, 0.5, "No feasible\ncontroller",
                             transform=axes2[0, j].transAxes, ha="center", va="center",
                             fontsize=12, color="grey")
        axes2[0, j].set_ylabel("W (energy)")
        axes2[1, j].set_ylabel("B (barrier)")
        axes2[2, j].set_ylabel(r"$\ell$ (resolution)")
        axes2[3, j].set_ylabel("P (power)")
        axes2[3, j].set_xlabel("t")
        for row in range(4):
            axes2[row, j].grid(True, alpha=0.15)

    fig2.tight_layout()
    p = os.path.join(outdir, "timeseries_comparison.png")
    fig2.savefig(p, dpi=150, bbox_inches="tight")
    LOG(f"  Saved {p}")
    plt.close(fig2)

    # ── Figure 3: Profit bars ────────────────────────────────────────────────
    LOG("Plotting Figure 3: profit bars...")
    err_cap = 0.5
    profits = {cond: [] for cond in ["Clean", "Degraded"]}
    active_names = []

    for name in regime_names:
        has_data = False
        row_profits = {}
        for condition in ["Clean", "Degraded"]:
            d = cache[(name, condition)]
            all_pulse = d["pulse_pts"] + d["kick_pts"]
            c_pts = d["const_pts"]
            if c_pts and all_pulse:
                c_ok = [p for p, e in c_pts if e < err_cap]
                p_ok = [p for p, e in all_pulse if e < err_cap]
                if not c_ok:
                    c_ok = [p for p, _ in c_pts]
                if not p_ok:
                    p_ok = [p for p, _ in all_pulse]
                bc, bp = min(c_ok), min(p_ok)
                row_profits[condition] = (bc - bp) / bc * 100 if bc > 0 else 0
                has_data = True
            else:
                row_profits[condition] = 0
        if has_data:
            active_names.append(name)
            for cond in ["Clean", "Degraded"]:
                profits[cond].append(row_profits[cond])

    na = len(active_names)
    fig3, ax3 = plt.subplots(figsize=(max(8, 3*na), 5))
    x = np.arange(na)
    w = 0.35
    bars1 = ax3.bar(x - w/2, profits["Clean"], w, label="Clean (ideal)", color="steelblue", alpha=0.85)
    bars2 = ax3.bar(x + w/2, profits["Degraded"], w,
                    label="Degraded (noise+lag+ELM)", color="darkorange", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(active_names, fontsize=8)
    ax3.set_ylabel("Pulsing power savings (%)", fontsize=11)
    ax3.set_title("Pulsed control advantage: ideal vs realistic", fontsize=13)
    ax3.axhline(0, c="grey", lw=1)
    ax3.legend(fontsize=10)
    ax3.grid(axis="y", alpha=0.2)
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 1 if h >= 0 else -1
        ax3.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width()/2, h + offset),
                     ha="center", va=va, fontsize=10, fontweight="bold")

    fig3.tight_layout()
    p = os.path.join(outdir, "profit_comparison.png")
    fig3.savefig(p, dpi=150, bbox_inches="tight")
    LOG(f"  Saved {p}")
    plt.close(fig3)

    # ── Figure 4: Detailed single regime ─────────────────────────────────────
    LOG("Plotting Figure 4: detailed comparison...")
    focus_name = regime_names[0]
    fig4, axes4 = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    fig4.suptitle(f"Detailed: {focus_name.split(chr(10))[0]}", fontsize=13, y=1.01)

    for col, condition in enumerate(["Clean", "Degraded"]):
        d = cache[(focus_name, condition)]
        entries = [
            (d["best_const_ts"], "steelblue", "Constant", 1.5),
            (d["best_pulse_ts"], "crimson", "Pulse train", 1.5),
            (d["best_kick_ts"], "seagreen", "Kick-hold", 1.5),
        ]
        for data, color, label, lw in entries:
            if data is None:
                continue
            tt, W, B, ell, Parr, m = data
            axes4[0, col].plot(tt, W, color=color, lw=lw, alpha=0.85,
                               label=f"{label}  \u0050\u0304={m['avg_power']:.3f}")
            axes4[1, col].plot(tt, B, color=color, lw=lw, alpha=0.85)
            axes4[2, col].plot(tt, ell, color=color, lw=lw, alpha=0.85)
            axes4[3, col].plot(tt, Parr, color=color, lw=0.7, alpha=0.75)

        axes4[0, col].axhline(W_TARGET, ls="--", c="grey", lw=0.8)
        axes4[0, col].axhline(W_MIN_FRAC * W_TARGET, ls=":", c="orange", lw=0.8)
        axes4[0, col].set_title(condition, fontsize=11)
        axes4[0, col].legend(fontsize=7, loc="lower right")
        axes4[3, col].set_xlabel("t", fontsize=10)

    for row, ylabel in enumerate(["W (plasma energy)", "B (barrier)",
                                   r"$\ell$ (resolution scale)", "P (applied power)"]):
        axes4[row, 0].set_ylabel(ylabel, fontsize=10)
        for col in range(2):
            axes4[row, col].grid(True, alpha=0.15)

    fig4.tight_layout()
    p = os.path.join(outdir, "detail_pulse_friendly.png")
    fig4.savefig(p, dpi=150, bbox_inches="tight")
    LOG(f"  Saved {p}")
    plt.close(fig4)

    total = time.time() - t_start
    LOG(f"\n=== Done in {total:.0f}s ({total/60:.1f} min). All figures in {outdir}/ ===")


if __name__ == "__main__":
    main()
