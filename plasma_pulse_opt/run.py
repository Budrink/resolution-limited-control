#!/usr/bin/env python3
"""
Two-stage autosearch: Stage A (feasible+stable regimes), Stage B (pulsing advantage).
  python run.py --stage AB --N_phys 300 --outdir results/autosearch
"""

import argparse
import os
import sys
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stage_a import run_stage_a
from src.stage_b import run_stage_b
from src.plots import generate_autosearch_plots


def main():
    p = argparse.ArgumentParser(description="Plasma autosearch: Stage A + Stage B")
    p.add_argument("--stage", choices=["A", "B", "AB"], default="AB")
    p.add_argument("--N_phys", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--W_min_frac", type=float, default=0.9)
    p.add_argument("--frac_tol", type=float, default=0.05)
    p.add_argument("--P_max", type=float, default=1.0)
    p.add_argument("--T_total", type=float, default=60.0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--enable_pi", action="store_true")
    # Degradation handles
    p.add_argument("--tau_actuator", type=float, default=0.0, help="Actuator lag (0=off)")
    p.add_argument("--sigma_W_base", type=float, default=0.0, help="Baseline measurement noise (0=off)")
    p.add_argument("--p_elm", type=float, default=0.0, help="ELM probability per unit time (0=off)")
    p.add_argument("--elm_crash_frac", type=float, default=0.3)
    p.add_argument("--w_switch", type=float, default=0.0, help="Switching-effort weight in J (0=off)")
    args = p.parse_args()

    degradation = {
        "tau_actuator": args.tau_actuator,
        "sigma_W_base": args.sigma_W_base,
        "p_elm": args.p_elm,
        "elm_crash_frac": args.elm_crash_frac,
    }

    if args.outdir is None:
        args.outdir = os.path.join("results", "autosearch_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    n_workers = 6
    if args.workers != "auto":
        try:
            n_workers = int(args.workers)
        except ValueError:
            n_workers = 6

    physics_df = None
    if args.stage in ("A", "AB"):
        physics_df = run_stage_a(
            N_phys=args.N_phys, seed=args.seed, P_max=args.P_max,
            W_target=1.0, T_total=args.T_total, T_short=20.0,
            W_min_frac=args.W_min_frac, frac_tol=args.frac_tol,
            transient_cut=20.0, window=10.0,
            outdir=outdir, n_workers=n_workers, degradation=degradation,
        )

    stage_b_df = None
    if args.stage in ("B", "AB"):
        if args.stage == "B":
            physics_path = os.path.join(outdir, "physics_samples.csv")
            if not os.path.isfile(physics_path):
                print(f"Stage B requires {physics_path}. Run --stage AB first.")
                return 1
            physics_df = pd.read_csv(physics_path)
        if physics_df is not None and "feasible_and_stable" in physics_df.columns and physics_df["feasible_and_stable"].any():
            weights_override = {"w_switch": args.w_switch} if args.w_switch > 0 else None
            stage_b_df = run_stage_b(
                physics_df=physics_df, seed=args.seed + 1000, P_max=args.P_max,
                W_target=1.0, T_total=args.T_total,
                W_min_frac=args.W_min_frac, frac_tol=args.frac_tol,
                enable_pi=args.enable_pi, weights=weights_override,
                outdir=outdir, n_workers=n_workers, degradation=degradation,
            )
        else:
            print("Stage B: no feasible+stable samples to run.")

    generate_autosearch_plots(
        physics_df if physics_df is not None else pd.DataFrame(),
        stage_b_df, outdir=outdir, P_max=args.P_max,
        W_target=1.0, T_total=args.T_total,
        W_min_frac=args.W_min_frac, seed=args.seed,
    )
    print(f"Outputs saved to {outdir}")
    return 0


if __name__ == "__main__":
    from multiprocessing import current_process
    if current_process().name == "MainProcess":
        sys.exit(main())
