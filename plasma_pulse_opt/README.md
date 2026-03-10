# Plasma Pulse Optimization

Explores **optimal pulsed power control for plasma confinement** using a
reduced-order (0-D) model with barrier memory, threshold hysteresis, and
**resolution-scale degradation**.

## Model

Four variables evolve in time:

| Variable | Meaning |
|----------|---------|
| **W(t)** | Plasma energy (≥ 0) |
| **B(t)** | Transport barrier state ∈ [0, 1] — improves confinement when high |
| **ℓ(t)** | Resolution scale (≥ 0) — grows with heating, degrades control |
| **P(t)** | Applied heating power ∈ [0, P_max] — the control input |

**Energy:**
`dW/dt = η·P − W/τE(B,ℓ) − β·W²`

**Barrier:**
`dB/dt = (B★ − B)/τB − γ_wash·max(0, ℓ − ℓ₀)²`

B★ switches via **hysteresis** on P thresholds (P_on / P_off), mimicking
L→H mode transitions. The barrier has **memory** (afterglow): it does not
collapse immediately when power drops.

**Resolution scale:**
`dℓ/dt = α_ℓ·P − (ℓ − ℓ₀)/τ_ℓ`

Heating power inflates ℓ; ℓ relaxes back to ℓ₀ when power is low.

**Confinement time:**
`τE(B,ℓ) = τ0·(1 + κ·B) / (1 + λ_wash·max(0, ℓ − ℓ₀))`

**Observation noise:**
`σ(ℓ) = σ₀ + β_ℓ·max(0, ℓ − ℓ₀)`

All units are dimensionless (W_target = 1).

## Key idea

The resolution scale ℓ creates a **dual degradation** mechanism:
1. **Barrier washout**: excess ℓ erodes the transport barrier directly
2. **Observation noise**: excess ℓ makes W measurements noisy, degrading
   controller accuracy

Continuous heating keeps ℓ perpetually inflated. Pulsed heating can
**briefly exceed P_on then rest**, letting ℓ relax while the barrier
survives on its memory (afterglow). This mirrors the resolution-limited
control mechanism from the abstract `control.ipynb` model.

## Usage

```bash
pip install -r requirements.txt

# Full pipeline (Stage A + Stage B)
python run.py --stage AB --N_phys 200

# With degradation handles
python run.py --stage AB --N_phys 200 \
  --tau_actuator 0.05 \
  --sigma_W_base 0.02 \
  --p_elm 0.5 --elm_crash_frac 0.3

# Visual comparison across hand-picked regimes
python run_comparison.py
```

## Pipeline

**Stage A** samples a broad physics parameter space, runs
existence controllers (constant, kick-and-hold, pulse), and classifies each
sample as feasible and stable.

**Stage B** takes feasible samples, finds the best constant (or PI) baseline,
then searches pulse parameters. Pulsing wins if J_pulse ≤ J_baseline × (1 − margin).

## Degradation handles

| Flag | Effect |
|------|--------|
| `--tau_actuator` | First-order actuator lag |
| `--sigma_W_base` | Baseline Gaussian noise on W measurement |
| `--p_elm` | ELM crash probability per unit time |
| `--elm_crash_frac` | Fraction of barrier B lost per ELM event |
| `--w_switch` | Switching-effort weight in objective J |

Note: power-dependent noise is now handled internally via the resolution
scale ℓ (parameter `beta_ell` in physics sampling).

## Structure

```
plasma_pulse_opt/
  run.py                  # single entry point
  run_comparison.py       # visual comparison across regimes
  requirements.txt
  demo.ipynb              # quick interactive demo
  src/
    model.py              # ODE: W, B, ℓ, hysteresis, τE
    controllers.py        # constant, PI, pulse train, kick-hold, event-triggered
    simulate.py           # RK4 fixed-step integration
    metrics.py            # objective J, avg_power, tracking_error, mean_ell_excess
    stability.py          # steady / limit-cycle / unstable classification
    physics_sampling.py   # broad parameter sampling (incl. ℓ params)
    stage_a.py            # Stage A: feasibility + stability discovery
    stage_b.py            # Stage B: pulsing advantage search
    plots.py              # feasibility map, advantage map, time-series
  results/                # auto-created; CSV and figures
```
