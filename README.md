# Geometry of Control

**When a controller acts, it deforms the geometry of its own observations.**

This project explores a simple but far-reaching idea: control effort degrades
observability. Every actuator injects noise, every measurement is distorted by
the very action the controller is taking. We formalise this through a
**resolution tensor** G_ik that couples control input to observation quality,
and study what happens when that tensor has its own dynamics — including
spontaneous structure formation.

## Structure

The project builds in three layers, from abstract to physical:

### 1. Scalar model — `control.ipynb`

A 1D system where a resolution scale ℓ(t) grows with control effort and
relaxes during pauses. Demonstrates:

- **U-shaped error curve**: too little control → state drifts; too much →
  observations degrade
- **Optimal pulse frequency**: periodic observe-then-act control outperforms
  continuous control in a specific frequency band
- **Phase diagram**: the dimensionless ratio k·τ_ℓ determines whether pulsed
  or continuous control is optimal

### 2. Tensor model — `tensor/`

Generalises ℓ to a 2×2 symmetric positive-definite tensor G_ik that encodes
**direction-dependent** observability degradation.

**`tensor_resolution.ipynb`** — Resolution degradation with anisotropic noise.
Shows that anisotropy-aware controllers outperform naive ones when G_ik
deforms under control, and that periodic observe-then-act strategies win
in the |Re(λ)|·τ_G < 1 regime.

**`tensor_structures.ipynb`** — Structural phase transitions. Adds Landau-type
self-interaction to G dynamics. Above a critical control intensity, G
spontaneously breaks symmetry — organising into a stable anisotropic
configuration. Key finding: **self-organised structure improves control by
~8.5%, but only when coupled back to dynamics.** Structure without physical
feedback is counterproductive.

### 3. Plasma confinement — `plasma_pulse_opt/`

A 0-D tokamak model with energy W, transport barrier B, and resolution
scale ℓ. The barrier forms via L→H hysteresis and has memory (afterglow).
Heating inflates ℓ, which degrades both the barrier and observation quality.

Demonstrates that **event-triggered control** (observe during hold, kick when
barrier drops) outperforms continuous PI by 12–60% in regimes with strong
resolution degradation — the plasma-physics instantiation of the abstract
pulsed-control advantage.

See [`plasma_pulse_opt/README.md`](plasma_pulse_opt/README.md) for model
equations and usage.

## Key insight

Control creates geometry. Geometry creates structure. Structure can help
control — but only through physical coupling, not mere information.

The self-organised anisotropy in G_ik produces a **complementary geometry**:
directions of high G (noisy observations) gain extra physical stability,
while directions of low G (clean observations) need active control. A
structure-aware controller exploits this division of labour.

## Requirements

```
numpy
matplotlib
scipy
```

Notebooks are self-contained; the plasma model additionally uses the modules
in `plasma_pulse_opt/src/`.

## License

MIT
