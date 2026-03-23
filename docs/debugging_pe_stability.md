# PE IMEX Stability Bug — Debugging Notes

## Problem

The PE IMEX leapfrog time stepper blows up at step 4 with a small vorticity perturbation, while Dinosaur (reference implementation) survives 20+ steps with physically identical initial conditions.

- **Resting state**: Stable for 100+ steps in both codes. Not affected.
- **Perturbed state**: Blows up at step 4. Divergence and lnps grow explosively.
- **Euler init**: Produces identical grid-point results to Dinosaur (ratio 1.0000 for div and lnps).
- **Step 2**: Vorticity matches Dinosaur, but divergence is 60x too large and lnps is 854x too large.

## What has been verified correct

| Component | Method | Result |
|-----------|--------|--------|
| Sigma ratios (α) | Matches Dinosaur | ✓ |
| Geopotential weights (G) | Matches Dinosaur | ✓ |
| Temperature implicit weights (H) | Matches Dinosaur (including K terms), rtol=1e-13 | ✓ |
| omega_over_pressure | Matches Dinosaur at 1e-8 (float32 limit) | ✓ |
| Implicit inverse roundtrip | 1e-17 error with random state + varying T_ref | ✓ |
| F+L=0 for resting state | Exact zero for both isothermal and lapse-rate | ✓ |
| SH normalization | Identified as √(4π) ratio for m=0; mode-dependent for m>0 | ✓ understood |
| Grid-point winds from matched vorticity | Ratio 1.0000 | ✓ |
| Grid-point ∇(lnps) from matched lnps | Match (both codes give same gradient) | ✓ |
| Euler init grid-point output | Ratio 1.0000 for div, lnps | ✓ |
| IMEX leapfrog formula | Matches Dinosaur's `semi_implicit_leapfrog` exactly | ✓ |
| Robert-Asselin formula | Matches Dinosaur's `robert_asselin_leapfrog_filter` | ✓ |

## What has been changed (during debugging)

1. **Temperature flux uses T' not full T** (`primitive_equations.py` line 272): `t_flux_a = t_prime_grid * ...` — matches Dinosaur's `horizontal_scalar_advection(temperature_variation)`.

2. **Implicit terms/inverse use T' for geopotential** (`semi_implicit.py`): `t_prime = state.temperature.at[:, 0].add(-t_ref)` before computing `G @ t_prime`. Matches Dinosaur which operates on `state.temperature_variation`.

3. **Vertical advection split** (`primitive_equations.py`): `σ̇_full` with T', `σ̇_explicit` with T_ref. Matches Dinosaur's `nodal_temperature_vertical_tendency`.

## Where the discrepancy appears

At step 2 (first IMEX leapfrog step after Euler init), with physically matched states:

| Field | Notus grid max | Dinosaur grid max (SI) | Ratio |
|-------|---------------|----------------------|-------|
| vorticity | 1.870e-6 | 1.870e-6 | 1.00 |
| divergence | 4.04e-5 | 6.68e-7 | 60x |
| lnps | 6.90e-1 | 8.08e-4 | 854x |

Vorticity (fully explicit, no implicit coupling) matches perfectly. Divergence and lnps (coupled through implicit solver) diverge massively.

## Explicit tendency comparison (F(x1) after Euler init)

Using grid-point values with matched physical states:

| Tendency | Notus | Dinosaur (SI) | Ratio |
|----------|-------|---------------|-------|
| F.div | 8.5e-10 /s | 1.0e-6 /s | 1/1000 |
| F.T | 2.4e-3 K/s | 2.1e-10 K/s | 11M x |
| F.lnps | 7.5e-5 /s | 1.2e-11 /s | 6.4M x |

The ratios for F.T and F.lnps are approximately equal to the Earth radius `a = 6.371e6`. The ratio for F.div is approximately `1/a`. This pattern (T and lnps too large by ~a, div too small by ~a) strongly suggests a radius scaling issue somewhere in the explicit tendency computation, BUT individual components (winds, gradients, omega_over_pressure) all match when tested in isolation.

## What to investigate next

### 1. Compare F(x1) term by term on the grid

The explicit temperature tendency has three contributions:
- Horizontal flux divergence: `-spectral_divergence(T'*u/cos², T'*v/cos²)`
- Vertical advection: `vert_adv(σ̇_full, T') + vert_adv(σ̇_explicit, T_ref)`
- Adiabatic heating: `κ*(T_ref * ω_p_explicit + T' * ω_p_full)`

Compute EACH term separately in both codes for the x1 state and compare grid-point values. This isolates which sub-term has the wrong magnitude.

### 2. Compare the Euler-init state in SPECTRAL space

Grid-point div and lnps match (ratio 1.0), but the spectral distributions could differ. The gradient of lnps depends on spectral content, not just the max. Dump the first ~20 spectral coefficients of lnps in both codes after Euler init and compare.

### 3. Compare intermediate state before implicit inverse in the IMEX step

The intermediate is `prev + 2*dt*(F(curr) + (1-α)*L(prev))`. Compute this in both codes and compare ALL fields on the grid BEFORE the implicit inverse. This isolates whether the problem is in forming the intermediate or in the solve.

### 4. Compare the spectral divergence operator action

Our `spectral_divergence(A, B, T, a)` divides by `a`. Dinosaur's `div_cos_lat(A, B)` uses `a=1` (nondimensional). When we pass SI-magnitude grid products, does the `1/a` division produce the correct physical result? Test: compute `spectral_divergence` of a known analytic vector field and compare the grid-point result against the analytical divergence.

### 5. Check the `1/a` bookkeeping in combined momentum flux

The combined momentum terms include `RT'∇(lnps)`. In our code:
```python
rt_grad_u = gas_constant * t_prime_grid * dlnps_dlam_bc
```
The `dlnps_dlam` is a spectral derivative that may or may not include `1/a`. If `zonal_derivative` returns `∂/∂λ` (no `1/a`), then the physical gradient is `(1/a)*∂/∂λ`, and the flux should account for this. Check whether `zonal_derivative` and `meridional_derivative` include the `1/a` factor and whether the combined momentum flux is consistent.

### 6. Build a minimal reproducer

Strip down to the simplest possible case: 2 levels, T5 (truncation 5), single spectral mode perturbation. Compute one explicit tendency call and one implicit inverse in both codes, comparing every intermediate array. The small dimensions make it feasible to print and diff entire arrays.

## Test status

- 25 IMEX tests pass (resting state, shapes, JIT, filters, mass conservation, physics consistency)
- 1 test xfail: `test_perturbation_20_steps` — documents the known bug
- 131 pre-existing tests pass
- All lint clean

## Files involved

- `src/notus/dynamics/primitive_equations.py` — explicit tendencies
- `src/notus/timestepping/semi_implicit.py` — implicit terms/inverse (T' fix applied)
- `src/notus/timestepping/imex.py` — IMEX leapfrog driver (verified correct)
- `src/notus/operators.py` — spectral operators (suspect area for `1/a` bookkeeping)
