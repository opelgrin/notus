# Roadmap

Notus is being built incrementally, each phase adding one layer of complexity on top of verified foundations. The design principle is: nothing advances until the current phase passes quantitative validation tests.

## Phase 1 — Spectral Foundations (complete)

Gaussian grid, spherical harmonic transforms, and spectral differential operators.

**What was built:**
- Gaussian grid with Gauss-Legendre quadrature (planet-agnostic, arbitrary truncation)
- Forward/inverse spherical harmonic transforms (FFT in longitude, Legendre in latitude)
- Spectral operators: Laplacian, inverse Laplacian, zonal derivative, meridional derivative (cos(phi) d/dphi via Legendre recurrence)
- Wind reconstruction from vorticity/divergence (streamfunction + velocity potential)
- Spectral curl and divergence operators (integration-by-parts D_mu recurrence)

**Validation:**
- Round-trip transforms exact to machine precision for band-limited fields
- Laplacian eigenvalues verified against analytical Y_n^m
- Solid-body rotation wind reconstruction round-trip
- All operators tested against analytical solutions

## Phase 2 — Shallow Water Equations (complete)

2D shallow water equations on the sphere in vorticity-divergence form. This phase validates the full spectral-transform machinery, nonlinear product computation, and time integration before adding vertical complexity.

**What was built:**
- Pseudo-spectral tendency computation: nonlinear products on the Gaussian grid, spectral operators for tendency assembly (Hoskins & Simmons 1975)
- IMEX leapfrog time stepper with clean separation of explicit (nonlinear) and implicit (gravity wave) terms, following the Dinosaur/NeuralGCM formulation
- Implicit gravity wave treatment: Helmholtz solve diagonal in spectral space via Schur complement
- Exponential spectral filter (Hou & Li 2007) for dealiasing stability near the poles
- Del-8 hyperdiffusion (default, configurable order)
- Robert-Asselin filter for leapfrog computational mode damping

**Key equations (vorticity-divergence form):**
```
dζ/dt = -div(ζ_a · v)              (vorticity)
dδ/dt = +curl(ζ_a · v) - nabla²(Phi + E)  (divergence)
dPhi/dt = -div(Phi · v)              (continuity)
```
where zeta_a = zeta + f (absolute vorticity), Phi = g*h (geopotential), E = kinetic energy.

**Validation — Williamson et al. (1992) test case 2:**
- Balanced solid-body rotation (u0 = 38.61 m/s) maintains steady state to < 1e-4 relative error after 1 day, < 1e-3 after 5 days
- Mass conservation to machine precision (< 1e-10 relative error)
- Stable for 10+ days at dt = 1200s (T21)

**Lessons learned:**
- The vorticity tendency uses -div(flux) and divergence tendency uses +curl(flux) — getting this backwards produces an instability that is invisible to balanced-state tests (where v=0 makes both operators degenerate)
- The 1/cos^2(phi) grid-point multiplication for spectral flux computation creates aliasing at the truncation wavenumber; the exponential filter (not just hyperdiffusion) is essential for stability
- The IMEX formulation (explicit tendencies exclude linear gravity-wave coupling) is cleaner and more stable than the Temperton post-correction approach

## Phase 3 — Primitive Equations (complete)

Full 3D hydrostatic primitive equations on sigma (p/ps) vertical coordinate with Lorenz staggering.

**What was built:**
- Sigma-coordinate vertical discretization (finite differences in the vertical, spectral in the horizontal) with uniform and stretched level options
- Hydrostatic geopotential from temperature via log-sigma vertical integration (Durran §8.6.5)
- Thermodynamic equation with advective-form horizontal advection, vertical advection, and adiabatic heating (κ·T·ω/p)
- Sigma-dot vertical velocity from the continuity equation, surface pressure tendency
- 3D semi-implicit Helmholtz solver: Schur complement reduction to L×L system per spectral mode, eigen-decomposed for O(L²) per-mode performance
- IMEX leapfrog time stepper with backward-forward Euler initialization
- Exponential spectral filter (Hou & Li 2007, matching Dinosaur/NeuralGCM conventions)
- Jablonowski-Williamson (2006) analytic initial conditions (balanced steady state + perturbation)
- Conservation diagnostics: global mass, kinetic/internal/potential energy, angular momentum via Gaussian quadrature and sigma integration

**Key equations (vorticity-divergence form, explicit tendencies):**
```
dζ/dt = -curl(F) + diffusion
dδ/dt = -div(F) - ∇²E - ∇²(g·zs) + diffusion
dT/dt = -v·∇T + σ̇·∂T/∂σ + κ·T·(ω/p) + diffusion
d(ln ps)/dt = -Σ (D + v·∇ln ps)·Δσ
```
where F = (ζ+f)(k̂×v) + σ̇·∂v/∂σ + R·T'·∇ln(ps). Implicit terms (-∇²Φ, -H·δ, -Δσᵀ·δ) handled by the 3D Helmholtz solver.

**Validation — Jablonowski & Williamson (2006) baroclinic instability:**
- Balanced steady state stationary to < 1e-3 divergence after 6 hours (T21 L20)
- 10-day baroclinic wave integration at T42 L20, dt = 600s:
  - Surface pressure minimum deepens to ~950 hPa by day 9 (reference: ~960 hPa)
  - Monotonic deepening during growth phase (days 4-9)
  - Stable for 15+ days with wave breaking and occlusion
- Mass conserved to < 10⁻⁶ relative error over 10 days
- Total energy conserved to < 0.1% over 10 days
- Angular momentum conserved to < 0.1% over 10 days

**Lessons learned:**
- The temperature advection MUST use the advective form (-v·∇T') for consistency with the semi-implicit splitting. The flux divergence form (-∇·(T'v)) differs by T'·δ, which creates a destabilizing feedback in the explicit/implicit coupling that grows exponentially — the balanced state blows up after ~3 days at T42. Adding the +T'·δ correction to convert to advective form completely stabilizes the integration.
- The exponential filter nondimensional timestep must use 2Ω (matching Dinosaur's time scale), and the wavenumber normalization must use T+1 (total_wavenumbers), not T. Getting either wrong changes the filter strength significantly at the truncation wavenumber.
- No vertical diffusion or sponge layer is needed for stability — Dinosaur/NeuralGCM also omit these, relying entirely on the spectral filter and Robert-Asselin filter for damping.

## Phase 4 — Held-Suarez Benchmark (complete)

Newtonian relaxation forcing + Rayleigh friction. The standard dry dynamical core intercomparison.

**What was built:**
- Newtonian temperature relaxation toward a prescribed equilibrium profile T_eq(phi, p) with equator-to-pole gradient and stratospheric temperature floor (200 K)
- Latitude- and level-dependent relaxation rate k_T (k_a in free atmosphere, enhanced to k_s at equatorial surface)
- Rayleigh friction k_v in the planetary boundary layer (sigma > 0.7), applied directly in spectral space (equivalent to velocity-space friction since k_v is level-only)
- Pluggable forcing via `Forcing` protocol, composed with dynamics in `build_pe_stepper`
- Isothermal rest-state initial conditions (264 K + random perturbation) for symmetry breaking
- Zonal-mean diagnostic framework (`ZonalMeanState`, `compute_zonal_mean_state`) for climatological analysis
- Benchmark validation script with 10 two-sided climatological checks
- Five-panel diagnostic visualization (zonal-mean U, T, EKE; spinup timeseries; surface pressure map)

**Validation — Held & Suarez (1994) climatology (T21 L20, 300 days):**
- Subtropical jets: ~32 m/s at ~40° latitude (reference: 25-30 m/s at ~30°)
- Surface westerlies: ~8.5 m/s in midlatitudes
- Equatorial surface temperature: ~309 K, polar surface: ~264 K (ΔT ≈ 44 K)
- Cold tropopause: ~196 K
- Midlatitude eddy kinetic energy: storm tracks in both hemispheres
- Near-perfect hemispheric symmetry (NH/SH jet ratio ≈ 1.0)

**Lessons learned:**
- Rayleigh friction on vorticity/divergence can be computed directly in spectral space (-k_v * zeta, -k_v * delta) when k_v depends only on sigma level — this is mathematically equivalent to Dinosaur's velocity-space approach but saves two spectral/grid round-trips per timestep
- The forcing protocol composing with explicit dynamics via `jax.tree.map(add, dyn_tend, phys_tend)` inside the JIT boundary is clean and adds negligible overhead
- 100 days of spinup is sufficient for the gross climate features to develop at T21; the full 1200-day T42 benchmark uses 200-day spinup per H&S convention

## Phase 5 — Simple Physics (complete)

Gray radiation, dry convective adjustment, and bulk surface flux — following Frierson et al. (2006) for an aquaplanet configuration.

**What was built:**
- Two-stream gray longwave radiation with latitude-dependent optical depth (tau_e=6.0, tau_p=0.1, mixed linear+p^4 pressure dependence), upward/downward fluxes via `jax.lax.scan`
- Dry convective adjustment (Manabe-Strickler 1964): bottom-up pairwise adjustment conserving column enthalpy, expressed as a relaxation tendency (tau=12h) for leapfrog stability
- Bulk aerodynamic surface sensible heat flux: H = rho * cp * C_D * |v| * (T_s - T_a) with constant drag coefficient C_D=0.0015
- Prescribed SST: Frierson Gaussian profile T_s(phi) = 271 + 29 exp(-0.5(phi/26deg)^2)
- Rayleigh boundary-layer drag (same formulation as Held-Suarez)
- No atmospheric shortwave absorption (Frierson convention: SW heats the surface only)
- `SimplePhysicsConfig` dataclass for all scheme parameters
- `SimplePhysics` forcing implementing the `Forcing` protocol (drop-in alternative to Held-Suarez)
- Aquaplanet example script and five-panel diagnostic visualization

**Validation:**
- 32 unit tests covering radiation, convection, SST, surface flux, and forcing protocol
- 10-day integration stability test at T21 L20
- All tests pass with 253 total tests across the project

**Lessons learned:**
- Convective adjustment expressed as (T_adj - T)/dt blows up with leapfrog timestepping because the 2dt effective timestep causes a 2x overcorrection. Using a finite relaxation timescale tau_adj >> 2dt (12 hours) stabilizes the scheme. This is a fundamental leapfrog limitation — IMEX Runge-Kutta schemes (e.g., Dinosaur/NeuralGCM) do not have this problem.
- Surface sensible heat flux is essential for realistic temperatures. Without it, the atmosphere is ~40 K too cold because LW radiation alone cannot efficiently couple the warm surface to the boundary layer.
- Frierson (2006) uses no atmospheric SW absorption — the shortwave heats the surface only, and heat enters the atmosphere through sensible flux and LW radiation.

## Phase 6 — Moisture (complete)

Add water vapor as a prognostic tracer with moist physics parameterizations.

**What was built:**
- Specific humidity as an optional field in `PrimitiveEquationState`, advected spectrally with the same flux-divergence + advective correction treatment as temperature
- Zero implicit terms (fully explicit), pass-through in semi-implicit solver
- Moisture thermodynamics: saturation vapor pressure (Bolton 1980), saturation specific humidity, moist pseudoadiabatic lapse rate
- Large-scale condensation: implicit Frierson (2006) eq. 21 scheme, iterative, energy-conserving per level (cp·ΔT + L·Δq = 0)
- Simplified Betts-Miller convection (Frierson 2007): parcel ascent with level of zero buoyancy (LZB), deep/shallow distinction via Pq/PT integrals, qref formulation for shallow convection, enthalpy-conserving ΔT offset applied only within the convective column
- Bulk aerodynamic surface latent heat flux (evaporation)
- `SimplePhysics` extended with automatic moist/dry pathway selection
- `moist_aquaplanet_initial_state` with RH-based humidity profile
- Grid-space humidity clipping after each time step to prevent accumulation of negative values from spectral Gibbs ringing (see lessons learned)
- Backward-compatible: dry states (humidity=None) work identically to Phase 5

**Validation:**
- 1200-day moist aquaplanet integration stable at T21 L20
- Equilibrium reached by ~day 300: q_mean ≈ 2.6 g/kg, T_mean ≈ 249 K
- Negative humidity bounded at 0.2–0.4% with grid-space clipping
- Temperature range 207–305 K, humidity 0–24 g/kg, surface pressure 969–1021 hPa
- Climatology: T_equator=300 K, T_pole=269 K, jet max=87 m/s, EKE max=83 m²/s²
- 49 new tests (303 total): thermodynamics, surface flux, condensation, BM convection, passive tracer transport, config validation, 10-day integration stability

**Lessons learned:**
- Betts-Miller convection must be vertically bounded by the level of zero buoyancy. Applying relaxation tendencies to the full column (including the stratosphere where the moist adiabat reference is meaningless) causes rapid temperature drift and blowup within days. This is universal across GCMs: both SpeedyWeather.jl (Frierson SBM) and SPEEDY (Tiedtke) limit tendencies to between the surface and the convection top.
- The implicit condensation denominator factor is L²ε²/(cp·R_d·T²), not L²ε/(cp·R_v·T²) computed from a reconstructed R_v. Using R_v = R_d/ε and simplifying algebraically avoids an intermediate variable that is easy to get wrong.
- Initializing humidity from q = RH × q_sat(T, p) on an isothermal atmosphere requires capping q_sat at the surface value, because q_sat diverges at low pressures when temperature is constant.
- Spectral Gibbs ringing creates negative humidity at sharp moisture gradients. Without clipping, negatives grow to 18%+ of grid points at L20 after 1200 days, biasing q_mean low by ~0.05 g/kg. Grid-space clipping in the time stepper (transform → clip → re-transform after each step) corrects the spectral representation itself and keeps negatives bounded at 0.3%. Clipping before physics alone is insufficient — it doesn't modify the spectral state, so negatives persist and accumulate.
- Quantitative moisture budget analysis (tracking the spectral (0,0) mode) showed the initial q_mean drift is dominated by the physics E-P imbalance (precipitation exceeds evaporation during spinup from RH=0.7 initial condition), not numerical sinks. Hyperdiffusion, spectral filter, and Robert-Asselin filter all contribute effectively zero to the global mean moisture tendency. The system equilibrates after ~300 days.

## Phase 6b — Virtual Temperature + Stability Consolidation (complete)

Virtual temperature in the semi-implicit solver, plus implicit treatment of stiff physics terms. This phase solidified the numerical foundations before adding further physics complexity.

**What was built:**

*Virtual temperature in the semi-implicit solver:*
- The implicit solver now linearizes around `T_v_ref = T_ref·(1 + ε_v·q_ref)` when a reference humidity profile is provided
- Geopotential weights scaled to `G_v = G·diag(1 + ε_v·q_ref)` so the implicit geopotential uses virtual temperature
- Coupling matrix `M_v = G_v·H + R·T_v_ref⊗Δσ` properly captures moisture contribution to pressure gradient
- H matrix stays dry (relates divergence to prognostic temperature tendency — correct)
- Explicit pressure gradient uses smaller residual `T_v - T_v_ref` instead of `T_v - T_ref`
- Pure dynamics stable to dt=1200+ at T21 L20

*Implicit surface fluxes:*
- Surface sensible heat flux, latent heat flux, and Rayleigh friction treated with backward Euler: `X_new = (X + dt/τ · X_ref) / (1 + dt/τ)`
- Unconditionally stable regardless of wind speed, drag coefficient, or dt
- Applied after the IMEX step via `SimplePhysics.apply_implicit(state, dt_implicit)`
- `build_pe_stepper` accepts `implicit_physics` callable for operator splitting

*Implicit Betts-Miller convection:*
- BM reference profile (moist adiabat, LZB, deep/shallow) computed explicitly at post-step state
- Relaxation toward the reference applied with backward Euler scaling: `dt / (1 + dt/τ_bm)`
- Unconditionally stable regardless of dt/tau_bm ratio
- Eliminates the computational mode excitation that previously limited dt

*All implicit physics enabled by default* via `SimplePhysicsConfig(implicit_surface=True)`. Explicit path retained for debugging and comparison.

**What was NOT implemented (and why):**
- Vorticity correction and moist κ: ~3% combined effect, deferred.
- RAW filter (Williams 2009): implemented and tested, but destabilizes moist runs because remaining explicit physics (radiation, condensation) still excites the computational mode. Removed — would need all physics implicit or a non-leapfrog integrator.

**Validation:**
- 303 existing tests pass (backward compatible)
- Moist aquaplanet stable at dt=1200s (default) for 50+ days, tested to dt=2400s for 950+ days
- Dry Held-Suarez and baroclinic wave tests unaffected
- Climatology at dt=1200: T_equator=293 K, T_pole=268 K, jet max=33 m/s, EKE=81 m²/s²

**Lessons learned:**
- The original Phase 6b only put T_v in the explicit pressure gradient, leaving `ε_v·q·T·∇lnps` as a fully unresolved fast mode. Moving T_v into the semi-implicit solver (G_v, T_v_ref) was essential for dynamics stability.
- The moist physics stability bottleneck was NOT surface fluxes (as initially assumed) but Betts-Miller convection exciting the leapfrog computational mode. BM's sharp on/off activation (CAPE threshold) creates large `x_{n-1} - 2·x_n + x_{n+1}` terms that grow without implicit treatment.
- Implicit treatment of relaxation terms (`(X_ref - X)/τ`) is trivial: just scale the explicit tendency by `1/(1 + dt/τ)`. No iteration needed.
- The Robert-Asselin-Williams (RAW) filter was harmful for moist runs: it feeds computational mode energy back into the future state, amplifying convective noise. Standard RA with implicit physics is more stable than RAW with explicit physics.
- Implicit physics adds negligible cost per step (one extra grid↔spectral round-trip for BM), but dt doubles, so net wallclock improves ~2x.

## Phase 7 — Seasonal Cycle + Slab Ocean

Two-band radiation with water vapor feedback, annual cycle, and interactive slab ocean.

**Plan:**
- Solar geometry: orbital parameters (obliquity, eccentricity, longitude of perihelion), daily-mean insolation with annual cycle
- Two-band radiation: Byrne/Isca LW with humidity-dependent optical depth (water vapor feedback), activate SW atmospheric absorption (Beer-Lambert)
- Slab ocean: mixed-layer ocean with prescribed Q-flux, replacing prescribed SST

## Phase 8 — Surface Coupling + Diurnal Cycle

Land surface, boundary layer, and diurnal cycle.

**Plan:**
- Diurnal cycle (instantaneous solar zenith angle, time-of-day dependent insolation)
- Simple land surface model (bucket hydrology, surface energy balance)
- Monin-Obukhov boundary layer parameterization (stability-dependent drag, prognostic BL depth)
- Upgrade from constant C_D to full surface similarity theory
- Sea ice thermodynamics (optional)
