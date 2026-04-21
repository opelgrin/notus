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
- `PhysicsSuiteConfig` dataclass for all scheme parameters
- `PhysicsSuite` forcing implementing the `Forcing` protocol (drop-in alternative to Held-Suarez)
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
- `PhysicsSuite` extended with automatic moist/dry pathway selection
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
- Applied after the IMEX step via `PhysicsSuite.apply_implicit(state, dt_implicit)`
- `build_pe_stepper` accepts `implicit_physics` callable for operator splitting

*Implicit Betts-Miller convection:*
- BM reference profile (moist adiabat, LZB, deep/shallow) computed explicitly at post-step state
- Relaxation toward the reference applied with backward Euler scaling: `dt / (1 + dt/τ_bm)`
- Unconditionally stable regardless of dt/tau_bm ratio
- Eliminates the computational mode excitation that previously limited dt

*All implicit physics enabled by default* via `PhysicsSuiteConfig(implicit_surface=True)`. Explicit path retained for debugging and comparison.

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

## Phase 7 — Seasonal Cycle + Slab Ocean (complete)

Two-band radiation with water vapor feedback, annual cycle, and interactive slab ocean.

**What was built:**
- Solar geometry: orbital parameters (obliquity, eccentricity, longitude of perihelion), daily-mean insolation with annual cycle
- Two-band radiation: Byrne/Isca LW with humidity-dependent optical depth (water vapor feedback), SW atmospheric absorption (Beer-Lambert)
- Slab ocean: mixed-layer ocean with prescribed Q-flux, replacing prescribed SST
- Q-flux diagnostic utility for computing implied ocean heat transport from prescribed SST equilibrium

## Phase 8A — Monin-Obukhov Surface Layer (complete)

Stability-dependent surface fluxes replacing the constant drag coefficient, plus several slab ocean coupling improvements discovered during validation.

**What was built:**

*Monin-Obukhov surface layer:*
- Louis (1979) stability functions: bulk Richardson number, analytic correction factors for momentum and heat transfer coefficients
- Neutral coefficients from log-profile: `C_DN = (k/ln(z/z0))²`, separate z0 for momentum and heat
- `SurfaceLayerConfig` dataclass: roughness lengths, Louis parameters, Ri clamp
- Wired into `PhysicsSuite` (explicit and implicit paths) and coupled slab ocean stepper

*Slab ocean coupling improvements:*
- Real downward LW flux from the two-stream radiation solver (`lw_down_surface()`), replacing crude `σT⁴(1-exp(-0.5))` approximation that underestimated LW_down by 60%
- Implicit slab ocean step: linearized Newton-style `T_new = T_old + dt·F/(C - dt·dF/dT_s)` for unconditional SST stability at any timestep
- Forcing SST sync: coupled driver updates `forcing.sst` with ocean SST each day so explicit LW radiation uses the current ocean temperature (analogous to `forcing.day_of_year` pattern)

*Warm-start infrastructure:*
- `spinup_prescribed_sst()` in `timestepping/spinup.py`: runs prescribed-SST spinup and Q-flux diagnosis in memory, returning a `SpinupResult(state, q_flux, ...)` ready to feed directly into `build_coupled_pe_stepper`. No disk I/O required.
- `save_restart()` / `load_restart()` in `initial_conditions.py` for optional disk-based restart files (useful when spinup and coupled runs are separate CLI invocations)

**Validation:**
- 28 unit tests for boundary layer module (362 total, all passing)
- 100-day slab ocean aquaplanet stable with MO at dt=900 (warm start + diagnosed Q-flux)
- MO produces physically correct differences from baseline: weaker surface fluxes (C_H≈0.0007 vs 0.0015), warmer equatorial SST, more moisture, stronger jets
- `verify_monin_obukhov.py` is fully self-contained: runs spinup + Q-flux diagnosis + baseline + MO comparison with zero external files

**Lessons learned:**
- Cold-starting a coupled slab ocean integration from an isothermal atmosphere creates violent radiative transients (T spike >500 K within days). The proper procedure is `spinup_prescribed_sst()` followed by coupled mode — standard practice in real GCMs but easy to forget in an idealized model.
- The one-line LW_down approximation `σT_lowest⁴(1-exp(-0.5))` gives ~120 W/m² instead of ~300 W/m² from the actual two-stream solver. The approximation was self-consistent (Q-flux diagnosis and slab ocean used the same formula), but distorted the surface energy budget. With the real LW_down, the diagnosed Q-flux has a -145 W/m² global mean, reflecting the simplified radiation scheme's inherent energy imbalance — this is physically correct (the scheme doesn't conserve energy globally), not a bug.
- The slab ocean surface energy balance `C·dT/dt = F(T_s)` is stiff because `dF/dT_s ≈ -6 W/m²/K` (from `4σT³` alone). Forward Euler overshoots at dt=900s when net fluxes are large. Linearized implicit stepping eliminates this constraint.
- Side effects on `forcing` attributes (sst, day_of_year) cannot happen inside `jax.jit` — they cause tracer leaks. These must be set by the driver loop outside JIT, between scan calls.

## Phase 8B — Surface Type Infrastructure (complete)

Per-gridpoint surface properties and idealized land-sea mask generators.

**What was built:**
- `SurfaceProperties` dataclass (JAX pytree): land_fraction, albedo, z0_momentum, z0_heat — all (n_lat, n_lon)
- `aquaplanet_surface()`: uniform ocean surface (default)
- `flat_continent_surface()`: rectangular continent with land/ocean blended properties
- Default physical constants: ocean (z0=1e-4, albedo=0.06), land (z0=0.05, albedo=0.25)
- `build_coupled_pe_stepper` accepts `surface_properties` for spatially varying albedo and roughness

**Validation:**
- 20 unit tests for surface properties, mask generators
- 4 integration tests: aquaplanet and flat continent 10-day coupled runs, continent cooler than aquaplanet, SST in physical range

## Phase 8C — Bucket Land Surface Model (complete)

Frierson (2006) / Manabe (1969) single-layer soil energy balance with bucket hydrology, evaporation resistance, and blended ocean-land fluxes.

**What was built:**

*Bucket land model:*
- `BucketLandConfig`: soil heat capacity (4×10⁶ J/(m²·K), ~2 m moist soil), bucket capacity (0.15 m), beta parameters, moisture-dependent albedo option
- `LandState` (JAX pytree): soil_temperature (n_lat, n_lon), bucket_depth (n_lat, n_lon)
- `SurfaceState`: wraps OceanState + optional LandState for clean coupled stepper signature
- `beta_function`: evaporation resistance β = min(1, W/W_crit) — linear ramp from dry (no evaporation) to saturated (unlimited)
- `compute_net_land_flux`: F_net = SW + LW_down − σT⁴ − H − β·L·E_pot, returns both net flux and evaporation rate
- `land_flux_derivative`: dF/dT_land for linearized implicit stepping
- `step_land_implicit`: same linearized Newton scheme as slab ocean, unconditionally stable
- `step_bucket_hydrology`: dW/dt = P − E, clamped to [0, W_max] (overflow = runoff)
- `diagnose_precipitation`: column-integrated moisture sink from Betts-Miller convection + large-scale condensation
- `moisture_dependent_albedo`: Frierson (2006) α = α_wet + (α_dry − α_wet)·(1 − W/W_max), blended with ocean albedo via land_fraction

*Coupled stepper:*
- `build_coupled_pe_stepper` accepts `land_config`, uses `SurfaceState` (backward compatible: land=None → ocean-only path)
- Land+ocean path: separate energy balances, precipitation diagnostic, blended implicit atmospheric decay toward (1−f)·SST + f·T_land
- Monin-Obukhov `compute_transfer_coefficients` accepts spatially varying z0 overrides from SurfaceProperties

**Validation:**
- 37 unit tests for all new functions (pytree round-trip, beta, fluxes, stepping, hydrology, albedo, precipitation)
- 4 integration tests: 10-day coupled land-ocean run stable, SST/T_land/bucket in physical ranges
- All 397 non-slow tests pass, all 8 slow integration tests pass (including ocean-only regression)

**Lessons learned:**
- The implicit atmospheric decay at the lowest level MUST be applied at land points (decaying toward T_land), not just ocean. Without it, the lowest-level temperature at land points is unconstrained and creates dynamical instability from large air-surface temperature contrasts within 2-3 days.
- Soil heat capacity of 1×10⁶ J/(m²·K) (thin dry soil) is too low for stability at T21 with dt=900s — the land heats rapidly when the bucket drains and evaporative cooling vanishes. Default of 4×10⁶ (~2 m moist soil) provides stable integration while maintaining realistic diurnal/synoptic response.
- The explicit LW radiation in `PhysicsSuite.__call__` uses `self.sst` as the surface emission boundary. For coupled land runs, this means LW radiation over land uses the ocean SST rather than T_land. The error is modest (~10 W/m² for a 15 K difference) because the land energy balance in the coupled post-step uses the correct T_land. A future improvement would pass the blended surface temperature through `forcing.sst`, but this requires updating it every timestep (not just per-day), which is incompatible with `jax.lax.scan`.
- The bucket drains significantly over 10 days (0.11 → 0.01 m) as evaporation exceeds precipitation during the cold-start transient. In equilibrium, the precipitation-evaporation balance should maintain the bucket near its critical depth.

## Phase 9 — Radiation Upgrade (complete)

Replaced the semi-gray Byrne scheme (which had a -145 W/m² global energy imbalance) with a SPEEDY-style multi-band radiation scheme, plus diagnostic clouds. Prioritized over topography because radiation balance is a prerequisite for meaningful coupled experiments.

**What was built:**

*Tier 1 — SW energy conservation:*
- `shortwave_heating()` returns `(heating_rate, sw_down_surface)` tuple — single source of truth for surface SW, eliminating the independent recomputation that caused the imbalance
- Surface energy balance functions (`compute_net_surface_flux`, `compute_net_land_flux`) accept the consistent `sw_down_surface` from the radiation solver
- Reflected upward beam undergoes Beer-Lambert absorption on the return pass (double-pass SW)
- Humidity-dependent SW optical depth: `byrne_shortwave_optical_depth()` with `dτ/d(p/p₀) = a_sw + b_sw·q`
- SW column closure exact to machine precision (verified by tests)

*Tier 2 — SPEEDY multi-band radiation (Molteni 2003):*
- 4-band LW with temperature-dependent band fractions (`radset` formula):
  - Band 0 (window): `ablwin=0.3` (dry air only)
  - Band 1 (CO₂): `ablco2=6.0` (well-mixed, tunable for climate sensitivity)
  - Band 2 (H₂O weak): `ablwv1=0.7·q`
  - Band 3 (H₂O strong): `ablwv2=50.0·q`
  - Band fractions shift with temperature — warmer surfaces emit more in the window band where the atmosphere is transparent, providing a critical negative feedback
- 2-band SW:
  - Band 1 (visible, 95%): dry air + aerosol(σ²) + weak H₂O (`abswv1=0.022`)
  - Band 2 (near-IR, 5%): strong H₂O (`abswv2=15.0`, 680× stronger)
- Two-stream LW via `jax.vmap` over bands for clean parallelization
- Surface emissivity < 1 (`emisfc=0.98`)
- Returns OLR for diagnostics

*Diagnostic cloud scheme:*
- RH-based cloud cover in free troposphere + precipitation contribution
- Stability-dependent stratiform clouds at PBL top (dry static energy gradient)
- SW: cloud reflection at cloud top (`albcl=0.43`) + stratiform reflection (`albcls=0.50`) + cloud absorption in visible band
- LW: thick cloud absorption (`ablcl1=12.0`) below cloud top in window band, thin cloud absorption (`ablcl2=0.6`) in window + H₂O bands above
- `CloudConfig` dataclass with SPEEDY defaults, `enable_clouds=True/False` flag

*Forcing protocol cleanup:*
- `build_coupled_pe_stepper` and `spinup_prescribed_sst` now take `PhysicsSuite` directly instead of the `Forcing` protocol with `hasattr` guards and `Any` casts
- Removed ~30 lines of defensive checks that obscured the actual requirements
- `build_pe_stepper` retains `Forcing` protocol (genuinely works with HeldSuarez)

**Validation (prescribed-SST, 300 days at T21 L20):**

| Metric | Byrne | SPEEDY clear | SPEEDY + clouds |
|--------|-------|-------------|-----------------|
| Net TOA flux | +109 W/m² | +3 W/m² | -7 W/m² |
| OLR | 218 | 318 | 328 |
| Greenhouse effect | 163 | 63 | 54 |
| Planetary albedo | 3.9% | 5.5% | 5.4% |
| H₂O SW feedback | 0.2 | 1.5 | 2.3 |
| SW column closure | exact | exact | exact |

- SPEEDY dramatically improves energy balance: +3 W/m² vs +109 W/m² (Byrne)
- Clouds add 16 W/m² LW greenhouse, warming atmosphere by 12 K
- Coupled slab ocean stable with SPEEDY at dt=900 (where Byrne land-ocean blows up)

**Lessons learned:**
- The 4-band LW with temperature-dependent fractions is the key to energy balance. The window band shifts emission toward transparent wavelengths at warm surface temperatures, providing OLR ≈ absorbed SW without fine-tuning. Single-band schemes cannot achieve this because they have no spectral degree of freedom.
- The SPEEDY near-IR H₂O absorption (`abswv2=15.0`) is 680× stronger than visible (`abswv1=0.022`). Nearly all humidity-dependent SW absorption happens in the near-IR band, which is only 5% of solar irradiance. A single-band SW scheme with `byrne_sw_b=0.2` vastly underestimates this effect.
- Diagnostic clouds from RH alone (without precipitation) still provide meaningful LW greenhouse effect. SW cloud albedo requires the precipitation contribution for realistic values — this will improve when precipitation is threaded through from the moist physics.
- The `Forcing` protocol was too narrow for the coupled stepper, which needs radiation config, SST, day_of_year, k_v, and implicit physics. Typing as `PhysicsSuite` directly is more honest and eliminates fragile duck-typing.

## Phase 10 — Topography (complete)

Prescribed orography and its dynamical/physical effects.

**What was built:**
- Idealized topography generators in `topography.py`: `gaussian_mountain` (isolated bell), `zonal_ridge` (zonally symmetric), `sinusoidal_mountains` (wavenumber-k chain for stationary Rossby wave tests)
- Spectral smoothing of orography via `smooth_orography()`: Lanczos σ-factor and exponential taper methods with configurable order, applied once to initial surface geopotential to suppress Gibbs ringing
- Hydrostatic surface pressure initialization via `orographic_log_surface_pressure()`: computes ln(ps/p₀) = −Φ_s/(R·T_ref) in grid space and transforms to spectral, ensuring pressure field is consistent with terrain
- All functions return spectral arrays drop-in compatible with `build_pe_stepper`

**Validation:**
- 25 unit tests: roundtrip height recovery, peak location, zonal symmetry (spectral m=0 check), wavenumber structure via FFT, polar vanishing, global mean, smoothing properties (global mean preservation, high-wavenumber damping, Gibbs reduction, order monotonicity)
- 8 dry integration tests: 5-day Gaussian and sinusoidal mountain runs with stability, temperature bounds, surface pressure reduction over peaks, mass conservation (<0.01%), energy conservation (<1%)
- 6 moist integration tests: 10-day moist aquaplanet with mountain, confirming precipitation develops, is non-negative, and is spatially modulated by the mountain (zonal symmetry broken)
- All 443 non-slow tests pass

**Orographic precipitation — resolved, no parameterization needed:**
Surveying comparable idealized GCMs (Isca, GFDL idealized moist, PlaSim, SPEEDY/JCM), the standard approach is to rely on resolved dynamics for orographic precipitation. The ∇²(g·z_s) divergence tendency plus sigma-coordinate lifting naturally produces spatially varying precipitation patterns around mountains. SPEEDY's "orographic correction" (lapse-rate T/q adjustment) is SPEEDY-specific tuning, not standard practice. Our moist integration confirms the resolved dynamics are sufficient: precipitation develops with clear spatial modulation by the mountain without explicit orographic physics.

**Lessons learned:**
- Spectral representation of sharp topography requires explicit smoothing beyond the inherent truncation. The Lanczos σ-factor (sinc taper) is simple, preserves the global mean exactly (σ(n=0)=1), and effectively suppresses Gibbs undershoots.
- Surface pressure initialization is essential — starting with uniform ps over a 2500 m mountain creates an immediate hydrostatic imbalance that generates spurious gravity waves. The hypsometric adjustment is a simple one-liner but critical for clean integrations.
- At T21 resolution, a 2500 m Gaussian mountain (half-width 20°) is well-resolved and the model remains stable for 10+ days in both dry and moist configurations with dt=600s.

## Phase 11A — Sea Ice Thermodynamics (complete)

Ice fraction, ice temperature, albedo feedback, and freezing/melting coupled to the slab ocean.

**What was built:**
- Zero-layer or single-layer Semtner thermodynamic ice model
- Ice fraction as a prognostic variable per grid point
- Freezing criterion: ocean temperature reaches freezing point (−1.8 °C for seawater)
- Ice growth/melt from surface energy balance and ocean heat flux
- Ice albedo feedback: high albedo over ice (∼0.65), blended with ocean albedo by ice fraction
- Ice insulation: suppresses turbulent fluxes between ocean and atmosphere
- Ice thickness evolution with conductive heat flux through ice
- Lead fraction (open water within ice): controls ocean–atmosphere exchange in partially ice-covered cells
- Implicit ice temperature stepping (consistent with existing slab ocean scheme)
- Integration into `CoupledStepper` ocean path and `SurfaceState`

**Validation:**
- Dedicated sea-ice unit + integration tests in `tests/test_sea_ice.py`
- Coupled 10-day sea-ice-ocean runs remain stable with finite SST / ice thickness / ice fraction
- SST remains clamped near freezing where ice is present and equatorial band stays ice-free
- API + xarray/restart I/O surface model wiring updated to carry sea-ice state in `SurfaceState` and persistence paths

## Phase 11B — Snow Cover

Snow accumulation and melt on land (and optionally sea ice), with albedo feedback and insulation.

**Planned:**
- Snow depth as a prognostic variable on land surface (extension to `LandState`)
- Snow accumulation from precipitation when surface temperature < 0 °C
- Snowmelt from surface energy balance, meltwater feeds bucket hydrology
- Snow albedo feedback: high albedo over snow-covered land (∼0.8), blended by snow fraction
- Snow insulation: reduces soil heat loss, decouples soil temperature from atmosphere
- Snow fraction parameterization (e.g., from snow depth / critical depth)
- Interaction with bucket hydrology (melt → soil moisture, sublimation)
- Optional snow on sea ice (builds on Phase 11A)

## Phase 12+ — Future Wishlist

Optional extensions for further realism.

**Candidates:**
- RRTMGP (aspirational): correlated-k method via pyrrtmgp or a JAX port for state-of-the-art accuracy
- Diurnal cycle (instantaneous solar zenith angle, time-of-day dependent insolation)
- Vegetation / land surface complexity (canopy, root zone, stomatal resistance)
- Multi-layer soil model
- Semi-Lagrangian moisture advection (removes CFL timestep constraint, enables shape-preserving transport without spectral Gibbs ringing)
