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

## Phase 5 — Simple Physics (next)

Gray radiation and dry convective adjustment — the minimum physics needed for a self-consistent climate.

**Plan:**
- Gray-atmosphere longwave radiation (optical depth as a function of pressure)
- Shortwave absorption (Beer-Lambert with prescribed solar constant)
- Dry convective adjustment (restore unstable profiles to dry adiabat)
- Surface energy balance (prescribed SST or slab ocean)

## Phase 6 — Moisture

Add water vapor as a tracer with large-scale condensation.

**Plan:**
- Specific humidity as a prognostic variable with spectral advection
- Large-scale condensation when supersaturated (with latent heating feedback)
- Simple convection scheme (Betts-Miller or simplified Arakawa-Schubert)
- Precipitation diagnostic

## Phase 7 — Seasonal Cycle

Time-varying solar forcing for realistic seasonal behavior.

**Plan:**
- Orbital parameters: obliquity, eccentricity, longitude of perihelion
- Diurnal cycle (rotating solar zenith angle)
- Annual cycle (varying declination angle)
- Two-band (shortwave + longwave) radiation with water vapor feedback

## Phase 8 — Surface Coupling

Interactive surface for land-ocean-atmosphere coupling.

**Plan:**
- Slab ocean with prescribed ocean heat transport (Q-flux)
- Simple land surface model (bucket hydrology, surface energy balance)
- Planetary boundary layer parameterization (bulk aerodynamic formulas)
- Sea ice thermodynamics (optional)
