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

## Phase 3 — Primitive Equations (next)

Extend to 3D with sigma (p/ps) vertical coordinate.

**Plan:**
- Sigma-coordinate vertical discretization (finite differences in the vertical, spectral in the horizontal)
- Hydrostatic equation: geopotential from temperature via vertical integration
- Thermodynamic equation: temperature tendency with adiabatic heating
- Vertical advection using the continuity equation for sigma-dot
- Semi-implicit treatment extended to the 3D Helmholtz problem (vertical coupling makes it tridiagonal per spectral mode)
- Vertical diffusion / sponge layer near model top

**Validation targets:**
- 3D solid-body rotation (Jablonowski & Williamson 2006 test case)
- Baroclinic wave development from analytic initial conditions
- Conservation of mass, energy, angular momentum

## Phase 4 — Held-Suarez Benchmark

Newtonian relaxation forcing + Rayleigh friction. The standard dry dynamical core intercomparison.

**Plan:**
- Newtonian temperature relaxation toward a prescribed equilibrium profile (equator-to-pole gradient, stratospheric cap)
- Rayleigh friction in the planetary boundary layer (lowest ~700 hPa)
- 1200-day integration at T42 L20 (after 200-day spinup)

**Validation targets (Held & Suarez 1994):**
- Zonal-mean zonal wind: subtropical jets at ~30 deg, ~30 m/s
- Zonal-mean temperature: realistic tropospheric lapse rate, tropopause height
- Eddy kinetic energy: midlatitude storm tracks
- Meridional heat and momentum fluxes

## Phase 5 — Simple Physics

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
