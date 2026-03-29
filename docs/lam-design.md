# Notus-LAM: Limited-Area Model Design Document

A nested limited-area model (LAM) for regional atmospheric simulation, driven by boundary conditions from the Notus global spectral GCM. Built from scratch in Python/JAX.

## Motivation

Global spectral models like Notus are efficient for planetary-scale circulation but impractical at the resolutions needed for regional weather and mesoscale phenomena (convective systems, orographic flows, land-sea breezes). A limited-area model solves the equations of motion on a small domain at high resolution, using the global model to supply lateral boundary conditions.

Notus-LAM complements Notus the same way operational LAMs (COSMO, HARMONIE, WRF) complement global models (IFS, GFS) — one-way nesting from coarse global to fine regional.

## Architecture

### Grid: rotated latitude-longitude, Arakawa C-grid

The LAM uses a regular lat-lon grid with a **rotated pole**, placed so the domain center sits on the rotated equator. This avoids meridian convergence within the domain and keeps all grid arrays rectangular — natural for JAX.

Variables are staggered on an **Arakawa C-grid**:

```
     v(i,j+1)
       |
u(i,j)--T(i,j)--u(i+1,j)
       |
     v(i,j)
```

- **T-points** (cell centers): temperature, pressure, humidity, geopotential
- **u-points** (east cell edges): zonal wind
- **v-points** (north cell edges): meridional wind

C-grid staggering gives second-order accurate gradient/divergence operators for free and produces correct gravity wave dispersion (no computational modes that plague the A-grid at mesoscale resolution).

The rotation from geographic (lambda, phi) to rotated (lambda_r, phi_r) coordinates is a rigid rotation of the sphere. It only appears in:
1. The Coriolis parameter f(phi) — computed from geographic latitude, not rotated latitude
2. Boundary condition interpolation — mapping Notus Gaussian grid points to rotated coordinates
3. Solar geometry — if physics needs true latitude

All dynamics and physics operate in the rotated frame.

### Vertical coordinate: hybrid sigma-pressure

```
p(k, i, j) = A(k) * p_ref + B(k) * p_s(i, j)
```

- Near surface: A ≈ 0, B ≈ 1 (terrain-following, resolves boundary layer over topography)
- Near model top: A → p_top/p_ref, B → 0 (flat pressure surfaces, no terrain imprint)

This is a strict generalization of Notus's pure sigma coordinate (A=0, B=sigma). Notus currently uses pure sigma levels (`SigmaLevels`); if/when it gains hybrid levels, the vertical discretization can be shared directly.

Standard A/B coefficient sets exist in the literature (ECMWF L60/L91, or simpler versions with fewer levels). The design should accept arbitrary A(k), B(k) arrays.

### Dynamics: finite difference primitive equations

The prognostic variables are **u, v, T, q, ln(ps)** — wind components rather than vorticity/divergence, since finite differences on a limited domain cannot use the spectral vorticity-divergence formulation.

Horizontal momentum equations (flux form):
```
du/dt = -u ∂u/∂x - v ∂u/∂y - sigma_dot ∂u/∂sigma - (1/rho) ∂p/∂x + f*v + F_u
dv/dt = -u ∂v/∂x - v ∂v/∂y - sigma_dot ∂v/∂sigma - (1/rho) ∂p/∂y - f*u + F_v
```

Thermodynamic equation:
```
dT/dt = -u ∂T/∂x - v ∂T/∂y - sigma_dot ∂T/∂sigma + kappa*T*(omega/p) + Q
```

Continuity (surface pressure tendency):
```
d(ln ps)/dt = -integral[ div(v) + v . grad(ln ps) ] d_sigma
```

Moisture:
```
dq/dt = -u ∂q/∂x - v ∂q/∂y - sigma_dot ∂q/∂sigma + E - C
```

### Advection

Horizontal advection is the most critical numerical component. Options in order of complexity:

1. **Centered 2nd-order** — simplest, but dispersive (Gibbs-like oscillations near sharp gradients). Acceptable for a first implementation.
2. **PPM (Piecewise Parabolic Method)** — 3rd-order, monotone with limiters. The standard for production LAMs. Moderate implementation effort.
3. **WENO (Weighted Essentially Non-Oscillatory)** — 5th-order, handles shocks. More complex, diminishing returns for smooth atmospheric fields.

Recommendation: start with centered 2nd-order for the initial phases, upgrade to PPM when adding moisture/topography where sharp gradients matter.

Vertical advection uses the same scheme applied in the sigma direction, with sigma_dot computed from the continuity equation (same as Notus).

### Time stepping

**IMEX (Implicit-Explicit)** splitting, similar in spirit to Notus:

- **Explicit**: advection, Coriolis, nonlinear terms
- **Implicit**: gravity wave terms (pressure gradient + divergence coupling), vertical diffusion

The implicit step requires solving a Helmholtz-like equation on the rectangular grid. Unlike Notus's spectral Helmholtz (diagonal per wavenumber), this is a 2D elliptic solve — options:
- Direct: FFT-based (if domain is periodic or with suitable extensions)
- Iterative: preconditioned conjugate gradient or multigrid
- Thomas algorithm in the vertical (tridiagonal), combined with horizontal iteration

For a regular rectangular grid, **FFT in both horizontal directions** with appropriate boundary treatment is likely the simplest and most JAX-friendly approach.

Time integration: leapfrog + Robert-Asselin (matching Notus) or explicit RK3 for the explicit part. RK3 avoids the computational mode issue and allows larger effective CFL, but requires 3 tendency evaluations per step.

### Boundary conditions: Davies relaxation

The lateral boundaries use a **Davies (1976) relaxation zone** — a buffer of N grid points (typically 8-10) where the LAM solution is blended toward the Notus-derived boundary state:

```
X_new = X_lam + alpha(d) * (X_notus - X_lam)
```

where alpha(d) decays exponentially from 1 at the boundary to 0 at the interior edge:

```
alpha(d) = exp(-d^2 / (2 * L^2))    for d = 0, 1, ..., N-1
```

or a cosine taper. Applied to all prognostic variables (u, v, T, q, ln ps) after each time step.

The boundary state is obtained by:
1. Notus outputs spectral state at regular intervals (e.g., every 3-6 hours)
2. Spectral → Gaussian grid via `notus.SpectralTransform.spectral_to_grid()`
3. Gaussian grid → rotated lat-lon LAM grid via bilinear or bicubic interpolation
4. Temporal interpolation (linear) between Notus output snapshots

The top boundary can use a **Rayleigh sponge layer** (same as Notus's boundary-layer drag but applied to the top N levels) to absorb vertically propagating waves that would otherwise reflect.

## Coupling interface with Notus

The LAM depends on `notus` as a Python package. Shared components:

### Imported directly from Notus
- `notus.constants.PlanetaryConstants`, `EARTH` — physical constants
- `notus.vertical.sigma.SigmaLevels` — vertical coordinate definition (pure sigma)
- `notus.vertical.operators` — geopotential integration, sigma_dot, vertical advection
- `notus.physics.radiation` — Frierson (gray LW + optional SW), Byrne (humidity-dependent), SPEEDY (4 LW + 2 SW multi-band)
- `notus.physics.convection` — dry convective adjustment, large-scale condensation (implicit), simplified Betts-Miller (deep/shallow)
- `notus.physics.moisture` — saturation thermodynamics (Bolton 1980), moist pseudoadiabat
- `notus.physics.surface` — bulk aerodynamic fluxes, slab ocean (implicit stepping), bucket land hydrology, `SurfaceState`/`OceanState`/`LandState`
- `notus.physics.boundary_layer` — Louis (1979) stability-dependent transfer coefficients, `SurfaceLayerConfig`
- `notus.physics.clouds` — diagnostic RH-based cloud scheme (convective + stratiform), coupled to SPEEDY radiation
- `notus.physics.solar` — orbital parameters, solar declination, daily-mean insolation (seasonal cycle)
- `notus.physics.physics_suite` — `PhysicsSuite` composable forcing (wraps radiation, convection, surface, boundary layer)
- `notus.topography` — idealized orography generators (Gaussian mountain, zonal ridge, sinusoidal), spectral smoothing, surface pressure initialization
- `notus.timestepping.coupled` — `CoupledStepper` / `build_coupled_pe_stepper()` coupling atmosphere + slab ocean + optional bucket land + surface geopotential
- `notus.transforms.SpectralTransform` — used only in the nesting coupler to convert Notus output to grid space

### Adapted / wrapped
- Physics forcing: Notus's `Forcing` protocol wraps spectral ↔ grid transforms around grid-point column physics. The LAM needs a grid-point forcing protocol that calls the underlying column physics directly, skipping the spectral round-trips:

```python
class GridForcing(Protocol):
    def __call__(
        self,
        u: jnp.ndarray,  # (n_levels, n_lat, n_lon)
        v: jnp.ndarray,
        temperature: jnp.ndarray,
        humidity: jnp.ndarray | None,
        surface_pressure: jnp.ndarray,  # (n_lat, n_lon)
        surface_temperature: jnp.ndarray,
    ) -> GridTendencies: ...
```

The column physics functions inside Notus (radiation, convection, condensation, surface flux, boundary layer, clouds) already work on grid-point arrays. The LAM wrappers call these directly. The `PhysicsSuite` class demonstrates how these components compose — the LAM `GridForcing` would follow a similar pattern but without spectral state conversion.

- Coupled surface: `CoupledStepper` currently threads `SurfaceState` (slab ocean + optional bucket land) through the atmospheric time stepping. The LAM equivalent would call the same surface stepping functions (`step_slab_ocean_implicit`, `step_bucket_hydrology`, `step_land_implicit`) directly on LAM grid-point fields.

### New to the LAM
- `RotatedLatLonGrid` — grid definition with rotation parameters
- `CGridState` — staggered wind + scalar fields
- Finite difference operators (gradient, divergence, curl on C-grid)
- Horizontal advection schemes
- Helmholtz solver for implicit time stepping
- `NestingBoundaryConditions` — interpolation and relaxation machinery

## Project structure

```
notus-lam/
├── pyproject.toml                  # dependency: notus
├── src/lam/
│   ├── __init__.py
│   ├── grid.py                     # RotatedLatLonGrid, rotation transforms
│   ├── state.py                    # CGridState (u, v, T, q, ps on staggered grid)
│   ├── constants.py                # LAM-specific defaults (domain size, resolution, etc.)
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── finite_difference.py    # C-grid gradient, divergence, curl
│   │   └── advection.py           # centered, PPM, vertical advection
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── shallow_water.py        # 2D C-grid shallow water (validation)
│   │   └── primitive_equations.py  # 3D hydrostatic PE on C-grid
│   ├── timestepping/
│   │   ├── __init__.py
│   │   ├── integrator.py           # RK3 or leapfrog + RAW filter
│   │   └── helmholtz.py            # 2D elliptic solver for implicit step
│   ├── nesting/
│   │   ├── __init__.py
│   │   ├── interpolation.py        # Gaussian grid → rotated lat-lon
│   │   ├── relaxation.py           # Davies relaxation zone
│   │   └── coupler.py              # End-to-end: Notus state → LAM BCs
│   ├── physics/
│   │   ├── __init__.py
│   │   └── forcing.py              # GridForcing protocol + Notus physics wrappers
│   └── diagnostics.py              # Domain-mean, cross-sections, etc.
├── tests/
│   └── ...
└── examples/
    ├── shallow_water_test.py       # C-grid SW validation
    ├── nested_held_suarez.py       # Global Notus → nested LAM
    └── regional_aquaplanet.py      # High-res regional simulation
```

## Phased roadmap

### Phase 1 — C-grid foundations

Rotated lat-lon grid, C-grid staggering, finite difference operators.

**Deliverables:**
- `RotatedLatLonGrid`: domain definition (center lat/lon, extent, resolution), rotation matrices, geographic ↔ rotated coordinate transforms
- C-grid indexing conventions and staggered field containers
- FD operators: gradient (T→u/v), divergence (u/v→T), curl (u/v→corner), Laplacian
- Advection: centered 2nd-order on C-grid

**Validation:**
- Operator identities: div(curl) = 0, curl(grad) = 0 to machine precision
- Advection of a cosine bell with known analytic solution
- Convergence rates match expected order (2nd for centered)

### Phase 2 — Shallow water on limited domain

2D shallow water equations on the C-grid, validating dynamics and time stepping with analytic or prescribed boundary conditions.

**Deliverables:**
- C-grid shallow water tendency computation
- Time stepper (RK3 or leapfrog-IMEX)
- Helmholtz solver for implicit gravity waves
- Prescribed boundary conditions (analytic or from a known test case)

**Validation:**
- Geostrophic adjustment: initial mass perturbation → balanced state
- Gravity wave propagation across domain with open boundaries
- Compare against Notus shallow water solution on the same region

### Phase 3 — 3D primitive equations

Full hydrostatic PE on the C-grid with hybrid sigma-pressure vertical coordinate.

**Deliverables:**
- 3D tendency computation (momentum, thermodynamic, continuity)
- Vertical operators imported from Notus (adapted for hybrid levels if available)
- 3D semi-implicit solver
- Isothermal rest-state and baroclinic wave initial conditions

**Validation:**
- Resting atmosphere remains at rest (no spurious circulations)
- Baroclinic wave growth rate and structure compared to Notus solution
- Mass and energy conservation

### Phase 4 — Nesting coupler

One-way nesting from Notus global output to LAM boundaries.

**Deliverables:**
- Notus output reader (spectral → grid conversion)
- Spatial interpolation: Gaussian grid → rotated lat-lon (bilinear/bicubic)
- Temporal interpolation between Notus snapshots
- Davies relaxation zone with configurable width and decay profile
- Upper sponge layer

**Validation:**
- Interpolation accuracy: Notus analytic field → LAM grid → compare with analytic
- Nested Held-Suarez: run Notus globally, nest LAM over a midlatitude region, verify LAM develops consistent climatology
- Boundary artifacts: no visible relaxation zone signature in interior solution

### Phase 5 — Physics and applications

Reuse Notus physics for realistic regional simulations.

**Deliverables:**
- `GridForcing` wrappers around Notus column physics (radiation, convection, condensation, boundary layer, clouds)
- Surface coupling: slab ocean, prescribed SST, or bucket land on LAM domain (reusing `step_slab_ocean_implicit`, `step_bucket_hydrology`, `step_land_implicit`)
- Seasonal forcing via Notus solar geometry (`daily_mean_insolation`, `OrbitalParameters`)
- Regional diagnostics and visualization

**Validation:**
- Nested moist aquaplanet: verify precipitation, humidity, temperature match Notus in LAM interior
- Resolution sensitivity: run same domain at multiple resolutions

### Phase 6 — Topography and land surface

Terrain-following coordinates and land-sea contrast.

**Deliverables:**
- Hybrid sigma-pressure with topographic surface geopotential (Notus already has surface geopotential support via `topography.py` and `build_coupled_pe_stepper`, but uses pure sigma; the LAM's hybrid coordinate needs terrain-following metric terms)
- Terrain-following metric terms in pressure gradient and advection
- Land surface model reusing Notus bucket land (`BucketLandConfig`, `LandState`, `step_land_implicit`, `step_bucket_hydrology`) with spatially varying surface properties (`SurfaceProperties`, `flat_continent_surface`)
- High-resolution topography on the LAM grid (can use Notus's `smooth_orography` for spectral smoothing, but the LAM may also want grid-space smoothing)

**Validation:**
- Flow over an idealized mountain (Schaer et al. 2002)
- Pressure gradient error test over steep terrain
- Realistic regional simulation with actual topography

## Key design decisions

### Why C-grid finite difference (not spectral, FV, or DG)?

- **Boundaries**: spectral methods need the whole sphere; FD handles limited domains naturally
- **Simplicity**: C-grid FD is the most mature and well-understood approach for hydrostatic LAMs
- **JAX compatibility**: regular rectangular arrays, no unstructured mesh indexing
- **Staggering benefits**: correct gravity wave dispersion without explicit filtering
- **Heritage**: WRF (C-grid), COSMO (C/D-grid), HARMONIE (spectral but moving to FD for LAM) all validate this choice

### Why rotated lat-lon (not Lambert conformal or Cartesian)?

- Regular arrays (no map-factor complications beyond a simple cos(phi) metric)
- Exact representation of spherical geometry (no projection distortion)
- Natural interface with Notus Gaussian grid (both are lat-lon based)
- Rotation is a trivial coordinate transform, not an approximation

### Why one-way nesting (not two-way)?

- Dramatically simpler to implement and debug
- Sufficient for most regional applications
- Two-way nesting (LAM feeds back into GCM) adds mass/energy conservation challenges and synchronization complexity
- Can be added later if needed, without changing the one-way infrastructure

## References

- Davies, H. C. (1976). A lateral boundary formulation for multi-level prediction models. *Quart. J. Roy. Meteor. Soc.*, 102, 405-418.
- Arakawa, A. & Lamb, V. R. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. *Methods in Computational Physics*, 17, 173-265.
- Colella, P. & Woodward, P. R. (1984). The Piecewise Parabolic Method (PPM) for gas-dynamical simulations. *J. Comput. Phys.*, 54, 174-201.
- Schaer, C. et al. (2002). A new terrain-following vertical coordinate formulation for atmospheric prediction models. *Mon. Wea. Rev.*, 130, 2459-2480.
- Berger, A. L. (1978). Long-term variations of daily insolation and Quaternary climatic changes. *J. Atmos. Sci.*, 35, 2362-2367.
