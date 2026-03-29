# Notus

A spectral-transform General Circulation Model written from scratch in Python/JAX.

Named after **Νότος** (Notus), the Greek god of the south wind.

## What is this?

Notus is an atmospheric GCM built for learning and experimentation. It implements a pseudo-spectral dynamical core using spherical harmonics on a Gaussian grid, with moist physics, radiation, and interactive surface models.

The model is planet-agnostic — it can simulate any rotating planet with an ideal-gas atmosphere.

## Architecture

- **Spectral transform**: spherical harmonic decomposition (FFT in longitude, Legendre transform in latitude)
- **Vertical coordinate**: sigma (p/ps) on a Lorenz grid
- **Time stepping**: IMEX leapfrog with Robert-Asselin filter + semi-implicit 3D Helmholtz solver (with virtual temperature linearization for moist dynamics)
- **Implicit physics**: surface fluxes (sensible + latent heat), Rayleigh friction, and Betts-Miller convection treated with backward Euler for unconditional stability
- **Filtering**: exponential spectral filter (Hou & Li 2007) + del-8 hyperdiffusion
- **Physics**: Held-Suarez forcing, Frierson (2006/2007) simple physics (Betts-Miller convection, large-scale condensation, bulk surface fluxes), pluggable via `Forcing` protocol
- **Moisture**: specific humidity as optional spectral tracer, surface evaporation, condensation with latent heating, virtual temperature feedback in both dynamics and semi-implicit solver
- **Radiation**: three schemes available — Frierson gray LW, Byrne/Isca semi-gray (humidity-dependent), and SPEEDY multi-band (4-band LW with temperature-dependent fractions + 2-band SW with near-IR H₂O absorption + diagnostic clouds). Seasonal insolation with orbital parameters. CO₂ sensitivity knob.
- **Boundary layer**: Monin-Obukhov surface layer (Louis 1979 stability functions), spatially varying roughness lengths
- **Surface models**: slab ocean (implicit, prescribed Q-flux) + bucket land surface (Frierson 2006 / Manabe 1969: soil energy balance, P-E-R hydrology, evaporation resistance)
- **Clouds**: diagnostic cloud scheme (SPEEDY-style) — RH + precipitation-based cloud cover, stability-dependent stratiform clouds, SW reflection and LW absorption
- **Computation**: JAX (JIT compilation, GPU support, autodiff)

## Installation

```bash
uv sync
```

## Usage

```python
from notus import GaussianGrid, SpectralTransform, EARTH, PlanetaryConstants

# Create a T42 grid (standard Held-Suarez resolution)
grid = GaussianGrid(truncation=42)

# Set up spectral transforms
transform = SpectralTransform(grid, EARTH.radius)

# Or define your own planet
my_planet = PlanetaryConstants(
    name="Arrakis",
    radius=6.0e6,
    rotation_rate=8.0e-5,
    gravity=9.1,
    gas_constant=287.0,
    specific_heat_cp=1005.0,
)
```

### Held-Suarez benchmark

Run the standard dry dynamical core intercomparison:

```bash
# Quick demo (T21, 300 days, ~45s)
uv run python examples/held_suarez.py --days 300 --spinup 100 --truncation 21 --dt 1200

# Full benchmark (T42, 1200 days)
uv run python examples/held_suarez.py

# Generate diagnostic plots
uv run python examples/plot_held_suarez.py
```

### Frierson aquaplanet (dry)

Run the gray-radiation aquaplanet with surface fluxes (Frierson et al. 2006):

```bash
# Quick demo (T21, 300 days)
uv run python examples/physics_suite_aquaplanet.py

# Generate diagnostic plots
uv run python examples/plot_physics_suite.py
```

### Moist aquaplanet

Run the moist aquaplanet with condensation, Betts-Miller convection, and surface evaporation:

```bash
# Quick demo (T21, 300 days, Frierson gray radiation)
uv run python examples/moist_aquaplanet.py

# With SPEEDY multi-band radiation + clouds
uv run python examples/moist_aquaplanet.py --scheme speedy --clouds

# Byrne two-band radiation
uv run python examples/byrne_aquaplanet.py --scheme speedy --clouds --days 300

# Generate diagnostic plots (U, T, q, EKE, spinup, surface pressure)
uv run python examples/plot_moist_aquaplanet.py
```

### Slab ocean + land

```bash
# Slab ocean with seasonal cycle (Byrne radiation, default)
uv run python examples/slab_ocean_aquaplanet.py

# Slab ocean with SPEEDY radiation + clouds
uv run python examples/slab_ocean_aquaplanet.py --scheme speedy --clouds

# Land-ocean aquaplanet with bucket hydrology
uv run python examples/land_ocean_aquaplanet.py --scheme speedy --clouds
```

### Radiation validation

```bash
# Validate radiation energy conservation (SW column closure, TOA balance)
uv run python examples/validate_radiation.py --scheme speedy --clouds --days 300

# Coupled slab ocean validation (Q-flux diagnosis + coupled integration)
uv run python examples/validate_radiation_coupled.py --scheme speedy --clouds
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for detailed descriptions of each phase.

- [x] **Phase 1 — Spectral foundations**: Gaussian grid, spherical harmonic transforms, spectral operators
- [x] **Phase 2 — Shallow water**: 2D shallow water equations on the sphere, Williamson et al. (1992) test case 2 (steady-state 10 days, mass conservation)
- [x] **Phase 3 — Primitive equations**: 3D hydrostatic dycore on sigma levels, Jablonowski-Williamson (2006) baroclinic wave (10-day integration, ps minimum ~950 hPa, conservation < 0.1%)
- [x] **Phase 4 — Held-Suarez**: Newtonian relaxation + Rayleigh friction, 1200-day integration at T42 L20, validated climatology (jets, temperature, eddies)
- [x] **Phase 5 — Simple physics**: gray radiation, dry convective adjustment, bulk surface flux, Frierson (2006) aquaplanet
- [x] **Phase 6 — Moisture**: specific humidity tracer, large-scale condensation, Betts-Miller convection, surface evaporation
- [x] **Phase 6b — Virtual temperature + stability**: T_v in SI solver, implicit surface fluxes + BM convection (dt: 580s → 1200s)
- [x] **Phase 7 — Seasonal cycle + slab ocean**: two-band radiation (H₂O feedback), annual cycle, slab ocean with Q-flux
- [x] **Phase 8A — Monin-Obukhov surface layer**: Louis (1979) stability functions, implicit slab ocean step, warm-start infrastructure
- [x] **Phase 8B — Surface type infrastructure**: land-sea masks, per-gridpoint albedo and roughness, flat continent generator
- [x] **Phase 8C — Bucket land surface**: Frierson/Manabe soil energy balance, bucket hydrology (P-E-R), evaporation resistance, blended ocean-land fluxes
- [x] **Phase 9 — Radiation upgrade**: SW energy conservation, SPEEDY 4-band LW + 2-band SW, diagnostic clouds, CO₂ knob (TOA imbalance: +109 → -7 W/m²)
- [ ] **Phase 10 — Topography**: prescribed orography, spectral smoothing, surface pressure initialization

## Tests

```bash
uv run pytest tests/ -v
```

## References

Built with guidance from:

- [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) — clean modular spectral GCM in Julia
- [Dinosaur](https://github.com/neuralgcm/dinosaur) — differentiable spectral dycore in JAX (Google)
- [SPEEDY](http://users.ictp.it/~kucharsk/speedy-doc.html) — simplified atmospheric GCM (ICTP)
- Held & Suarez (1994), *A Proposal for the Intercomparison of the Dynamical Cores of Atmospheric General Circulation Models*, BAMS
- Frierson et al. (2006), *A Gray-Radiation Aquaplanet Moist GCM*, JAS
- Frierson (2007), *The Dynamics of Idealized Convection Schemes and Their Effect on the Zonally Averaged Tropical Circulation*, JAS
