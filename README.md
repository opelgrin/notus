# Notus

A spectral-transform General Circulation Model written from scratch in Python/JAX.

Named after **Νότος** (Notus), the Greek god of the south wind.

## What is this?

Notus is an atmospheric GCM built for learning and experimentation. It implements a pseudo-spectral dynamical core using spherical harmonics on a Gaussian grid, targeting the [Held-Suarez (1994)](https://doi.org/10.1175/1520-0477(1994)075<1825:APFTIO>2.0.CO;2) benchmark as its first milestone.

The model is planet-agnostic — it can simulate any rotating planet with an ideal-gas atmosphere.

## Architecture

- **Spectral transform**: spherical harmonic decomposition (FFT in longitude, Legendre transform in latitude)
- **Vertical coordinate**: sigma (p/ps) on a Lorenz grid
- **Time stepping**: IMEX leapfrog with Robert-Asselin filter + semi-implicit 3D Helmholtz solver (with virtual temperature linearization for moist dynamics)
- **Implicit physics**: surface fluxes (sensible + latent heat), Rayleigh friction, and Betts-Miller convection treated with backward Euler for unconditional stability
- **Filtering**: exponential spectral filter (Hou & Li 2007) + del-8 hyperdiffusion
- **Physics**: Held-Suarez forcing, Frierson (2006/2007) simple physics (gray radiation, Betts-Miller convection, large-scale condensation, bulk surface fluxes), pluggable via `Forcing` protocol
- **Moisture**: specific humidity as optional spectral tracer, surface evaporation, condensation with latent heating, virtual temperature feedback in both dynamics and semi-implicit solver
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
uv run python examples/simple_physics_aquaplanet.py

# Generate diagnostic plots
uv run python examples/plot_simple_physics.py
```

### Moist aquaplanet

Run the moist aquaplanet with condensation, Betts-Miller convection, and surface evaporation:

```bash
# Quick demo (T21, 300 days)
uv run python examples/moist_aquaplanet.py

# Longer run
uv run python examples/moist_aquaplanet.py --days 500 --spinup 200

# Generate diagnostic plots (U, T, q, EKE, spinup, surface pressure)
uv run python examples/plot_moist_aquaplanet.py
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
- [ ] **Phase 7 — Seasonal cycle**: orbital parameters, shortwave/longwave radiation, diurnal and annual cycles
- [ ] **Phase 8 — Surface coupling**: slab ocean, simple land surface, Monin-Obukhov boundary layer

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
