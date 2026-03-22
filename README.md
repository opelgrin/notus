# Notus

A spectral-transform General Circulation Model written from scratch in Python/JAX.

Named after **Νότος** (Notus), the Greek god of the south wind.

## What is this?

Notus is an atmospheric GCM built for learning and experimentation. It implements a pseudo-spectral dynamical core using spherical harmonics on a Gaussian grid, targeting the [Held-Suarez (1994)](https://doi.org/10.1175/1520-0477(1994)075<1825:APFTIO>2.0.CO;2) benchmark as its first milestone.

The model is planet-agnostic — it can simulate any rotating planet with an ideal-gas atmosphere.

## Architecture

- **Spectral transform**: spherical harmonic decomposition (FFT in longitude, Legendre transform in latitude)
- **Vertical coordinate**: sigma (p/ps), Eulerian
- **Time stepping**: leapfrog with Robert-Asselin-Williams filter + semi-implicit gravity waves
- **Diffusion**: biharmonic (del-4) hyperdiffusion
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
transform = SpectralTransform(grid)

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

## Roadmap

- [x] **Phase 1 — Foundations**: Gaussian grid, spherical harmonic transforms, spectral operators
- [ ] **Phase 2 — Shallow water**: 2D shallow water equations on the sphere (validates the spectral machinery + time stepping). Williamson et al. (1992) standard test cases.
- [ ] **Phase 3 — Primitive equations**: extend to 3D with sigma vertical coordinate, hydrostatic equation, semi-implicit gravity wave treatment
- [ ] **Phase 4 — Held-Suarez**: Newtonian relaxation + Rayleigh friction, 1200-day integration at T42 L20, validate against published zonal-mean diagnostics
- [ ] **Phase 5 — Simple physics**: gray radiation, dry convective adjustment
- [ ] **Phase 6 — Moisture**: specific humidity tracer, large-scale condensation, simple convection scheme (Betts-Miller or similar)
- [ ] **Phase 7 — Seasonal cycle**: orbital parameters (obliquity, eccentricity), shortwave/longwave radiation, diurnal and annual cycles
- [ ] **Phase 8 — Surface coupling**: slab ocean, simple land surface, boundary layer parameterization

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
