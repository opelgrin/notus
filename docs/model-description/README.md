# Notus Model Description

Notus is a spectral-transform atmospheric General Circulation Model (GCM) written from
scratch in Python/JAX. It solves the hydrostatic primitive equations on a rotating sphere
using spherical harmonics, targeting standard benchmarks and idealized climate
configurations.

This document set describes the mathematical formulation, numerical methods, and physical
parameterizations implemented in the model. It is intended to provide sufficient detail to
understand what the model does and why particular choices were made, without descending into
code-level implementation details.

## Contents

1. [Spectral Transform Method](spectral-transform.md) -- Gaussian grid, spherical
   harmonics, FFT + Legendre transforms, anti-aliasing, spectral operators
2. [Governing Equations](governing-equations.md) -- Primitive equations in
   vorticity-divergence form on sigma coordinates
3. [Vertical Discretization](vertical-discretization.md) -- Sigma coordinate, Lorenz
   staggering, geopotential integration, vertical transport
4. [Spectral Dynamics](spectral-dynamics.md) -- Spectral tendency computation, nonlinear
   product evaluation, hyperdiffusion
5. [Time Integration](time-integration.md) -- IMEX leapfrog, semi-implicit gravity wave
   treatment, Robert-Asselin-Williams filter
6. [Radiation](radiation.md) -- Frierson gray, Byrne humidity-dependent, and SPEEDY
   multi-band radiation schemes
7. [Moist Physics](moist-physics.md) -- Moisture thermodynamics, large-scale condensation,
   simplified Betts-Miller convection, dry convective adjustment
8. [Boundary Layer and Surface Fluxes](boundary-layer.md) -- Louis (1979) stability
   functions, bulk aerodynamic surface exchange
9. [Surface Models and Coupled Integration](surface-models.md) -- Slab ocean, bucket land
   surface, solar geometry, topography, coupled atmosphere-surface integration
10. [Standard Test Cases](test-cases.md) -- Held-Suarez, Jablonowski-Williamson, aquaplanet
    configurations

## References

References are collected at the end of each document.
