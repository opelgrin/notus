#!/usr/bin/env python3
"""Notus quickstart — moist atmosphere with a mountain in ~30 lines.

Runs a 50-day moist aquaplanet with a 2500 m mountain and prints
basic diagnostics. This is the simplest complete Notus simulation.

    uv run python examples/quickstart.py
"""

import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus import (
    EARTH,
    GaussianGrid,
    PhysicsSuite,
    SpectralTransform,
    build_pe_stepper,
    exponential_filter,
    gaussian_mountain,
    grid_surface_pressure,
    moist_aquaplanet_initial_state,
    orographic_log_surface_pressure,
    run_simulation,
    smooth_orography,
    standard_sigma_levels,
)


# Grid and vertical levels
grid = GaussianGrid(truncation=21)
transform = SpectralTransform(grid, EARTH.radius)
levels = standard_sigma_levels(20)

# Topography: 2500 m mountain at 30°N, smoothed to avoid Gibbs ringing
surface_phi = gaussian_mountain(
    transform, EARTH, height=2500.0, center_lat=np.radians(30), center_lon=np.pi
)
surface_phi = smooth_orography(surface_phi, transform.arrays)

# Moist initial conditions with surface pressure adjusted for terrain
state, ref_temps, _ = moist_aquaplanet_initial_state(transform, EARTH, levels)
ln_ps = orographic_log_surface_pressure(surface_phi, transform, EARTH, 264.0)
state = state.replace(log_surface_pressure=ln_ps)

# Physics (gray radiation, convection, condensation, surface fluxes)
physics = PhysicsSuite(transform, EARTH, levels)
dt = 900.0

# Build time stepper
init_fn, step_fn = build_pe_stepper(
    transform,
    EARTH,
    levels,
    ref_temps,
    surface_phi,
    dt=dt,
    spectral_filter=exponential_filter(transform.arrays, dt),
    forcing=physics,
)

# Run 50 days
result = run_simulation(init_fn=init_fn, step_fn=step_fn, initial_state=state, dt=dt, n_days=50)

# Print results
final = result.state
t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(final.temperature))
ps_grid = np.asarray(grid_surface_pressure(final, transform, EARTH))
q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(final.humidity))

print(f"Wall time: {result.wall_time:.0f}s")
print(f"Temperature: [{t_grid.min():.1f}, {t_grid.max():.1f}] K")
print(f"Surface pressure: [{ps_grid.min() / 100:.1f}, {ps_grid.max() / 100:.1f}] hPa")
print(f"Mean humidity: {q_grid.mean() * 1000:.2f} g/kg")
