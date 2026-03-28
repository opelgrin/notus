#!/usr/bin/env python3
"""Diagnose Q-flux from a prescribed-SST aquaplanet run.

Runs a moist aquaplanet with prescribed SST (Byrne radiation) and
accumulates the time-mean net surface energy flux.  The diagnosed
Q-flux is then ``Q_flux(lat) = -F_net(lat)``, i.e. the heat transport
needed to maintain the prescribed SST pattern.

The output is a .npz file containing ``q_flux``, ``latitudes``, and
``sin_lat`` arrays that can be loaded for slab ocean experiments.

Usage
-----
    uv run python examples/diagnose_qflux.py
    uv run python examples/diagnose_qflux.py --days 500 --spinup 200 --output qflux.npz
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import STEFAN_BOLTZMANN
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


def diagnose_surface_flux(
    state,
    transform: SpectralTransform,
    forcing: SimplePhysics,
    surface_pressure: jnp.ndarray,
) -> jnp.ndarray:
    """Compute net downward surface energy flux [W/m²].

    Returns the zonal-mean net flux, shape ``(n_lat,)``.
    Positive means the ocean is gaining energy.
    """
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    lowest = levels.n_levels - 1
    sin_lat = transform.grid.sin_lat

    # Atmospheric temperature at lowest level
    t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
    t_lowest = t_grid[lowest]

    # Surface winds
    u_cos_spec, v_cos_spec = uv_from_vordiv(
        state.vorticity[lowest],
        state.divergence[lowest],
        transform.arrays,
    )
    u_cos_grid = transform.spectral_to_grid(u_cos_spec)
    v_cos_grid = transform.spectral_to_grid(v_cos_spec)
    cos_lat = transform.grid.cos_lat[:, None]
    cos_lat_safe = jnp.maximum(cos_lat, 1.0e-6)
    u_grid = u_cos_grid / cos_lat_safe
    v_grid = v_cos_grid / cos_lat_safe
    wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

    # Surface density
    t_air_safe = jnp.maximum(t_lowest, 1.0)
    rho_sfc = surface_pressure / (planet.gas_constant * t_air_safe)

    sst = forcing.sst  # (n_lat,)
    sst_bc = sst[:, None]

    # --- SW absorbed at surface ---
    if cfg.sw_tau_0 > 0.0:
        insolation = (
            planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
        )
        sw_surface = insolation[:, None] * jnp.exp(-cfg.sw_tau_0) * (1.0 - planet.surface_albedo)
    else:
        sw_surface = jnp.zeros_like(t_lowest)

    # --- LW down at surface ---
    # Approximate: use atmospheric emission at lowest level
    # A better approach would extract it from the two-stream solver,
    # but this is consistent with the coupled stepper's approximation.
    lw_down = STEFAN_BOLTZMANN * t_lowest**4 * (1.0 - jnp.exp(-0.5))

    # --- LW up from surface ---
    lw_up = STEFAN_BOLTZMANN * sst_bc**4

    # --- Sensible heat flux (positive = upward) ---
    h_flux = rho_sfc * planet.specific_heat_cp * cfg.c_d * wind_speed * (sst_bc - t_lowest)

    # --- Latent heat flux (positive = upward) ---
    q_lowest = jnp.zeros_like(t_lowest)
    if state.humidity is not None:
        q_lowest = jnp.maximum(
            transform.spectral_to_grid(state.humidity[lowest]),
            0.0,
        )
    q_sat_sfc = saturation_specific_humidity(sst_bc, surface_pressure, planet.epsilon_moisture)
    e_flux = (
        rho_sfc * planet.latent_heat_vaporization * cfg.c_d * wind_speed * (q_sat_sfc - q_lowest)
    )

    # --- Net surface flux (positive = into ocean) ---
    net_flux = sw_surface + lw_down - lw_up - h_flux - e_flux

    # Zonal mean
    return jnp.mean(net_flux, axis=-1)


def run_diagnose_qflux(
    n_days: int = 300,
    spinup_days: int = 100,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    output_path: str = "qflux.npz",
) -> bool:
    """Run prescribed-SST integration and diagnose Q-flux."""
    print(f"Q-flux diagnosis: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # --- Setup ---
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(n_levels)

    state, ref_temps, surface_phi = moist_aquaplanet_initial_state(
        transform,
        EARTH,
        levels,
        initial_rh=0.7,
        seed=42,
    )

    config = SimplePhysicsConfig(
        radiation_scheme="byrne",
        sw_tau_0=0.22,
    )
    forcing = SimplePhysics(transform, EARTH, levels, config=config)
    filt = exponential_filter(transform.arrays, dt)
    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        spectral_filter=filt,
        forcing=forcing,
    )

    steps_per_day = int(86400 / dt)

    def one_day(carry, _):
        prev, curr = carry

        def step(carry, _):
            p, c = carry
            p, c = step_fn(p, c)
            return (p, c), None

        (prev, curr), _ = jax.lax.scan(step, (prev, curr), None, length=steps_per_day)
        return (prev, curr), None

    one_day_jit = jax.jit(one_day)

    diagnose_jit = jax.jit(lambda s, ps: diagnose_surface_flux(s, transform, forcing, ps))

    # --- Initialize ---
    print("Initializing...")
    t0 = time.perf_counter()
    prev, curr = init_fn(state)
    (prev, curr), _ = one_day_jit((prev, curr), None)
    t_compile = time.perf_counter() - t0
    print(f"Day 1 (incl. JIT compile): {t_compile:.1f}s")

    # --- Accumulate surface flux ---
    n_lat = grid.n_lat
    flux_accum = np.zeros(n_lat)
    n_samples = 0

    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        (prev, curr), _ = one_day_jit((prev, curr), None)

        if day > spinup_days:
            # Compute surface pressure
            lnps_grid = transform.spectral_to_grid(curr.log_surface_pressure)
            ps_grid = EARTH.reference_pressure * jnp.exp(lnps_grid)

            net_flux_zm = diagnose_jit(curr, ps_grid)
            flux_accum += np.asarray(net_flux_zm)
            n_samples += 1

        if day <= 10 or day % 50 == 0 or day == n_days:
            elapsed = time.perf_counter() - t_start
            dps = (day - 1) / elapsed if elapsed > 0 else 0
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            phase = "spinup" if day <= spinup_days else "averaging"
            print(f"  Day {day:5d} [{phase:>9s}]: T_mean={np.mean(t_grid):.1f} K  [{dps:.1f} d/s]")

    total_time = time.perf_counter() - t0
    print(f"\nDone. Wall time: {total_time:.0f}s")
    print(f"Averaged over {n_samples} daily samples (days {spinup_days + 1}-{n_days})")

    # --- Compute Q-flux ---
    mean_flux = flux_accum / n_samples
    q_flux = -mean_flux  # Q = -F_net to maintain SST

    lat_deg = np.degrees(np.asarray(grid.latitudes))
    sin_lat = np.asarray(grid.sin_lat)

    print("\n--- Diagnosed Q-flux ---")
    eq_idx = np.argmin(np.abs(lat_deg))
    pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))
    mid_idx = np.argmin(np.abs(np.abs(lat_deg) - 45.0))
    print(f"  Q(equator)   = {q_flux[eq_idx]:+.1f} W/m^2")
    print(f"  Q(45 deg)    = {q_flux[mid_idx]:+.1f} W/m^2")
    print(f"  Q(pole)      = {q_flux[pole_idx]:+.1f} W/m^2")
    print(f"  Q range      = [{np.min(q_flux):.1f}, {np.max(q_flux):.1f}] W/m^2")

    # Global mean should be near zero (energy conservation)
    cos_lat_arr = np.sqrt(1.0 - sin_lat**2)
    global_mean = np.sum(q_flux * cos_lat_arr) / np.sum(cos_lat_arr)
    print(f"  Global mean  = {global_mean:+.2f} W/m^2 (should be ~0)")

    # --- Save ---
    sst = np.asarray(forcing.sst)
    np.savez(
        output_path,
        q_flux=q_flux,
        latitudes=np.asarray(grid.latitudes),
        sin_lat=sin_lat,
        lat_deg=lat_deg,
        mean_net_flux=mean_flux,
        sst=sst,
        n_samples=n_samples,
    )
    print(f"\nSaved to {output_path}")
    print(f"  Load with: data = np.load('{output_path}'); q_flux = data['q_flux']")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Q-flux from prescribed-SST run")
    parser.add_argument("--days", type=int, default=300, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--output", type=str, default="qflux.npz", help="Output .npz path")
    args = parser.parse_args()

    passed = run_diagnose_qflux(
        n_days=args.days,
        spinup_days=args.spinup,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        output_path=args.output,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
