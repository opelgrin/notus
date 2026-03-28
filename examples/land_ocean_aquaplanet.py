#!/usr/bin/env python3
"""Land-ocean coupled aquaplanet with bucket hydrology (Phase 8C).

Runs a coupled integration with:
- Slab ocean (50 m mixed layer) over ocean gridpoints
- Bucket land surface model over a flat rectangular continent
- Byrne humidity-dependent radiation with shortwave absorption
- Seasonal insolation from orbital parameters
- In-memory prescribed-SST spinup for Q-flux diagnosis

The bucket land model (Manabe 1969) tracks soil temperature and a
single-layer water reservoir.  Evaporation is moisture-limited when the
bucket depth falls below 75% of field capacity (0.15 m).

Usage
-----
    uv run python examples/land_ocean_aquaplanet.py
    uv run python examples/land_ocean_aquaplanet.py --days 600 --spinup 200
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
from notus.diagnostics import ZonalMeanState, compute_zonal_mean_state
from notus.grid import GaussianGrid
from notus.initial_conditions import moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.solar import EARTH_ORBIT
from notus.physics.surface import (
    BucketLandConfig,
    OceanState,
    PrescribedSST,
    SlabOceanConfig,
    SurfaceState,
    compute_sst,
    init_land_state,
)
from notus.physics.surface_types import flat_continent_surface
from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.timestepping.spinup import spinup_prescribed_sst
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


def run_land_ocean(
    n_days: int = 600,
    spinup_days: int = 200,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    prescribed_spinup_days: int = 100,
    prescribed_averaging_days: int = 100,
) -> bool:
    """Run a land-ocean coupled aquaplanet with bucket hydrology.

    Returns True if the integration completes without blowup.
    """
    print(f"Land-ocean aquaplanet: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Coupled spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")
    print(f"  Prescribed-SST spinup: {prescribed_spinup_days}+{prescribed_averaging_days} days")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # --- Grid and vertical setup ---
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

    # --- Physics configuration ---
    config = SimplePhysicsConfig(
        radiation_scheme="byrne",
        sw_tau_0=0.22,
        orbital=EARTH_ORBIT,
    )
    forcing = SimplePhysics(transform, EARTH, levels, config=config)

    # --- Surface configuration ---
    # Flat continent: 30S-60N, 0-180E (roughly Eurasia-like in extent)
    lat_deg = np.degrees(np.asarray(grid.latitudes))
    sfc_props = flat_continent_surface(
        grid.latitudes,
        grid.longitudes,
        lat_south=-30.0,
        lat_north=60.0,
        lon_west=0.0,
        lon_east=180.0,
    )
    land_frac = np.asarray(sfc_props.land_fraction)
    land_pct = float(np.mean(land_frac)) * 100
    print(f"  Land fraction: {land_pct:.0f}% of globe")
    print("  Land extent: 30S-60N, 0-180E")

    # Ocean and land configs
    ocean_config = SlabOceanConfig(mixed_layer_depth=50.0)
    land_config = BucketLandConfig(
        soil_heat_capacity=4.0e6,
        bucket_capacity=0.15,
        moisture_dependent_albedo=True,
    )
    print(
        f"  Bucket: W_max={land_config.bucket_capacity} m, "
        f"moisture-dependent albedo: {land_config.moisture_dependent_albedo}"
    )

    # --- Prescribed-SST spinup for Q-flux diagnosis ---
    n_pre = prescribed_spinup_days + prescribed_averaging_days
    print(f"\n--- Prescribed-SST spinup ({n_pre} days) ---")
    # Use aquaplanet surface for spinup (no land)
    filt = exponential_filter(transform.arrays, dt)
    spinup_forcing = SimplePhysics(transform, EARTH, levels, config=config)

    t_spinup = time.perf_counter()
    result = spinup_prescribed_sst(
        state,
        spinup_forcing,
        transform,
        EARTH,
        levels,
        ref_temps,
        surface_phi,
        dt,
        spinup_days=prescribed_spinup_days,
        averaging_days=prescribed_averaging_days,
        spectral_filter=filt,
        surface_albedo=EARTH.surface_albedo,
        verbose=True,
    )
    t_spinup_done = time.perf_counter() - t_spinup
    print(f"  Spinup complete in {t_spinup_done:.0f}s")
    print(
        f"  Q-flux range: [{float(jnp.min(result.q_flux)):.1f}, "
        f"{float(jnp.max(result.q_flux)):.1f}] W/m^2"
    )

    # --- Initialize surface state ---
    sst_init = compute_sst(
        PrescribedSST(t_min=config.sst_t_min, t_delta=config.sst_t_delta, phi_w=config.sst_phi_w),
        grid.latitudes,
    )
    ocean = OceanState(surface_temperature=sst_init)
    land = init_land_state(sfc_props.land_fraction, sst_init, land_config)
    surface = SurfaceState(ocean=ocean, land=land)

    print(f"\n  Initial SST: [{float(jnp.min(sst_init)):.1f}, {float(jnp.max(sst_init)):.1f}] K")
    bucket_pct = float(jnp.mean(land.bucket_depth)) / land_config.bucket_capacity * 100
    print(
        f"  Initial bucket depth: {float(jnp.mean(land.bucket_depth)):.3f} m "
        f"({bucket_pct:.0f}% capacity)"
    )

    # --- Build coupled stepper ---
    init_fn, step_fn = build_coupled_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        forcing=forcing,
        ocean_config=ocean_config,
        q_flux=result.q_flux,
        surface_properties=sfc_props,
        land_config=land_config,
        spectral_filter=filt,
    )

    steps_per_day = int(86400 / dt)
    days_per_year = EARTH_ORBIT.days_per_year

    def one_day(
        carry: tuple,
        day_of_year: jnp.ndarray,
    ) -> tuple[tuple, None]:
        prev, curr, sfc = carry
        forcing.day_of_year = day_of_year

        def step(carry: tuple, _: None) -> tuple[tuple, None]:
            p, c, s = carry
            p, c, s = step_fn(p, c, s)
            return (p, c, s), None

        (prev, curr, sfc), _ = jax.lax.scan(step, (prev, curr, sfc), None, length=steps_per_day)
        return (prev, curr, sfc), None

    one_day_jit = jax.jit(one_day)

    # --- Initialize with spun-up state ---
    print("\n--- Coupled integration ---")
    print("Initializing...")
    t0 = time.perf_counter()
    prev, curr, surface = init_fn(result.state, surface)

    # Force JIT compilation
    (prev, curr, surface), _ = one_day_jit((prev, curr, surface), jnp.float64(0.0))
    t_compile = time.perf_counter() - t0
    print(f"Day 1 (incl. JIT compile): {t_compile:.1f}s")

    # --- Accumulator ---
    n_avg = 0
    accum_zm: ZonalMeanState | None = None

    def _accumulate(curr_state: jnp.ndarray) -> None:
        nonlocal n_avg, accum_zm
        zm = compute_zonal_mean_state(curr_state, transform)
        n_avg += 1
        if accum_zm is None:
            accum_zm = zm
        else:
            accum_zm = ZonalMeanState(
                u=accum_zm.u + zm.u,
                v=accum_zm.v + zm.v,
                temperature=accum_zm.temperature + zm.temperature,
                u_prime_sq=accum_zm.u_prime_sq + zm.u_prime_sq,
                v_prime_sq=accum_zm.v_prime_sq + zm.v_prime_sq,
                uv_prime=accum_zm.uv_prime + zm.uv_prime,
                vt_prime=accum_zm.vt_prime + zm.vt_prime,
            )

    eq_idx = np.argmin(np.abs(lat_deg))
    jet_level = max(0, n_levels // 4)

    def _print_status(day: int, elapsed: float) -> bool:
        """Print diagnostics. Returns False if blowup detected."""
        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
        sst = np.asarray(surface.ocean.surface_temperature)

        u_spec, _ = uv_from_vordiv(
            curr.vorticity[jet_level],
            curr.divergence[jet_level],
            transform.arrays,
        )
        u_grid = np.asarray(transform.spectral_to_grid(u_spec))
        cos_lat = np.asarray(grid.cos_lat)
        u_grid = u_grid / cos_lat[:, None]

        t_mean = float(np.mean(t_grid))
        days_per_sec = (day - 1) / elapsed if elapsed > 0 else 0
        phase = "spinup" if day <= spinup_days else "averaging"

        # Land diagnostics
        land_t = np.asarray(surface.land.soil_temperature)
        bucket = np.asarray(surface.land.bucket_depth)
        land_mask = land_frac > 0.5
        land_t_mean = float(np.mean(land_t[land_mask])) if np.any(land_mask) else 0.0
        bucket_mean = float(np.mean(bucket[land_mask])) if np.any(land_mask) else 0.0

        q_str = ""
        if curr.has_humidity:
            q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
            q_mean_gkg = float(np.mean(q_grid)) * 1000
            q_str = f"  q={q_mean_gkg:.2f}g/kg"

        print(
            f"  Day {day:5d} [{phase:>9s}]: "
            f"SST=[{np.min(sst):.1f},{np.max(sst):.1f}] "
            f"T_land={land_t_mean:.1f} K "
            f"bucket={bucket_mean:.3f}m "
            f"|U|={np.max(np.abs(u_grid)):.1f}"
            f"{q_str}"
            f"  [{days_per_sec:.1f} d/s]"
        )

        if not np.isfinite(t_mean) or not np.all(np.isfinite(sst)):
            print("ERROR: Integration has blown up!")
            return False
        return True

    # --- Main integration loop ---
    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        day_of_year = jnp.float64(day % days_per_year)
        forcing.sst = surface.ocean.surface_temperature
        (prev, curr, surface), _ = one_day_jit((prev, curr, surface), day_of_year)

        if day > spinup_days:
            _accumulate(curr)

        if day <= 10 or day % 50 == 0 or day == n_days:
            elapsed = time.perf_counter() - t_start
            if not _print_status(day, elapsed):
                return False

    total_time = time.perf_counter() - t0
    print(f"\nDone. Total wall time: {total_time:.0f}s ({total_time / 3600:.1f}h)")
    print(f"Averaged over {n_avg} daily samples")

    # --- Final summary ---
    sst_final = np.asarray(surface.ocean.surface_temperature)
    land_t_final = np.asarray(surface.land.soil_temperature)
    bucket_final = np.asarray(surface.land.bucket_depth)
    land_mask = land_frac > 0.5

    print("\n--- Final surface state ---")
    print(f"  SST range:        [{np.min(sst_final):.1f}, {np.max(sst_final):.1f}] K")
    if np.any(land_mask):
        print(
            f"  Land T range:     [{np.min(land_t_final[land_mask]):.1f}, "
            f"{np.max(land_t_final[land_mask]):.1f}] K"
        )
        print(
            f"  Bucket depth:     [{np.min(bucket_final[land_mask]):.4f}, "
            f"{np.max(bucket_final[land_mask]):.4f}] m  "
            f"(mean {np.mean(bucket_final[land_mask]):.4f})"
        )
        n_dry = float(np.sum(bucket_final[land_mask] < land_config.w_crit))
        dry_pct = n_dry / np.sum(land_mask) * 100
        print(f"  Dry land (W<W_c): {dry_pct:.0f}%")

    if accum_zm is not None and n_avg > 0:
        mean_zm = ZonalMeanState(
            u=accum_zm.u / n_avg,
            v=accum_zm.v / n_avg,
            temperature=accum_zm.temperature / n_avg,
            u_prime_sq=accum_zm.u_prime_sq / n_avg,
            v_prime_sq=accum_zm.v_prime_sq / n_avg,
            uv_prime=accum_zm.uv_prime / n_avg,
            vt_prime=accum_zm.vt_prime / n_avg,
        )

        sigma_full = np.asarray(levels.sigma_full)
        surface_idx = np.argmax(sigma_full)
        upper_mask = sigma_full < 0.4

        print("\n--- Atmospheric climatology ---")
        print(f"  T(equator, sfc) = {mean_zm.temperature[surface_idx, eq_idx]:.1f} K")
        pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))
        print(f"  T(pole, sfc)    = {mean_zm.temperature[surface_idx, pole_idx]:.1f} K")
        jet_max = float(np.max(np.abs(mean_zm.u[upper_mask, :])))
        print(f"  Jet max |U|     = {jet_max:.1f} m/s")
        eke = 0.5 * (mean_zm.u_prime_sq + mean_zm.v_prime_sq)
        print(f"  EKE max         = {float(np.max(eke)):.1f} m^2/s^2")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Land-ocean coupled aquaplanet (bucket hydrology)")
    parser.add_argument("--days", type=int, default=600, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=200, help="Coupled spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument(
        "--prescribe-spinup", type=int, default=100, help="Prescribed-SST spinup days"
    )
    parser.add_argument(
        "--prescribe-avg", type=int, default=100, help="Prescribed-SST averaging days for Q-flux"
    )
    args = parser.parse_args()

    passed = run_land_ocean(
        n_days=args.days,
        spinup_days=args.spinup,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        prescribed_spinup_days=args.prescribe_spinup,
        prescribed_averaging_days=args.prescribe_avg,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
