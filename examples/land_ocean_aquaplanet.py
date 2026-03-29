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
import logging
import sys

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus import (
    EARTH,
    EARTH_ORBIT,
    BucketLandConfig,
    ByrneRadiation,
    GaussianGrid,
    OceanState,
    PhysicsSuite,
    PhysicsSuiteConfig,
    PrescribedSST,
    SlabOceanConfig,
    SpectralTransform,
    SpeedyRadiation,
    SurfaceState,
    ZonalMeanState,
    build_coupled_pe_stepper,
    compute_sst,
    compute_zonal_mean_state,
    exponential_filter,
    flat_continent_surface,
    grid_winds_at_level,
    init_land_state,
    moist_aquaplanet_initial_state,
    run_simulation,
    spinup_prescribed_sst,
    standard_sigma_levels,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_land_ocean(
    n_days: int = 600,
    spinup_days: int = 200,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    prescribed_spinup_days: int = 100,
    prescribed_averaging_days: int = 100,
    scheme: str = "byrne",
    clouds: bool = False,
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
    if scheme == "speedy":
        from notus.physics.clouds import CloudConfig

        config = PhysicsSuiteConfig(
            radiation=SpeedyRadiation(clouds=CloudConfig() if clouds else None),
            orbital=EARTH_ORBIT,
        )
    else:
        config = PhysicsSuiteConfig(radiation=ByrneRadiation(sw_tau_0=0.22), orbital=EARTH_ORBIT)
    forcing = PhysicsSuite(transform, EARTH, levels, config=config)

    # --- Surface configuration ---
    # Flat continent: 30S-60N, 0-180E (roughly Eurasia-like in extent)
    lat_deg = np.asarray(grid.latitudes_deg)
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
    spinup_forcing = PhysicsSuite(transform, EARTH, levels, config=config)

    spinup_result = spinup_prescribed_sst(
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
    print(
        f"  Q-flux range: [{float(jnp.min(spinup_result.q_flux)):.1f}, "
        f"{float(jnp.max(spinup_result.q_flux)):.1f}] W/m^2"
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
    days_per_year = EARTH_ORBIT.days_per_year

    init_fn, step_fn = build_coupled_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        forcing=forcing,
        ocean_config=ocean_config,
        q_flux=spinup_result.q_flux,
        surface_properties=sfc_props,
        land_config=land_config,
        spectral_filter=filt,
    )

    # --- Accumulator ---
    n_avg = 0
    accum_zm: ZonalMeanState | None = None
    blew_up = False

    eq_idx = np.argmin(np.abs(lat_deg))
    jet_level = max(0, n_levels // 4)
    land_mask = land_frac > 0.5

    def on_day(day: int, curr_state: object, sfc: object, _diags: object) -> None:
        nonlocal n_avg, accum_zm, blew_up

        if day > spinup_days:
            zm = compute_zonal_mean_state(curr_state, transform)  # type: ignore[arg-type]
            n_avg += 1
            accum_zm = zm if accum_zm is None else jax.tree.map(np.add, accum_zm, zm)

        if day <= 10 or day % 50 == 0 or day == n_days:
            t_grid = np.asarray(
                jax.vmap(transform.spectral_to_grid)(curr_state.temperature)  # type: ignore[union-attr]
            )
            sst = np.asarray(sfc.ocean.surface_temperature)  # type: ignore[union-attr]

            u_grid, _v = grid_winds_at_level(curr_state, jet_level, transform)  # type: ignore[arg-type]
            u_grid = np.asarray(u_grid)

            t_mean = float(np.mean(t_grid))
            phase = "spinup" if day <= spinup_days else "averaging"

            # Land diagnostics
            land_t = np.asarray(sfc.land.soil_temperature)  # type: ignore[union-attr]
            bucket = np.asarray(sfc.land.bucket_depth)  # type: ignore[union-attr]
            land_t_mean = float(np.mean(land_t[land_mask])) if np.any(land_mask) else 0.0
            bucket_mean = float(np.mean(bucket[land_mask])) if np.any(land_mask) else 0.0

            q_str = ""
            if curr_state.has_humidity:  # type: ignore[union-attr]
                q_grid = np.asarray(
                    jax.vmap(transform.spectral_to_grid)(curr_state.humidity)  # type: ignore[union-attr]
                )
                q_mean_gkg = float(np.mean(q_grid)) * 1000
                q_str = f"  q={q_mean_gkg:.2f}g/kg"

            print(
                f"  Day {day:5d} [{phase:>9s}]: "
                f"SST=[{np.min(sst):.1f},{np.max(sst):.1f}] "
                f"T_land={land_t_mean:.1f} K "
                f"bucket={bucket_mean:.3f}m "
                f"|U|={np.max(np.abs(u_grid)):.1f}"
                f"{q_str}"
            )

            if not np.isfinite(t_mean) or not np.all(np.isfinite(sst)):
                print("ERROR: Integration has blown up!")
                blew_up = True

    # --- Run ---
    print("\n--- Coupled integration ---")
    result = run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=spinup_result.state,
        dt=dt,
        n_days=n_days,
        surface=surface,
        forcing=forcing,
        days_per_year=days_per_year,
        on_day=on_day,
        verbose=False,
    )

    if blew_up:
        return False

    print(f"\nDone. Total wall time: {result.wall_time:.0f}s ({result.wall_time / 3600:.1f}h)")
    print(f"Averaged over {n_avg} daily samples")

    # --- Final summary ---
    final_surface = result.surface
    if final_surface is None:
        return False
    sst_final = np.asarray(final_surface.ocean.surface_temperature)
    land_t_final = np.asarray(final_surface.land.soil_temperature)
    bucket_final = np.asarray(final_surface.land.bucket_depth)

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
        mean_zm = jax.tree.map(lambda x: x / n_avg, accum_zm)

        sigma_full = np.asarray(levels.sigma_full)
        surface_idx = np.argmax(sigma_full)
        upper_mask = sigma_full < 0.4

        print("\n--- Atmospheric climatology ---")
        print(f"  T(equator, sfc) = {mean_zm.temperature[surface_idx, eq_idx]:.1f} K")
        pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))
        print(f"  T(pole, sfc)    = {mean_zm.temperature[surface_idx, pole_idx]:.1f} K")
        jet_max = float(np.max(np.abs(mean_zm.u[upper_mask, :])))
        print(f"  Jet max |U|     = {jet_max:.1f} m/s")
        eke = mean_zm.eke
        print(f"  EKE max         = {float(np.max(eke)):.1f} m^2/s^2")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Land-ocean coupled aquaplanet (bucket hydrology)")
    parser.add_argument("--days", type=int, default=600, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=200, help="Coupled spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--scheme", type=str, default="byrne", choices=["byrne", "speedy"])
    parser.add_argument("--clouds", action="store_true", help="Enable diagnostic clouds (speedy)")
    parser.add_argument(
        "--prescribe-spinup", type=int, default=100, help="Prescribed-SST spinup days"
    )
    parser.add_argument(
        "--prescribe-avg",
        type=int,
        default=100,
        help="Prescribed-SST averaging days for Q-flux",
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
        scheme=args.scheme,
        clouds=args.clouds,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
