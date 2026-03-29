"""Land-ocean planet with Gaussian mountains — visualization demo.

Showcases coupled atmosphere-ocean-land dynamics with topography:

- Slab ocean with diagnosed Q-flux from prescribed-SST spinup
- Bucket land surface model over a continent (30°S-60°N, 0°-120°E)
- Two Gaussian mountains: midlatitude (45°N, 60°E, 3000 m) and
  tropical (10°S, 90°E, 2000 m)
- Byrne humidity-dependent radiation with shortwave

Produces an 8-panel figure demonstrating the full visualization suite:
maps of orography, land fraction, surface pressure, soil temperature,
precipitation; zonal-mean wind, streamfunction; and KE spectrum.

Usage
-----
    uv run python examples/demo_land_ocean_mountains.py
    uv run python examples/demo_land_ocean_mountains.py --days 400 --spinup 100
"""

from __future__ import annotations

import argparse
import logging

import cartopy.crs as ccrs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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
    PrimitiveEquationState,
    SlabOceanConfig,
    SpectralTransform,
    SurfaceState,
    ZonalMeanState,
    build_coupled_pe_stepper,
    compute_ke_spectrum,
    compute_sst,
    compute_streamfunction,
    compute_zonal_mean_state,
    exponential_filter,
    flat_continent_surface,
    gaussian_mountain,
    init_land_state,
    moist_aquaplanet_initial_state,
    orographic_log_surface_pressure,
    plot_map,
    plot_spectrum,
    plot_zonal_mean,
    run_simulation,
    smooth_orography,
    spinup_prescribed_sst,
    standard_sigma_levels,
    state_to_dataset,
    zonal_mean_to_dataset,
)
from notus.physics.forcing import PhysicsDiagnostics


logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Land-ocean-mountains visualization demo")
    parser.add_argument("--days", type=int, default=300, help="Total coupled days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days (after Q-flux)")
    parser.add_argument("--spinup-qflux", type=int, default=200, help="Prescribed-SST spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--output", type=str, default="demo_land_ocean_mountains.png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Save precipitation animation as GIF",
    )
    args = parser.parse_args()

    # --- Grid and physics ---
    grid = GaussianGrid(truncation=args.truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(args.levels)

    radiation = ByrneRadiation()
    cfg = PhysicsSuiteConfig(
        radiation=radiation,
        orbital=EARTH_ORBIT,
        implicit_surface=True,
    )
    forcing = PhysicsSuite(transform, EARTH, levels, config=cfg)

    # --- Topography: two Gaussian mountains ---
    mountain_1 = gaussian_mountain(
        transform,
        EARTH,
        height=3000.0,
        center_lat=np.deg2rad(45.0),
        center_lon=np.deg2rad(60.0),
        half_width=np.deg2rad(15.0),
    )
    mountain_2 = gaussian_mountain(
        transform,
        EARTH,
        height=2000.0,
        center_lat=np.deg2rad(-10.0),
        center_lon=np.deg2rad(90.0),
        half_width=np.deg2rad(12.0),
    )
    surface_phi = smooth_orography(mountain_1 + mountain_2, transform.arrays)

    # --- Initial conditions ---
    state, ref_temps, _flat_phi = moist_aquaplanet_initial_state(
        transform,
        EARTH,
        levels,
        initial_rh=0.7,
        seed=42,
    )
    # Adjust surface pressure for topography
    lnps_topo = orographic_log_surface_pressure(surface_phi, transform, EARTH, 264.0)
    state = state.replace(log_surface_pressure=lnps_topo)

    # --- Surface properties (continent: 30°S-60°N, 0°-120°E) ---
    lat_deg = np.asarray(grid.latitudes_deg)
    lon_deg = np.degrees(np.asarray(grid.longitudes))
    surface_properties = flat_continent_surface(
        lat_deg,
        lon_deg,
        lat_south=-30.0,
        lat_north=60.0,
        lon_west=0.0,
        lon_east=120.0,
    )

    # --- Q-flux spinup (prescribed SST) ---
    filt = exponential_filter(transform.arrays, args.dt)
    print(f"Prescribed-SST spinup: {args.spinup_qflux} days...")
    spinup_result = spinup_prescribed_sst(
        state,
        forcing,
        transform,
        EARTH,
        levels,
        ref_temps,
        surface_phi,
        args.dt,
        spinup_days=args.spinup_qflux // 2,
        averaging_days=args.spinup_qflux - args.spinup_qflux // 2,
        spectral_filter=filt,
        surface_albedo=surface_properties.albedo,
        verbose=True,
    )
    state = spinup_result.state
    q_flux = spinup_result.q_flux
    print(f"  Q-flux range: [{float(jnp.min(q_flux)):.1f}, {float(jnp.max(q_flux)):.1f}] W/m²")

    # --- Initial surface state ---
    land_config = BucketLandConfig()
    sst = compute_sst(PrescribedSST(), transform.grid.latitudes)
    ocean = OceanState(surface_temperature=sst)
    land = init_land_state(surface_properties.land_fraction, sst, land_config)
    surface = SurfaceState(ocean=ocean, land=land)

    # --- Build coupled stepper ---
    ocean_config = SlabOceanConfig()
    init_fn, step_fn = build_coupled_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=args.dt,
        forcing=forcing,
        ocean_config=ocean_config,
        q_flux=q_flux,
        surface_properties=surface_properties,
        land_config=land_config,
        spectral_filter=filt,
    )

    # --- Diagnostic callback ---
    n_samples = 0
    accum_zm: ZonalMeanState | None = None
    accum_precip: np.ndarray | None = None
    precip_snapshots: list[xr.DataArray] = []
    sigma = np.asarray(levels.sigma_full)

    def on_day(
        day: int,
        curr: PrimitiveEquationState,
        sfc: SurfaceState,
        diags: PhysicsDiagnostics,
    ) -> None:
        nonlocal n_samples, accum_zm, accum_precip

        if day <= args.spinup:
            return

        n_samples += 1
        zm = compute_zonal_mean_state(curr, transform)
        accum_zm = zm if accum_zm is None else jax.tree.map(np.add, accum_zm, zm)

        if diags.precipitation is not None:
            precip = np.asarray(diags.precipitation)
            accum_precip = precip if accum_precip is None else accum_precip + precip

            # Collect precipitation snapshots for animation (every 5 days)
            if (day - args.spinup) % 5 == 0:
                precip_mm = precip * 86400.0  # kg/m²/s -> mm/day
                precip_snapshots.append(
                    xr.DataArray(
                        precip_mm,
                        dims=["lat", "lon"],
                        coords={
                            "lat": lat_deg,
                            "lon": lon_deg,
                        },
                        name="precipitation",
                    ),
                )

    # --- Run ---
    print(
        f"\nCoupled run: {args.days} days "
        f"({args.spinup} spinup + {args.days - args.spinup} averaging)"
    )
    result = run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=state,
        dt=args.dt,
        n_days=args.days,
        surface=surface,
        forcing=forcing,
        days_per_year=365.25,
        on_day=on_day,
        verbose=True,
    )
    print(f"Done in {result.wall_time:.0f}s, averaged {n_samples} samples.")

    if accum_zm is None or n_samples == 0:
        print("No samples collected!")
        return

    # --- Compute time-averages ---
    mean_zm = jax.tree.map(lambda x: x / n_samples, accum_zm)
    psi = compute_streamfunction(mean_zm.v, grid.cos_lat, levels, EARTH)
    zm_ds = zonal_mean_to_dataset(
        mean_zm,
        lat_deg,
        sigma,
        streamfunction=np.asarray(psi),
    )

    # Final state dataset (with orography and surface properties)
    final_surface = result.surface
    ds = state_to_dataset(
        result.state,
        transform,
        EARTH,
        levels,
        surface=final_surface,
        surface_geopotential=surface_phi,
        surface_properties=surface_properties,
    )

    # Mean precipitation (kg/m²/s → mm/day)
    mean_precip_mm = None
    if accum_precip is not None:
        mean_precip_mm = (accum_precip / n_samples) * 86400.0

    # KE spectrum
    wn, ke = compute_ke_spectrum(result.state, transform, levels)

    # --- Build 4×2 figure ---
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        f"Land-Ocean Planet with Mountains  |  T{args.truncation} L{args.levels}  |  "
        f"Days {args.spinup + 1}-{args.days} average",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1: Orography, land fraction
    ax1 = fig.add_subplot(4, 2, 1, projection=ccrs.Robinson())
    plot_map(ds["orography"], ax=ax1)

    ax2 = fig.add_subplot(4, 2, 2, projection=ccrs.Robinson())
    plot_map(ds["land_fraction"], ax=ax2)

    # Row 2: Zonal-mean u, streamfunction
    ax3 = fig.add_subplot(4, 2, 3)
    plot_zonal_mean(zm_ds["u"], ax=ax3)

    ax4 = fig.add_subplot(4, 2, 4)
    plot_zonal_mean(zm_ds["streamfunction"], ax=ax4)

    # Row 3: Surface pressure, soil temperature
    ax5 = fig.add_subplot(4, 2, 5, projection=ccrs.Robinson())
    plot_map(ds["surface_pressure"], ax=ax5)

    ax6 = fig.add_subplot(4, 2, 6, projection=ccrs.Robinson())
    if final_surface is not None and final_surface.land is not None:
        soil_t_da = xr.DataArray(
            np.asarray(final_surface.land.soil_temperature),
            dims=["lat", "lon"],
            coords={"lat": ds["lat"], "lon": ds["lon"]},
            name="soil_temperature",
        )
        plot_map(
            soil_t_da,
            ax=ax6,
            cmap="Spectral_r",
            title="Soil Temperature",
            colorbar_label="K",
        )

    # Row 4: Precipitation, KE spectrum
    ax7 = fig.add_subplot(4, 2, 7, projection=ccrs.Robinson())
    if mean_precip_mm is not None:
        precip_da = xr.DataArray(
            mean_precip_mm,
            dims=["lat", "lon"],
            coords={"lat": ds["lat"], "lon": ds["lon"]},
            name="precipitation",
        )
        plot_map(
            precip_da,
            ax=ax7,
            cmap="Blues",
            levels=np.linspace(0, 12, 13),
            extend="max",
            title="Mean Precipitation",
            colorbar_label="mm/day",
        )

    ax8 = fig.add_subplot(4, 2, 8)
    plot_spectrum(wn, ke, ax=ax8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {args.output}")

    # --- Animation: precipitation evolution ---
    if args.animate and len(precip_snapshots) > 1:
        from notus import animate_field

        anim_file = args.output.rsplit(".", 1)[0] + "_precip_anim.gif"
        print(f"Generating precipitation animation ({len(precip_snapshots)} frames)...")
        anim_fig, anim = animate_field(
            precip_snapshots,
            plot_fn="map",
            title_fmt="Day {i}",
            interval=150,
            cmap="Blues",
            levels=np.linspace(0, 15, 16),
            extend="max",
            colorbar_label="mm/day",
        )
        anim.save(anim_file, writer="pillow", fps=6)
        plt.close(anim_fig)
        print(f"Saved to {anim_file}")


if __name__ == "__main__":
    main()
