#!/usr/bin/env python3
"""Moist aquaplanet — full visualization suite demonstration.

Showcases the complete Notus visualization and diagnostics pipeline:

- Zonal-mean cross-sections: zonal wind, temperature, humidity, streamfunction
- Global maps: surface pressure, time-mean precipitation
- Kinetic energy spectrum with reference slopes
- Hovmöller diagram of equatorial surface pressure

Usage
-----
    uv run python examples/demo_moist_aquaplanet.py
    uv run python examples/demo_moist_aquaplanet.py --days 400 --spinup 150
"""

from __future__ import annotations

import argparse
import logging

import cartopy.crs as ccrs
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


jax.config.update("jax_enable_x64", True)

from notus import (
    EARTH,
    GaussianGrid,
    PhysicsSuite,
    PrimitiveEquationState,
    SpectralTransform,
    ZonalMeanState,
    build_pe_stepper,
    compute_ke_spectrum,
    compute_streamfunction,
    compute_zonal_mean_state,
    exponential_filter,
    moist_aquaplanet_initial_state,
    plot_hovmoller,
    plot_map,
    plot_spectrum,
    plot_zonal_mean,
    run_simulation,
    standard_sigma_levels,
    state_to_dataset,
    zonal_mean_to_dataset,
)
from notus.physics.forcing import PhysicsDiagnostics


logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moist aquaplanet visualization demo")
    parser.add_argument("--days", type=int, default=300, help="Total days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--output", type=str, default="demo_moist_aquaplanet.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    # --- Setup ---
    grid = GaussianGrid(truncation=args.truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(args.levels)

    state, ref_temps, surface_phi = moist_aquaplanet_initial_state(
        transform,
        EARTH,
        levels,
        initial_rh=0.7,
        seed=42,
    )
    forcing = PhysicsSuite(transform, EARTH, levels)
    filt = exponential_filter(transform.arrays, args.dt)
    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=args.dt,
        spectral_filter=filt,
        forcing=forcing,
    )

    # --- Diagnostic callback ---
    n_samples = 0
    accum_zm: ZonalMeanState | None = None
    accum_q_zm: np.ndarray | None = None
    accum_precip: np.ndarray | None = None
    ps_snapshots: list[xr.DataArray] = []
    lat_deg = np.asarray(grid.latitudes_deg)
    sigma = np.asarray(levels.sigma_full)

    def on_day(
        day: int,
        curr: PrimitiveEquationState,
        diags: PhysicsDiagnostics,
    ) -> None:
        nonlocal n_samples, accum_zm, accum_q_zm, accum_precip

        if day <= args.spinup:
            return

        n_samples += 1
        zm = compute_zonal_mean_state(curr, transform)
        accum_zm = zm if accum_zm is None else jax.tree.map(np.add, accum_zm, zm)

        # Humidity zonal mean
        if curr.humidity is not None:
            q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
            q_zm = np.mean(q_grid, axis=-1)
            accum_q_zm = q_zm if accum_q_zm is None else accum_q_zm + q_zm

        # Precipitation from diagnostics
        if diags.precipitation is not None:
            precip = np.asarray(diags.precipitation)
            accum_precip = precip if accum_precip is None else accum_precip + precip

        # Collect surface pressure snapshots for Hovmöller (every 5 days)
        if (day - args.spinup) % 5 == 0:
            ds = state_to_dataset(curr, transform, EARTH, levels)
            ps_snapshots.append(ds["surface_pressure"])

    # --- Run ---
    print(
        f"Running T{args.truncation} L{args.levels}, dt={args.dt:.0f}s, "
        f"{args.days} days ({args.spinup} spinup + {args.days - args.spinup} averaging)"
    )
    result = run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=state,
        dt=args.dt,
        n_days=args.days,
        on_day=on_day,
        verbose=True,
    )
    print(f"Done in {result.wall_time:.0f}s, averaged {n_samples} samples.")

    if accum_zm is None or n_samples == 0:
        print("No samples collected!")
        return

    # --- Compute time-averages ---
    mean_zm = jax.tree.map(lambda x: x / n_samples, accum_zm)
    zm_ds = zonal_mean_to_dataset(mean_zm, lat_deg, sigma)

    if accum_q_zm is not None:
        zm_ds["specific_humidity"] = xr.DataArray(
            accum_q_zm / n_samples,
            dims=["sigma", "lat"],
            name="specific_humidity",
        )

    # Streamfunction
    psi = compute_streamfunction(mean_zm.v, grid.cos_lat, levels, EARTH)
    zm_ds["streamfunction"] = xr.DataArray(
        np.asarray(psi),
        dims=["sigma", "lat"],
        name="streamfunction",
    )

    # Final state dataset
    ds = state_to_dataset(result.state, transform, EARTH, levels)

    # Mean precipitation (convert kg/m²/s → mm/day)
    mean_precip = None
    if accum_precip is not None:
        mean_precip = (accum_precip / n_samples) * 86400.0

    # KE spectrum
    wn, ke = compute_ke_spectrum(result.state, transform, levels)

    # --- Build 4×2 figure ---
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        f"Moist Aquaplanet  |  T{args.truncation} L{args.levels}  |  "
        f"Days {args.spinup + 1}-{args.days} average",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1: Zonal-mean u, T
    ax1 = fig.add_subplot(4, 2, 1)
    plot_zonal_mean(zm_ds["u"], ax=ax1)

    ax2 = fig.add_subplot(4, 2, 2)
    plot_zonal_mean(zm_ds["temperature"], ax=ax2, contour_labels=True)

    # Row 2: Zonal-mean q, streamfunction
    ax3 = fig.add_subplot(4, 2, 3)
    plot_zonal_mean(zm_ds["specific_humidity"], ax=ax3, contour_labels=True)

    ax4 = fig.add_subplot(4, 2, 4)
    plot_zonal_mean(zm_ds["streamfunction"], ax=ax4)

    # Row 3: Surface pressure map, precipitation map
    ax5 = fig.add_subplot(4, 2, 5, projection=ccrs.Mollweide())
    plot_map(ds["surface_pressure"], ax=ax5)

    ax6 = fig.add_subplot(4, 2, 6, projection=ccrs.Mollweide())
    if mean_precip is not None:
        precip_da = xr.DataArray(
            mean_precip,
            dims=["lat", "lon"],
            coords={"lat": ds["lat"], "lon": ds["lon"]},
            name="precipitation",
        )
        plot_map(
            precip_da,
            ax=ax6,
            cmap="Blues",
            levels=np.linspace(0, 12, 13),
            extend="max",
            title="Mean Precipitation",
            colorbar_label="mm/day",
        )
    else:
        ax6.set_title("Precipitation: N/A (dry run)")

    # Row 4: KE spectrum, Hovmöller
    ax7 = fig.add_subplot(4, 2, 7)
    plot_spectrum(wn, ke, ax=ax7)

    ax8 = fig.add_subplot(4, 2, 8)
    if len(ps_snapshots) > 1:
        # Build equatorial Hovmöller: select nearest latitude to equator
        eq_idx = int(np.argmin(np.abs(lat_deg)))
        hov_data = np.stack([np.asarray(s.values[eq_idx]) for s in ps_snapshots])
        hov_da = xr.DataArray(
            hov_data / 100.0,
            dims=["day", "lon"],
            coords={
                "day": np.arange(len(ps_snapshots)) * 5 + args.spinup,
                "lon": ds["lon"].values,
            },
            name="surface_pressure",
        )
        plot_hovmoller(
            hov_da,
            ax=ax8,
            cmap="RdBu_r",
            title="Equatorial Ps Hovmöller",
            colorbar_label="hPa",
        )
    else:
        ax8.set_title("Hovmöller: not enough snapshots")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
