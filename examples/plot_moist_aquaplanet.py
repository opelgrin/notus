#!/usr/bin/env python3
"""Generate moist aquaplanet diagnostic visualizations.

Runs a moist aquaplanet integration and produces six-panel diagnostics:

1. Zonal-mean zonal wind U(lat, sigma)
2. Zonal-mean temperature T(lat, sigma)
3. Zonal-mean specific humidity q(lat, sigma)
4. Zonal-mean eddy kinetic energy EKE(lat, sigma)
5. Global-mean temperature time series (spinup diagnostic)
6. Instantaneous surface pressure map (Mollweide projection)

Usage
-----
    uv run python examples/plot_moist_aquaplanet.py
    uv run python examples/plot_moist_aquaplanet.py --days 500 --spinup 200
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
    SigmaLevels,
    SpectralTransform,
    ZonalMeanState,
    build_pe_stepper,
    compute_zonal_mean_state,
    exponential_filter,
    moist_aquaplanet_initial_state,
    run_simulation,
    standard_sigma_levels,
    state_to_dataset,
)
from notus.viz import plot_map, plot_zonal_mean, zonal_mean_to_dataset


logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def run_integration(
    truncation: int,
    n_levels: int,
    dt: float,
    n_days: int,
    spinup_days: int,
) -> tuple[
    ZonalMeanState,
    np.ndarray,
    PrimitiveEquationState,
    list[float],
    SpectralTransform,
    GaussianGrid,
    SigmaLevels,
]:
    """Run moist aquaplanet and return diagnostics for plotting."""
    print(
        f"Running T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days "
        f"({spinup_days} spinup + {n_days - spinup_days} averaging)"
    )

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

    forcing = PhysicsSuite(transform, EARTH, levels)
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

    # --- Diagnostic callback ---
    n_samples = 0
    accum: ZonalMeanState | None = None
    accum_q: np.ndarray | None = None
    daily_mean_t: list[float] = []
    weights = np.asarray(grid.lat_weights)
    w_sum = float(np.sum(weights))
    blew_up = False

    def on_day(day: int, curr: PrimitiveEquationState, _diags: object) -> None:
        nonlocal n_samples, accum, accum_q, blew_up

        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
        if not np.all(np.isfinite(t_grid)):
            print("ERROR: Integration has blown up!")
            blew_up = True
            return

        t_global = float(np.sum(np.mean(t_grid, axis=(0, -1)) * weights) / w_sum)
        daily_mean_t.append(t_global)

        if day > spinup_days:
            zm = compute_zonal_mean_state(curr, transform)
            n_samples += 1

            # Zonal-mean humidity (computed separately)
            if curr.humidity is not None:
                q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
                q_zm = np.mean(q_grid, axis=-1)  # (n_levels, n_lat)
            else:
                q_zm = np.zeros_like(np.asarray(zm.u))

            if accum is None:
                accum = zm
                accum_q = q_zm
            else:
                accum = jax.tree.map(np.add, accum, zm)
                accum_q = accum_q + q_zm

        if day <= 10 or day % 50 == 0 or day == n_days:
            q_info = ""
            if curr.humidity is not None:
                q_g = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
                q_info = f"  q_mean={float(np.mean(q_g)) * 1000:.2f}g/kg"
            print(f"  Day {day:5d}: T_mean={t_global:.1f} K{q_info}")

    # --- Run ---
    result = run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=state,
        dt=dt,
        n_days=n_days,
        on_day=on_day,
        verbose=False,
    )

    if blew_up:
        msg = "Integration blew up"
        raise RuntimeError(msg)

    print(f"Done in {result.wall_time:.0f}s. Averaged {n_samples} samples.")

    if accum is None or accum_q is None:
        msg = "No averaging samples collected"
        raise RuntimeError(msg)

    mean_zm = jax.tree.map(lambda x: x / n_samples, accum)
    mean_q_zm = accum_q / n_samples

    return mean_zm, mean_q_zm, result.state, daily_mean_t, transform, grid, levels


# ---------------------------------------------------------------------------
# Plotting helpers (timeseries only — zonal/map plots use notus.viz)
# ---------------------------------------------------------------------------


def plot_temperature_timeseries(
    daily_mean_t: list[float],
    spinup_days: int,
    ax: plt.Axes,
) -> None:
    """Global-mean temperature vs time."""
    days = np.arange(1, len(daily_mean_t) + 1)
    ax.plot(days, daily_mean_t, "k-", linewidth=0.8)
    ax.axvline(spinup_days, color="r", linestyle="--", linewidth=0.8, label="End of spinup")
    ax.set_xlabel("Day")
    ax.set_ylabel("Global-Mean Temperature [K]")
    ax.set_title("Spinup: Global-Mean Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Moist aquaplanet visualizations")
    parser.add_argument("--days", type=int, default=300, help="Total days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument(
        "--output",
        type=str,
        default="moist_aquaplanet.png",
        help="Output filename",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    args = parser.parse_args()

    mean_zm, mean_q_zm, final_state, daily_mean_t, transform, grid, levels = run_integration(
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        n_days=args.days,
        spinup_days=args.spinup,
    )

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(
        f"Moist Aquaplanet  |  T{args.truncation} L{args.levels}  |  "
        f"Days {args.spinup + 1}-{args.days} average",
        fontsize=14,
        fontweight="bold",
    )

    # Convert to xarray for viz module
    lat_deg = np.asarray(grid.latitudes_deg)
    sigma = np.asarray(levels.sigma_full)
    zm_ds = zonal_mean_to_dataset(mean_zm, lat_deg, sigma)
    zm_ds["specific_humidity"] = xr.DataArray(
        np.asarray(mean_q_zm),
        dims=["sigma", "lat"],
        name="specific_humidity",
    )
    ds = state_to_dataset(final_state, transform, EARTH, levels)

    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6, projection=ccrs.Mollweide())

    plot_zonal_mean(zm_ds["u"], ax=ax1)
    plot_zonal_mean(zm_ds["temperature"], ax=ax2, contour_labels=True)
    plot_zonal_mean(zm_ds["specific_humidity"], ax=ax3, contour_labels=True)
    plot_zonal_mean(zm_ds["eke"], ax=ax4)
    plot_temperature_timeseries(daily_mean_t, args.spinup, ax5)
    plot_map(ds["surface_pressure"], ax=ax6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
