#!/usr/bin/env python3
"""Generate Frierson aquaplanet diagnostic visualizations.

Runs a physics suite integration and produces diagnostic plots:

1. Zonal-mean zonal wind U(lat, sigma)
2. Zonal-mean temperature T(lat, sigma)
3. Zonal-mean eddy kinetic energy EKE(lat, sigma)
4. Global-mean temperature time series (spinup diagnostic)
5. Instantaneous surface pressure map (Mollweide projection)

Usage
-----
    uv run python examples/plot_physics_suite.py
    uv run python examples/plot_physics_suite.py --truncation 42 --days 600 --dt 600
"""

from __future__ import annotations

import argparse
import logging

import cartopy.crs as ccrs
import jax
import matplotlib.pyplot as plt
import numpy as np


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
    grid_surface_pressure,
    physics_suite_initial_state,
    run_simulation,
    uniform_sigma_levels,
)


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
    PrimitiveEquationState,
    list[float],
    SpectralTransform,
    GaussianGrid,
    SigmaLevels,
]:
    """Run a physics suite integration and return diagnostics for plotting."""
    print(
        f"Running T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days "
        f"({spinup_days} spinup + {n_days - spinup_days} averaging)"
    )

    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = physics_suite_initial_state(
        transform,
        EARTH,
        levels,
        perturbation_amplitude=1.0,
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
    daily_mean_t: list[float] = []
    weights = np.asarray(grid.lat_weights)
    w_sum = float(np.sum(weights))
    blew_up = False

    def on_day(day: int, curr: PrimitiveEquationState) -> None:
        nonlocal n_samples, accum, blew_up

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
            accum = zm if accum is None else jax.tree.map(np.add, accum, zm)

        if day <= 10 or day % 50 == 0 or day == n_days:
            print(f"  Day {day:5d}: T_mean={t_global:.1f} K")

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

    if accum is None:
        msg = "No averaging samples collected"
        raise RuntimeError(msg)
    mean_zm = jax.tree.map(lambda x: x / n_samples, accum)

    return mean_zm, result.state, daily_mean_t, transform, grid, levels


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_zonal_mean_u(
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
    ax: plt.Axes,
) -> None:
    """Zonal-mean zonal wind U(lat, sigma)."""
    lat = np.asarray(grid.latitudes_deg)
    sigma = np.asarray(levels.sigma_full)
    u = np.asarray(mean_zm.u)

    clevels = np.arange(-10, 36, 5)
    cf = ax.contourf(lat, sigma, u, levels=clevels, cmap="RdBu_r", extend="both")
    ax.contour(lat, sigma, u, levels=clevels, colors="k", linewidths=0.3)
    ax.contour(lat, sigma, u, levels=[0], colors="k", linewidths=1.0)
    plt.colorbar(cf, ax=ax, label="m/s")
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Sigma")
    ax.set_title("Zonal-Mean Zonal Wind")


def plot_zonal_mean_t(
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
    ax: plt.Axes,
) -> None:
    """Zonal-mean temperature T(lat, sigma)."""
    lat = np.asarray(grid.latitudes_deg)
    sigma = np.asarray(levels.sigma_full)
    t = np.asarray(mean_zm.temperature)

    clevels = np.arange(190, 315, 10)
    cf = ax.contourf(lat, sigma, t, levels=clevels, cmap="Spectral_r", extend="both")
    cs = ax.contour(lat, sigma, t, levels=clevels, colors="k", linewidths=0.3)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    plt.colorbar(cf, ax=ax, label="K")
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Sigma")
    ax.set_title("Zonal-Mean Temperature")


def plot_zonal_mean_eke(
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
    ax: plt.Axes,
) -> None:
    """Zonal-mean eddy kinetic energy EKE(lat, sigma)."""
    lat = np.asarray(grid.latitudes_deg)
    sigma = np.asarray(levels.sigma_full)
    eke = np.asarray(mean_zm.eke)

    clevels = np.arange(0, 175, 25)
    cf = ax.contourf(lat, sigma, eke, levels=clevels, cmap="YlOrRd", extend="max")
    ax.contour(lat, sigma, eke, levels=clevels, colors="k", linewidths=0.3)
    plt.colorbar(cf, ax=ax, label="m$^2$/s$^2$")
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Sigma")
    ax.set_title("Zonal-Mean Eddy Kinetic Energy")


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


def plot_surface_pressure_snapshot(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    grid: GaussianGrid,
    ax: plt.Axes,
) -> None:
    """Instantaneous surface pressure on a Mollweide projection."""
    ps_hpa = np.asarray(grid_surface_pressure(state, transform, EARTH)) / 100.0

    lon = np.degrees(np.asarray(grid.longitudes))
    lat = np.asarray(grid.latitudes_deg)
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    clevels = np.arange(970, 1035, 5)
    cf = ax.contourf(
        lon_2d,
        lat_2d,
        ps_hpa,
        levels=clevels,
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    ax.contour(
        lon_2d,
        lat_2d,
        ps_hpa,
        levels=clevels,
        colors="k",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines(linewidth=0.5, color="gray")
    ax.set_global()
    plt.colorbar(cf, ax=ax, label="hPa", orientation="horizontal", pad=0.05, shrink=0.8)
    ax.set_title("Surface Pressure (Snapshot)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple physics aquaplanet visualizations")
    parser.add_argument("--days", type=int, default=300, help="Total days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=600.0, help="Timestep [s]")
    parser.add_argument("--output", type=str, default="physics_suite.png", help="Output filename")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    args = parser.parse_args()

    mean_zm, final_state, daily_mean_t, transform, grid, levels = run_integration(
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        n_days=args.days,
        spinup_days=args.spinup,
    )

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Frierson Aquaplanet  |  T{args.truncation} L{args.levels}  |  "
        f"Days {args.spinup + 1}-{args.days} average",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 1, 3, projection=ccrs.Mollweide())

    plot_zonal_mean_u(mean_zm, grid, levels, ax1)
    plot_zonal_mean_t(mean_zm, grid, levels, ax2)
    plot_zonal_mean_eke(mean_zm, grid, levels, ax3)
    plot_temperature_timeseries(daily_mean_t, args.spinup, ax4)
    plot_surface_pressure_snapshot(final_state, transform, grid, ax5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
