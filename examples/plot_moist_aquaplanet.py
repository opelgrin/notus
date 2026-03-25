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
import time

import cartopy.crs as ccrs
import jax
import matplotlib.pyplot as plt
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus.constants import EARTH
from notus.diagnostics import ZonalMeanState, compute_zonal_mean_state
from notus.grid import GaussianGrid
from notus.initial_conditions import moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.physics.simple_physics import SimplePhysics
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels, uniform_sigma_levels


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
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = moist_aquaplanet_initial_state(
        transform,
        EARTH,
        levels,
        initial_rh=0.7,
        seed=42,
    )

    forcing = SimplePhysics(transform, EARTH, levels)
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

    t0 = time.perf_counter()
    prev, curr = init_fn(state)
    (prev, curr), _ = one_day_jit((prev, curr), None)
    print(f"  Day 1 (JIT compile): {time.perf_counter() - t0:.1f}s")

    n_samples = 0
    accum: ZonalMeanState | None = None
    accum_q: np.ndarray | None = None
    daily_mean_t: list[float] = []
    weights = np.asarray(grid.lat_weights)
    w_sum = float(np.sum(weights))

    t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
    t_global = float(np.sum(np.mean(t_grid, axis=(0, -1)) * weights) / w_sum)
    daily_mean_t.append(t_global)

    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        (prev, curr), _ = one_day_jit((prev, curr), None)

        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
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
                accum = ZonalMeanState(
                    u=np.asarray(zm.u),
                    v=np.asarray(zm.v),
                    temperature=np.asarray(zm.temperature),
                    u_prime_sq=np.asarray(zm.u_prime_sq),
                    v_prime_sq=np.asarray(zm.v_prime_sq),
                    uv_prime=np.asarray(zm.uv_prime),
                    vt_prime=np.asarray(zm.vt_prime),
                )
                accum_q = q_zm
            else:
                accum = ZonalMeanState(
                    u=accum.u + np.asarray(zm.u),
                    v=accum.v + np.asarray(zm.v),
                    temperature=accum.temperature + np.asarray(zm.temperature),
                    u_prime_sq=accum.u_prime_sq + np.asarray(zm.u_prime_sq),
                    v_prime_sq=accum.v_prime_sq + np.asarray(zm.v_prime_sq),
                    uv_prime=accum.uv_prime + np.asarray(zm.uv_prime),
                    vt_prime=accum.vt_prime + np.asarray(zm.vt_prime),
                )
                accum_q = accum_q + q_zm

        if day <= 10 or day % 50 == 0 or day == n_days:
            elapsed = time.perf_counter() - t_start
            speed = (day - 1) / elapsed if elapsed > 0 else 0
            q_info = ""
            if curr.humidity is not None:
                q_g = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
                q_info = f"  q_mean={float(np.mean(q_g)) * 1000:.2f}g/kg"
            print(f"  Day {day:5d}: T_mean={t_global:.1f} K{q_info}  [{speed:.1f} days/s]")

    total = time.perf_counter() - t0
    print(f"Done in {total:.0f}s. Averaged {n_samples} samples.")

    if accum is None or accum_q is None:
        msg = "No averaging samples collected"
        raise RuntimeError(msg)

    mean_zm = ZonalMeanState(
        u=accum.u / n_samples,
        v=accum.v / n_samples,
        temperature=accum.temperature / n_samples,
        u_prime_sq=accum.u_prime_sq / n_samples,
        v_prime_sq=accum.v_prime_sq / n_samples,
        uv_prime=accum.uv_prime / n_samples,
        vt_prime=accum.vt_prime / n_samples,
    )
    mean_q_zm = accum_q / n_samples

    return mean_zm, mean_q_zm, curr, daily_mean_t, transform, grid, levels


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_zonal_mean_u(
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
    ax: plt.Axes,
) -> None:
    """Zonal-mean zonal wind."""
    lat = np.degrees(np.asarray(grid.latitudes))
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
    """Zonal-mean temperature."""
    lat = np.degrees(np.asarray(grid.latitudes))
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


def plot_zonal_mean_q(
    mean_q_zm: np.ndarray,
    grid: GaussianGrid,
    levels: SigmaLevels,
    ax: plt.Axes,
) -> None:
    """Zonal-mean specific humidity."""
    lat = np.degrees(np.asarray(grid.latitudes))
    sigma = np.asarray(levels.sigma_full)
    q_gkg = np.asarray(mean_q_zm) * 1000  # convert to g/kg

    clevels = np.arange(0, 22, 2)
    cf = ax.contourf(lat, sigma, q_gkg, levels=clevels, cmap="YlGnBu", extend="max")
    cs = ax.contour(lat, sigma, q_gkg, levels=clevels, colors="k", linewidths=0.3)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    plt.colorbar(cf, ax=ax, label="g/kg")
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("Latitude [deg]")
    ax.set_ylabel("Sigma")
    ax.set_title("Zonal-Mean Specific Humidity")


def plot_zonal_mean_eke(
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
    ax: plt.Axes,
) -> None:
    """Zonal-mean eddy kinetic energy."""
    lat = np.degrees(np.asarray(grid.latitudes))
    sigma = np.asarray(levels.sigma_full)
    eke = 0.5 * (np.asarray(mean_zm.u_prime_sq) + np.asarray(mean_zm.v_prime_sq))

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
    lnps_grid = np.asarray(transform.spectral_to_grid(state.log_surface_pressure))
    ps_hpa = EARTH.reference_pressure * np.exp(lnps_grid) / 100.0

    lon = np.degrees(np.asarray(grid.longitudes))
    lat = np.degrees(np.asarray(grid.latitudes))
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
    parser = argparse.ArgumentParser(description="Moist aquaplanet visualizations")
    parser.add_argument("--days", type=int, default=300, help="Total days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=600.0, help="Timestep [s]")
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

    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6, projection=ccrs.Mollweide())

    plot_zonal_mean_u(mean_zm, grid, levels, ax1)
    plot_zonal_mean_t(mean_zm, grid, levels, ax2)
    plot_zonal_mean_q(mean_q_zm, grid, levels, ax3)
    plot_zonal_mean_eke(mean_zm, grid, levels, ax4)
    plot_temperature_timeseries(daily_mean_t, args.spinup, ax5)
    plot_surface_pressure_snapshot(final_state, transform, grid, ax6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
