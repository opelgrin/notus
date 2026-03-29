#!/usr/bin/env python3
"""Seasonal aquaplanet with gray radiation.

Demonstrates time-varying solar forcing driven by Earth's orbital
parameters (obliquity, eccentricity).  Uses the Frierson (2006) gray
radiation scheme with prescribed SST — the simplest configuration that
shows a seasonal cycle.

Prints monthly-mean diagnostics so the seasonal march of temperature
and jet position is visible in the output.

Usage
-----
    uv run python examples/seasonal_aquaplanet.py
    uv run python examples/seasonal_aquaplanet.py --days 730 --spinup 365
"""

from __future__ import annotations

import argparse
import logging
import sys

import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus import (
    EARTH,
    EARTH_ORBIT,
    GaussianGrid,
    SimplePhysics,
    SimplePhysicsConfig,
    SpectralTransform,
    build_pe_stepper,
    exponential_filter,
    grid_winds_at_level,
    moist_aquaplanet_initial_state,
    run_simulation,
    standard_sigma_levels,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_seasonal_aquaplanet(
    n_days: int = 730,
    spinup_days: int = 365,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    scheme: str = "frierson",
    clouds: bool = False,
) -> bool:
    """Run a seasonal aquaplanet with prescribed SST.

    Returns True if the integration completes without blowup.
    """
    days_per_year = EARTH_ORBIT.days_per_year
    print(f"Seasonal aquaplanet: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")
    print(
        f"  Orbital: obliquity={np.degrees(EARTH_ORBIT.obliquity):.2f} deg, "
        f"eccentricity={EARTH_ORBIT.eccentricity:.4f}"
    )

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

    if scheme == "speedy":
        config = SimplePhysicsConfig(
            radiation_scheme="speedy",
            orbital=EARTH_ORBIT,
            enable_clouds=clouds,
        )
    elif scheme == "byrne":
        config = SimplePhysicsConfig(
            radiation_scheme="byrne",
            sw_tau_0=0.22,
            orbital=EARTH_ORBIT,
        )
    else:
        config = SimplePhysicsConfig(radiation_scheme="frierson", orbital=EARTH_ORBIT)
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

    lat_deg = np.asarray(grid.latitudes_deg)
    eq_idx = np.argmin(np.abs(lat_deg))
    nh_mid_idx = np.argmin(np.abs(lat_deg - 45.0))
    sh_mid_idx = np.argmin(np.abs(lat_deg + 45.0))

    # Monthly accumulator for seasonal diagnostics
    month_t_sum = np.zeros(len(lat_deg))
    month_u_sum = np.zeros(len(lat_deg))
    month_days = 0
    month_number = 0
    jet_level = max(0, n_levels // 4)
    blew_up = False

    def _flush_month(label: str) -> None:
        nonlocal month_t_sum, month_u_sum, month_days, month_number
        if month_days == 0:
            return
        t_m = month_t_sum / month_days
        u_m = month_u_sum / month_days
        month_number += 1
        print(
            f"    Month {month_number:2d} ({label}): "
            f"T=[{t_m[eq_idx]:.1f} eq, {t_m[nh_mid_idx]:.1f} 45N, "
            f"{t_m[sh_mid_idx]:.1f} 45S] K  "
            f"|U|_max={np.max(np.abs(u_m)):.1f} m/s"
        )
        month_t_sum[:] = 0
        month_u_sum[:] = 0
        month_days = 0

    # --- Callback ---
    def on_day(day: int, curr: object) -> None:
        nonlocal month_t_sum, month_u_sum, month_days, blew_up

        if day > spinup_days:
            # Accumulate monthly surface temperature and jet
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))  # type: ignore[union-attr]
            surface_idx = n_levels - 1
            month_t_sum += t_grid[surface_idx].mean(axis=-1)

            u_grid, _v = grid_winds_at_level(curr, jet_level, transform)  # type: ignore[arg-type]
            month_u_sum += np.asarray(u_grid).mean(axis=-1)
            month_days += 1

            if month_days == 30:
                phase = "spinup" if day <= spinup_days else "averaging"
                _flush_month(phase)

        if day <= 5 or day % 100 == 0:
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))  # type: ignore[union-attr]
            t_mean = float(np.mean(t_grid))
            phase = "spinup" if day <= spinup_days else "averaging"
            print(f"  Day {day:5d} [{phase:>9s}]: T_mean={t_mean:.1f} K")
            if not np.isfinite(t_mean):
                print("ERROR: Integration has blown up!")
                blew_up = True

    # --- Run ---
    result = run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=state,
        dt=dt,
        n_days=n_days,
        forcing=forcing,
        days_per_year=days_per_year,
        on_day=on_day,
        verbose=False,
    )

    if blew_up:
        return False

    # Flush remaining partial month
    if month_days > 0:
        _flush_month("averaging")

    print(f"\nDone. Total wall time: {result.wall_time:.0f}s ({result.wall_time / 3600:.1f}h)")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal aquaplanet (gray radiation)")
    parser.add_argument("--days", type=int, default=730, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=365, help="Spinup days before averaging")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument(
        "--scheme",
        type=str,
        default="frierson",
        choices=["frierson", "byrne", "speedy"],
    )
    parser.add_argument("--clouds", action="store_true", help="Enable diagnostic clouds (speedy)")
    args = parser.parse_args()

    passed = run_seasonal_aquaplanet(
        n_days=args.days,
        spinup_days=args.spinup,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        scheme=args.scheme,
        clouds=args.clouds,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
