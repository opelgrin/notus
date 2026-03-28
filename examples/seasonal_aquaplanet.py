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
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.solar import EARTH_ORBIT
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


def run_seasonal_aquaplanet(
    n_days: int = 730,
    spinup_days: int = 365,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
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

    # Frierson gray LW + seasonal insolation (no atmospheric SW absorption)
    config = SimplePhysicsConfig(
        radiation_scheme="frierson",
        orbital=EARTH_ORBIT,
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

    def one_day(
        carry: tuple[jnp.ndarray, ...],
        day_of_year: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, ...], None]:
        prev, curr = carry
        forcing.day_of_year = day_of_year

        def step(
            carry: tuple[jnp.ndarray, ...],
            _: None,
        ) -> tuple[tuple[jnp.ndarray, ...], None]:
            p, c = carry
            p, c = step_fn(p, c)
            return (p, c), None

        (prev, curr), _ = jax.lax.scan(step, (prev, curr), None, length=steps_per_day)
        return (prev, curr), None

    one_day_jit = jax.jit(one_day)

    # --- Initialize ---
    print("Initializing...")
    t0 = time.perf_counter()
    prev, curr = init_fn(state)
    (prev, curr), _ = one_day_jit((prev, curr), jnp.float64(0.0))
    t_compile = time.perf_counter() - t0
    print(f"Day 1 (incl. JIT compile): {t_compile:.1f}s")

    lat_deg = np.degrees(np.asarray(grid.latitudes))
    eq_idx = np.argmin(np.abs(lat_deg))
    nh_mid_idx = np.argmin(np.abs(lat_deg - 45.0))
    sh_mid_idx = np.argmin(np.abs(lat_deg + 45.0))

    # Monthly accumulator for seasonal diagnostics
    month_t_sum = np.zeros(len(lat_deg))
    month_u_sum = np.zeros(len(lat_deg))
    month_days = 0
    month_number = 0
    jet_level = max(0, n_levels // 4)

    def _flush_month(label: str) -> None:
        nonlocal month_t_sum, month_u_sum, month_days, month_number
        if month_days == 0:
            return
        t_m = month_t_sum / month_days
        u_m = month_u_sum / month_days
        month_number += 1
        print(
            f"    Month {month_number:2d} ({label}): "
            f"T=[{t_m[eq_idx]:.1f} eq, {t_m[nh_mid_idx]:.1f} 45N, {t_m[sh_mid_idx]:.1f} 45S] K  "
            f"|U|_max={np.max(np.abs(u_m)):.1f} m/s"
        )
        month_t_sum[:] = 0
        month_u_sum[:] = 0
        month_days = 0

    # --- Main integration loop ---
    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        day_of_year = jnp.float64(day % days_per_year)
        (prev, curr), _ = one_day_jit((prev, curr), day_of_year)

        if day > spinup_days:
            # Accumulate monthly surface temperature and jet
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            surface_idx = n_levels - 1
            month_t_sum += t_grid[surface_idx].mean(axis=-1)

            u_spec, _ = uv_from_vordiv(
                curr.vorticity[jet_level],
                curr.divergence[jet_level],
                transform.arrays,
            )
            u_grid = np.asarray(transform.spectral_to_grid(u_spec))
            cos_lat = np.asarray(grid.cos_lat)
            month_u_sum += (u_grid / cos_lat[:, None]).mean(axis=-1)
            month_days += 1

            if month_days == 30:
                phase = "spinup" if day <= spinup_days else "averaging"
                _flush_month(phase)

        if day <= 5 or day % 100 == 0:
            elapsed = time.perf_counter() - t_start
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            t_mean = float(np.mean(t_grid))
            days_per_sec = (day - 1) / elapsed if elapsed > 0 else 0
            phase = "spinup" if day <= spinup_days else "averaging"
            print(
                f"  Day {day:5d} [{phase:>9s}]: T_mean={t_mean:.1f} K  [{days_per_sec:.1f} days/s]"
            )
            if not np.isfinite(t_mean):
                print("ERROR: Integration has blown up!")
                return False

    # Flush remaining partial month
    if month_days > 0:
        _flush_month("averaging")

    total_time = time.perf_counter() - t0
    print(f"\nDone. Total wall time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal aquaplanet (gray radiation)")
    parser.add_argument("--days", type=int, default=730, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=365, help="Spinup days before averaging")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    args = parser.parse_args()

    passed = run_seasonal_aquaplanet(
        n_days=args.days,
        spinup_days=args.spinup,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
