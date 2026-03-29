#!/usr/bin/env python3
"""Dry Frierson aquaplanet integration.

Runs an aquaplanet with gray longwave radiation, dry convective adjustment,
bulk surface sensible heat flux, Rayleigh boundary-layer drag, and prescribed
SST.  No moisture, no shortwave absorption (Frierson 2006 convention).
Starts from an isothermal rest state at T21 L20.

After a spinup period, accumulates time-averaged zonal-mean fields and prints
basic diagnostics.

Usage
-----
    uv run python examples/dry_aquaplanet.py
    uv run python examples/dry_aquaplanet.py --days 300 --spinup 100
    uv run python examples/dry_aquaplanet.py --truncation 42 --dt 600
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
    GaussianGrid,
    PhysicsSuite,
    PrimitiveEquationState,
    SpectralTransform,
    ZonalMeanState,
    build_pe_stepper,
    compute_zonal_mean_state,
    exponential_filter,
    grid_surface_pressure,
    grid_winds_at_level,
    physics_suite_initial_state,
    run_simulation,
    uniform_sigma_levels,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Main integration
# ---------------------------------------------------------------------------


def run_aquaplanet(
    n_days: int = 300,
    spinup_days: int = 100,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 600.0,
    output_path: str | None = None,
) -> bool:
    """Run a Frierson aquaplanet integration.

    Returns True if the integration completes without blowup.
    """
    print(f"Dry aquaplanet: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # --- Setup ---
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
    n_averaging_samples = 0
    accum_zm: ZonalMeanState | None = None
    blew_up = False

    def on_day(day: int, curr_state: PrimitiveEquationState) -> None:
        nonlocal n_averaging_samples, accum_zm, blew_up

        # Blowup check
        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr_state.temperature))
        t_mean = float(np.mean(t_grid))
        if not np.isfinite(t_mean):
            print("ERROR: Integration has blown up!")
            blew_up = True
            return

        # Accumulate after spinup
        if day > spinup_days:
            zm = compute_zonal_mean_state(curr_state, transform)
            n_averaging_samples += 1
            accum_zm = zm if accum_zm is None else jax.tree.map(np.add, accum_zm, zm)

        # Print status at intervals
        if day <= 10 or day % 50 == 0 or day == n_days:
            ps_grid = np.asarray(grid_surface_pressure(curr_state, transform, EARTH))
            jet_level = max(0, n_levels // 4)
            u_grid, _v = grid_winds_at_level(curr_state, jet_level, transform)
            u_grid = np.asarray(u_grid)
            phase = "spinup" if day <= spinup_days else "averaging"
            print(
                f"  Day {day:5d} [{phase:>9s}]: "
                f"ps=[{np.min(ps_grid) / 100:.1f}, {np.max(ps_grid) / 100:.1f}] hPa  "
                f"T=[{np.min(t_grid):.1f}, {np.max(t_grid):.1f}] K (mean {t_mean:.1f})  "
                f"|U|_max={np.max(np.abs(u_grid)):.1f} m/s"
            )

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
        return False

    print(f"\nDone. Total wall time: {result.wall_time:.0f}s ({result.wall_time / 3600:.1f}h)")
    print(f"Averaged over {n_averaging_samples} daily samples (days {spinup_days + 1}-{n_days})")

    # --- Print summary ---
    if accum_zm is None or n_averaging_samples == 0:
        print("ERROR: No averaging samples collected")
        return False

    mean_zm = jax.tree.map(lambda x: x / n_averaging_samples, accum_zm)

    lat_deg = np.asarray(grid.latitudes_deg)
    sigma_full = np.asarray(levels.sigma_full)

    print("\n--- Climatology summary ---")
    surface_idx = np.argmax(sigma_full)
    equator_idx = np.argmin(np.abs(lat_deg))
    print(f"  T(equator, surface) = {mean_zm.temperature[surface_idx, equator_idx]:.1f} K")
    pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))
    print(f"  T(pole, surface)    = {mean_zm.temperature[surface_idx, pole_idx]:.1f} K")
    upper_mask = sigma_full < 0.4
    jet_max = float(np.max(np.abs(mean_zm.u[upper_mask, :])))
    print(f"  Jet max |U|         = {jet_max:.1f} m/s")
    eke = mean_zm.eke
    print(f"  EKE max             = {float(np.max(eke)):.1f} m^2/s^2")

    # --- Optional CSV output ---
    if output_path is not None:
        _save_output(output_path, mean_zm, grid, levels)

    return True


def _save_output(
    path: str,
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: object,
) -> None:
    """Save time-averaged zonal-mean fields to CSV."""
    import csv

    lat_deg = np.asarray(grid.latitudes_deg)
    sigma = np.asarray(levels.sigma_full)
    eke = mean_zm.eke

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "lat_deg", "U_ms", "T_K", "EKE_m2s2"])
        for k in range(len(sigma)):
            for j in range(len(lat_deg)):
                writer.writerow([
                    f"{sigma[k]:.6f}",
                    f"{lat_deg[j]:.2f}",
                    f"{mean_zm.u[k, j]:.4f}",
                    f"{mean_zm.temperature[k, j]:.4f}",
                    f"{eke[k, j]:.4f}",
                ])
    print(f"Zonal-mean climatology saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frierson aquaplanet")
    parser.add_argument("--days", type=int, default=300, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days before averaging")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=600.0, help="Timestep [s]")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    passed = run_aquaplanet(
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
