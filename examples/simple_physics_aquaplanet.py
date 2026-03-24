#!/usr/bin/env python3
"""Frierson et al. (2006) simple physics aquaplanet integration.

Runs an aquaplanet with gray longwave radiation, dry convective adjustment,
bulk surface sensible heat flux, Rayleigh boundary-layer drag, and prescribed
SST.  No atmospheric shortwave absorption (Frierson convention).  Starts from
an isothermal rest state at T21 L20.

After a spinup period, accumulates time-averaged zonal-mean fields and prints
basic diagnostics.

Usage
-----
    uv run python examples/simple_physics_aquaplanet.py
    uv run python examples/simple_physics_aquaplanet.py --days 300 --spinup 100
    uv run python examples/simple_physics_aquaplanet.py --truncation 42 --dt 600
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
from notus.initial_conditions import simple_physics_initial_state
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.simple_physics import SimplePhysics
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import uniform_sigma_levels


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
    print(f"Simple physics aquaplanet: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # --- Setup ---
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = simple_physics_initial_state(
        transform,
        EARTH,
        levels,
        perturbation_amplitude=1.0,
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

    # --- Build a scan function for one day ---
    def one_day(
        carry: tuple[jnp.ndarray, ...],
        _: None,
    ) -> tuple[tuple[jnp.ndarray, ...], None]:
        prev, curr = carry

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

    # Force compilation on first day
    (prev, curr), _ = one_day_jit((prev, curr), None)
    t_compile = time.perf_counter() - t0
    print(f"Day 1 (incl. JIT compile): {t_compile:.1f}s")

    # --- Accumulator for time-averaged zonal means ---
    n_averaging_samples = 0
    accum_zm: ZonalMeanState | None = None

    def _accumulate(curr_state: jnp.ndarray) -> None:
        nonlocal n_averaging_samples, accum_zm
        zm = compute_zonal_mean_state(curr_state, transform)
        n_averaging_samples += 1
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

    def _print_status(
        day: int,
        curr_state: jnp.ndarray,
        elapsed: float,
    ) -> bool:
        """Print diagnostics and return False if blowup detected."""
        lnps_grid = np.asarray(transform.spectral_to_grid(curr_state.log_surface_pressure))
        ps_grid = EARTH.reference_pressure * np.exp(lnps_grid)
        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr_state.temperature))

        jet_level = max(0, n_levels // 4)
        u_spec, _ = uv_from_vordiv(
            curr_state.vorticity[jet_level],
            curr_state.divergence[jet_level],
            transform.arrays,
        )
        u_grid = np.asarray(transform.spectral_to_grid(u_spec))
        cos_lat = np.asarray(grid.cos_lat)
        u_grid = u_grid / cos_lat[:, None]

        t_mean = float(np.mean(t_grid))
        days_per_sec = (day - 1) / elapsed if elapsed > 0 else 0
        phase = "spinup" if day <= spinup_days else "averaging"

        print(
            f"  Day {day:5d} [{phase:>9s}]: "
            f"ps=[{np.min(ps_grid) / 100:.1f}, {np.max(ps_grid) / 100:.1f}] hPa  "
            f"T=[{np.min(t_grid):.1f}, {np.max(t_grid):.1f}] K (mean {t_mean:.1f})  "
            f"|U|_max={np.max(np.abs(u_grid)):.1f} m/s  "
            f"[{days_per_sec:.1f} days/s]"
        )

        if not np.isfinite(t_mean):
            print("ERROR: Integration has blown up!")
            return False
        return True

    # --- Main integration loop ---
    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        (prev, curr), _ = one_day_jit((prev, curr), None)

        # Accumulate zonal-mean statistics after spinup
        if day > spinup_days:
            _accumulate(curr)

        # Print diagnostics at intervals
        if day <= 10 or day % 50 == 0 or day == n_days:
            elapsed = time.perf_counter() - t_start
            if not _print_status(day, curr, elapsed):
                return False

    total_time = time.perf_counter() - t0
    print(f"\nDone. Total wall time: {total_time:.0f}s ({total_time / 3600:.1f}h)")
    print(f"Averaged over {n_averaging_samples} daily samples (days {spinup_days + 1}-{n_days})")

    # --- Print summary ---
    if accum_zm is None or n_averaging_samples == 0:
        print("ERROR: No averaging samples collected")
        return False

    mean_zm = ZonalMeanState(
        u=accum_zm.u / n_averaging_samples,
        v=accum_zm.v / n_averaging_samples,
        temperature=accum_zm.temperature / n_averaging_samples,
        u_prime_sq=accum_zm.u_prime_sq / n_averaging_samples,
        v_prime_sq=accum_zm.v_prime_sq / n_averaging_samples,
        uv_prime=accum_zm.uv_prime / n_averaging_samples,
        vt_prime=accum_zm.vt_prime / n_averaging_samples,
    )

    lat_deg = np.degrees(np.asarray(grid.latitudes))
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
    eke = 0.5 * (mean_zm.u_prime_sq + mean_zm.v_prime_sq)
    print(f"  EKE max             = {float(np.max(eke)):.1f} m^2/s^2")

    # --- Optional CSV output ---
    if output_path is not None:
        _save_output(output_path, mean_zm, grid, levels)

    return True


def _save_output(
    path: str,
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: jnp.ndarray,
) -> None:
    """Save time-averaged zonal-mean fields to CSV."""
    import csv

    lat_deg = np.degrees(np.asarray(grid.latitudes))
    sigma = np.asarray(levels.sigma_full)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "lat_deg", "U_ms", "T_K", "EKE_m2s2"])
        for k in range(len(sigma)):
            for j in range(len(lat_deg)):
                eke = 0.5 * (mean_zm.u_prime_sq[k, j] + mean_zm.v_prime_sq[k, j])
                writer.writerow([
                    f"{sigma[k]:.6f}",
                    f"{lat_deg[j]:.2f}",
                    f"{mean_zm.u[k, j]:.4f}",
                    f"{mean_zm.temperature[k, j]:.4f}",
                    f"{eke:.4f}",
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
