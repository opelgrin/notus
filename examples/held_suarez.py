#!/usr/bin/env python3
"""Held-Suarez (1994) benchmark integration with climatological validation.

Runs the standard dry dynamical core intercomparison: Newtonian relaxation
toward an equilibrium temperature + Rayleigh friction in the boundary layer,
starting from an isothermal rest state at T42 L20.

After a spinup period, accumulates time-averaged zonal-mean fields and
validates them against the known Held-Suarez climatology:
  - Subtropical jet strength and position
  - Zonal-mean temperature structure
  - Surface westerlies
  - Eddy kinetic energy

Usage
-----
    uv run python examples/held_suarez.py [--days 1200] [--spinup 200]
    uv run python examples/held_suarez.py --days 400 --spinup 200  # shorter run
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus import (
    EARTH,
    GaussianGrid,
    HeldSuarez,
    PrimitiveEquationState,
    SigmaLevels,
    SpectralTransform,
    ZonalMeanState,
    build_pe_stepper,
    compute_zonal_mean_state,
    exponential_filter,
    grid_surface_pressure,
    grid_winds_at_level,
    held_suarez_initial_state,
    run_simulation,
    uniform_sigma_levels,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Benchmark validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    value: float
    expected: str
    message: str


def validate_climatology(
    mean_state: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
) -> list[ValidationResult]:
    """Validate time-averaged zonal-mean fields against Held-Suarez targets.

    Targets based on Held & Suarez (1994, Fig. 1-4) and the multi-model
    intercomparison results.  Ranges are deliberately wide to accommodate
    differences in resolution, diffusion, and vertical levels.

    Parameters
    ----------
    mean_state : ZonalMeanState
        Time-averaged zonal-mean fields.
    grid : GaussianGrid
        Gaussian grid (for latitude values).
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    list[ValidationResult]
        Results for each validation check.
    """
    results: list[ValidationResult] = []
    lat_deg = np.asarray(grid.latitudes_deg)  # north-to-south
    sigma_full = np.asarray(levels.sigma_full)

    u_zm = mean_state.u  # (n_levels, n_lat)
    t_zm = mean_state.temperature
    eke = mean_state.eke

    # ---- 1. Subtropical jet: peak U at upper levels (σ < 0.4) ----
    # H&S Fig. 1: jets at ~30°N/S, 25-35 m/s
    upper_mask = sigma_full < 0.4
    u_upper = u_zm[upper_mask, :]  # only upper-troposphere levels
    jet_max = float(np.max(np.abs(u_upper)))
    results.append(
        ValidationResult(
            name="Subtropical jet strength",
            passed=20.0 <= jet_max <= 50.0,
            value=jet_max,
            expected="20-50 m/s",
            message=f"Peak upper-level |U| = {jet_max:.1f} m/s",
        )
    )

    # Jet latitude: find the latitude of maximum U in upper levels
    # Average U over upper levels, find peak in NH (lat_deg > 0)
    u_upper_mean = np.mean(u_upper, axis=0)  # (n_lat,)
    nh_mask = lat_deg > 0
    nh_lats = lat_deg[nh_mask]
    nh_u = u_upper_mean[nh_mask]
    jet_lat = float(nh_lats[np.argmax(nh_u)])
    results.append(
        ValidationResult(
            name="Subtropical jet latitude (NH)",
            passed=15.0 <= jet_lat <= 50.0,
            value=jet_lat,
            expected="15-50°N",
            message=f"NH jet peak at {jet_lat:.1f}°N",
        )
    )

    # ---- 2. Surface westerlies ----
    # H&S Fig. 1: midlatitude surface westerlies ~5-15 m/s
    surface_idx = np.argmax(sigma_full)  # closest to σ=1
    u_surface = u_zm[surface_idx, :]  # (n_lat,)
    midlat_mask = (np.abs(lat_deg) > 30.0) & (np.abs(lat_deg) < 60.0)
    u_surface_midlat = float(np.max(u_surface[midlat_mask]))
    results.append(
        ValidationResult(
            name="Midlatitude surface westerlies",
            passed=2.0 <= u_surface_midlat <= 20.0,
            value=u_surface_midlat,
            expected="2-20 m/s",
            message=f"Peak midlatitude surface U = {u_surface_midlat:.1f} m/s",
        )
    )

    # ---- 3. Temperature structure ----
    # H&S Fig. 2: equatorial surface ~295-305 K, polar surface ~250-265 K
    equator_idx = np.argmin(np.abs(lat_deg))
    pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))

    t_eq_surface = float(t_zm[surface_idx, equator_idx])
    results.append(
        ValidationResult(
            name="Equatorial surface temperature",
            passed=290.0 <= t_eq_surface <= 315.0,
            value=t_eq_surface,
            expected="290-315 K",
            message=f"T_eq(surface) = {t_eq_surface:.1f} K",
        )
    )

    t_pole_surface = float(t_zm[surface_idx, pole_idx])
    results.append(
        ValidationResult(
            name="Polar surface temperature",
            passed=245.0 <= t_pole_surface <= 275.0,
            value=t_pole_surface,
            expected="245-275 K",
            message=f"T_pole(surface) = {t_pole_surface:.1f} K",
        )
    )

    # Equator-to-pole gradient at surface
    dt_surface = t_eq_surface - t_pole_surface
    results.append(
        ValidationResult(
            name="Surface equator-to-pole ΔT",
            passed=25.0 <= dt_surface <= 55.0,
            value=dt_surface,
            expected="25-55 K",
            message=f"ΔT(eq-pole) = {dt_surface:.1f} K",
        )
    )

    # Tropopause temperature (upper levels, ~σ=0.1-0.3)
    tropo_mask = (sigma_full > 0.05) & (sigma_full < 0.3)
    if np.any(tropo_mask):
        t_tropo_min = float(np.min(t_zm[tropo_mask, :]))
        results.append(
            ValidationResult(
                name="Minimum tropopause temperature",
                passed=180.0 <= t_tropo_min <= 230.0,
                value=t_tropo_min,
                expected="180-230 K",
                message=f"T_min(tropopause) = {t_tropo_min:.1f} K",
            )
        )

    # ---- 4. Eddy kinetic energy ----
    # H&S Fig. 3-4: EKE peaks in midlatitudes at upper levels
    eke_max = float(np.max(eke))
    results.append(
        ValidationResult(
            name="Eddy kinetic energy (peak)",
            passed=10.0 <= eke_max <= 500.0,
            value=eke_max,
            expected="10-500 m²/s²",
            message=f"EKE_max = {eke_max:.1f} m²/s²",
        )
    )

    # EKE peak should be in midlatitudes (20°-70°)
    eke_col_mean = np.mean(eke[upper_mask, :], axis=0)  # upper-level EKE vs lat
    eke_peak_lat = float(lat_deg[np.argmax(np.abs(eke_col_mean))])
    results.append(
        ValidationResult(
            name="EKE peak latitude",
            passed=20.0 <= np.abs(eke_peak_lat) <= 70.0,
            value=eke_peak_lat,
            expected="20-70° (either hemisphere)",
            message=f"EKE peak at {eke_peak_lat:.1f}°",
        )
    )

    # ---- 5. Hemispheric symmetry (approximate) ----
    # Time-mean should be roughly N-S symmetric. Check that both
    # hemispheres have jets of similar strength (within factor of ~3)
    sh_mask = lat_deg < 0
    sh_u = u_upper_mean[sh_mask]
    sh_jet = float(np.max(sh_u))
    nh_jet = float(np.max(nh_u))
    ratio = min(sh_jet, nh_jet) / max(sh_jet, nh_jet) if max(sh_jet, nh_jet) > 1 else 1.0
    results.append(
        ValidationResult(
            name="Hemispheric jet symmetry",
            passed=0.3 <= ratio <= 1.0,
            value=ratio,
            expected="0.3-1.0 (SH/NH or NH/SH)",
            message=f"Jet ratio = {ratio:.2f} (NH={nh_jet:.1f}, SH={sh_jet:.1f} m/s)",
        )
    )

    return results


def print_validation_report(results: list[ValidationResult]) -> bool:
    """Print validation results and return True if all checks passed."""
    print("\n" + "=" * 70)
    print("HELD-SUAREZ BENCHMARK VALIDATION")
    print("=" * 70)

    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}  (expected: {r.expected})")

    print("-" * 70)
    all_passed = n_passed == n_total
    if all_passed:
        print(f"  Result: ALL {n_total} CHECKS PASSED")
    else:
        print(f"  Result: {n_passed}/{n_total} checks passed, {n_total - n_passed} FAILED")
    print("=" * 70)

    return all_passed


# ---------------------------------------------------------------------------
# Main integration
# ---------------------------------------------------------------------------


def run_held_suarez(
    n_days: int = 1200,
    spinup_days: int = 200,
    truncation: int = 42,
    n_levels: int = 20,
    dt: float = 600.0,
    output_path: str | None = None,
) -> bool:
    """Run a Held-Suarez integration with validation.

    Returns True if all benchmark validation checks pass.
    """
    print(f"Held-Suarez benchmark: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # --- Setup ---
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = held_suarez_initial_state(
        transform,
        EARTH,
        levels,
        perturbation_amplitude=1.0,
        seed=42,
    )

    forcing = HeldSuarez(transform, EARTH, levels)
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
        if not np.all(np.isfinite(t_grid)):
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
                f"T=[{np.min(t_grid):.1f}, {np.max(t_grid):.1f}] K "
                f"(mean {float(np.mean(t_grid)):.1f})  "
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

    # --- Compute time-averaged climatology ---
    if accum_zm is None or n_averaging_samples == 0:
        print("ERROR: No averaging samples collected")
        return False

    mean_zm = jax.tree.map(lambda x: x / n_averaging_samples, accum_zm)

    # --- Validate against benchmark ---
    results = validate_climatology(mean_zm, grid, levels)
    all_passed = print_validation_report(results)

    # --- Optional CSV output ---
    if output_path is not None:
        _save_output(output_path, mean_zm, grid, levels)

    return all_passed


def _save_output(
    path: str,
    mean_zm: ZonalMeanState,
    grid: GaussianGrid,
    levels: SigmaLevels,
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
    parser = argparse.ArgumentParser(description="Held-Suarez benchmark")
    parser.add_argument("--days", type=int, default=1200, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=200, help="Spinup days before averaging")
    parser.add_argument("--truncation", type=int, default=42, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=600.0, help="Timestep [s]")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    passed = run_held_suarez(
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
