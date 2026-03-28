#!/usr/bin/env python3
"""Verify Monin-Obukhov surface layer on a slab ocean aquaplanet.

Runs two 100-day integrations side by side:
  1. Baseline (constant C_D = 0.0015)
  2. MO-enabled (Louis 1979 stability-dependent transfer coefficients)

Both should remain stable. Prints comparative diagnostics.

Usage
-----
    uv run python examples/verify_mo.py
    uv run python examples/verify_mo.py --days 300
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
from notus.physics.boundary_layer import SurfaceLayerConfig
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.solar import EARTH_ORBIT
from notus.physics.surface import OceanState, PrescribedSST, SlabOceanConfig, compute_sst
from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


def run_one(
    label: str,
    config: SimplePhysicsConfig,
    n_days: int,
    q_flux_data: jnp.ndarray | None = None,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
) -> dict:
    """Run a single integration and return diagnostics."""
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(n_levels)

    state, ref_temps, surface_phi = moist_aquaplanet_initial_state(
        transform, EARTH, levels, initial_rh=0.7, seed=42,
    )

    forcing = SimplePhysics(transform, EARTH, levels, config=config)

    ocean_config = SlabOceanConfig(mixed_layer_depth=50.0)
    if q_flux_data is not None:
        q_flux = q_flux_data
    else:
        q_flux_amplitude = 30.0
        q_flux = q_flux_amplitude * (1.0 - 2.0 * grid.sin_lat**2)

    sst_init = compute_sst(
        PrescribedSST(t_min=config.sst_t_min, t_delta=config.sst_t_delta, phi_w=config.sst_phi_w),
        grid.latitudes,
    )
    ocean = OceanState(surface_temperature=sst_init)

    filt = exponential_filter(transform.arrays, dt)
    init_fn, step_fn = build_coupled_pe_stepper(
        transform=transform, planet=EARTH, levels=levels,
        reference_temperature=ref_temps, surface_geopotential=surface_phi,
        dt=dt, forcing=forcing, ocean_config=ocean_config, q_flux=q_flux,
        spectral_filter=filt,
    )

    steps_per_day = int(86400 / dt)
    days_per_year = EARTH_ORBIT.days_per_year

    def scan_body(carry, _):
        p, c, o = carry
        p, c, o = step_fn(p, c, o)
        return (p, c, o), None

    # Initialize
    forcing.day_of_year = jnp.float64(0.0)
    prev, curr, ocean = init_fn(state, ocean)

    t_start = time.perf_counter()
    for day in range(1, n_days + 1):
        forcing.day_of_year = jnp.float64(day % days_per_year)
        (prev, curr, ocean), _ = jax.lax.scan(scan_body, (prev, curr, ocean), None, length=steps_per_day)

        if day % 25 == 0 or day == n_days:
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            sst = np.asarray(ocean.surface_temperature)
            t_mean = float(np.mean(t_grid))

            q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
            q_mean = float(np.mean(q_grid)) * 1000

            elapsed = time.perf_counter() - t_start
            rate = (day - 1) / elapsed if elapsed > 0 else 0

            print(
                f"  [{label}] Day {day:4d}: "
                f"SST=[{np.min(sst):.1f},{np.max(sst):.1f}]  "
                f"T_atm=[{np.min(t_grid):.1f},{np.max(t_grid):.1f}]  "
                f"q={q_mean:.2f}g/kg  "
                f"[{rate:.1f} d/s]"
            )

            if not np.isfinite(t_mean):
                print(f"  [{label}] BLOWUP at day {day}!")
                return {"label": label, "stable": False, "day_failed": day}

    elapsed = time.perf_counter() - t_start

    # Final diagnostics
    t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
    sst = np.asarray(ocean.surface_temperature)
    q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
    lat_deg = np.degrees(np.asarray(grid.latitudes))
    eq_idx = np.argmin(np.abs(lat_deg))

    jet_level = max(0, n_levels // 4)
    u_spec, _ = uv_from_vordiv(
        curr.vorticity[jet_level], curr.divergence[jet_level], transform.arrays,
    )
    u_grid = np.asarray(transform.spectral_to_grid(u_spec))
    u_grid = u_grid / np.asarray(grid.cos_lat)[:, None]

    lnps = np.asarray(transform.spectral_to_grid(curr.log_surface_pressure))
    ps = EARTH.reference_pressure * np.exp(lnps)

    return {
        "label": label,
        "stable": True,
        "elapsed": elapsed,
        "sst_eq": float(sst[eq_idx]),
        "sst_min": float(np.min(sst)),
        "sst_max": float(np.max(sst)),
        "t_mean": float(np.mean(t_grid)),
        "t_min": float(np.min(t_grid)),
        "t_max": float(np.max(t_grid)),
        "q_mean_gkg": float(np.mean(q_grid)) * 1000,
        "jet_max": float(np.max(np.abs(u_grid))),
        "ps_min": float(np.min(ps)),
        "ps_max": float(np.max(ps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MO surface layer")
    parser.add_argument("--days", type=int, default=100, help="Integration days")
    parser.add_argument(
        "--q-flux-file", type=str, default=None,
        help="Q-flux .npz file from diagnose_qflux.py (recommended for stability)",
    )
    args = parser.parse_args()

    n_days = args.days

    # Load diagnosed Q-flux if provided
    q_flux_data = None
    if args.q_flux_file is not None:
        data = np.load(args.q_flux_file)
        q_flux_data = jnp.array(data["q_flux"])
        print(f"Using diagnosed Q-flux from {args.q_flux_file}")
        print(f"  Q range: [{float(jnp.min(q_flux_data)):.1f}, {float(jnp.max(q_flux_data)):.1f}] W/m^2")
    else:
        print("Using analytic Q-flux: 30*(1-2sin^2(lat))")

    print(f"\n=== MO Verification: {n_days}-day slab ocean aquaplanet (T21 L20) ===\n")

    # Baseline: constant C_D
    print("--- Baseline (constant C_D=0.0015) ---")
    baseline_cfg = SimplePhysicsConfig(
        radiation_scheme="byrne", sw_tau_0=0.22, orbital=EARTH_ORBIT,
    )
    r_base = run_one("BASE", baseline_cfg, n_days, q_flux_data=q_flux_data)

    print()

    # MO-enabled
    print("--- Monin-Obukhov (Louis 1979, z0=1e-4) ---")
    mo_cfg = SimplePhysicsConfig(
        radiation_scheme="byrne", sw_tau_0=0.22, orbital=EARTH_ORBIT,
        surface_layer=SurfaceLayerConfig(z0_momentum=1e-4),
    )
    r_mo = run_one("MO", mo_cfg, n_days, q_flux_data=q_flux_data)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'':>20s}  {'Baseline':>12s}  {'MO':>12s}")
    print("-" * 60)
    for key in ["stable", "sst_eq", "sst_min", "sst_max", "t_mean", "t_min", "t_max",
                "q_mean_gkg", "jet_max", "ps_min", "ps_max"]:
        if key in r_base and key in r_mo:
            v1 = r_base[key]
            v2 = r_mo[key]
            if isinstance(v1, bool):
                print(f"  {key:>18s}  {str(v1):>12s}  {str(v2):>12s}")
            else:
                print(f"  {key:>18s}  {v1:>12.2f}  {v2:>12.2f}")
    print("=" * 60)

    if r_base["stable"] and r_mo["stable"]:
        print("\nBoth integrations stable. MO surface layer verified.")
        sys.exit(0)
    else:
        print("\nFAILURE: One or both integrations blew up!")
        sys.exit(1)


if __name__ == "__main__":
    main()
