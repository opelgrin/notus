#!/usr/bin/env python3
"""Verify Monin-Obukhov surface layer on a slab ocean aquaplanet.

Runs two coupled integrations side by side:
  1. Baseline (constant C_D = 0.0015)
  2. MO-enabled (Louis 1979 stability-dependent transfer coefficients)

By default, performs an in-memory prescribed-SST spinup to produce a
warm atmospheric state and self-consistent Q-flux before switching to
coupled mode. This avoids the violent cold-start transient.

Usage
-----
    # Default: automatic spinup + coupled run
    uv run python examples/verify_monin_obukhov.py

    # From pre-computed files (faster for repeated runs)
    uv run python examples/verify_monin_obukhov.py \
        --restart-file restart.npz --q-flux-file qflux.npz
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus import (
    EARTH,
    EARTH_ORBIT,
    ByrneRadiation,
    GaussianGrid,
    OceanState,
    PhysicsSuite,
    PhysicsSuiteConfig,
    PrescribedSST,
    SlabOceanConfig,
    SpectralTransform,
    SpeedyRadiation,
    SurfaceLayerConfig,
    SurfaceState,
    build_coupled_pe_stepper,
    compute_sst,
    exponential_filter,
    grid_surface_pressure,
    grid_winds_at_level,
    load_restart,
    moist_aquaplanet_initial_state,
    spinup_prescribed_sst,
    standard_sigma_levels,
)


def run_one(
    label: str,
    config: PhysicsSuiteConfig,
    n_days: int,
    warm_state: jnp.ndarray,
    q_flux: jnp.ndarray,
    ref_temps: np.ndarray,
    surface_phi: jnp.ndarray,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
) -> dict:
    """Run a single coupled integration and return diagnostics."""
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(n_levels)

    forcing = PhysicsSuite(transform, EARTH, levels, config=config)

    ocean_config = SlabOceanConfig(mixed_layer_depth=50.0)
    sst_init = compute_sst(
        PrescribedSST(t_min=config.sst_t_min, t_delta=config.sst_t_delta, phi_w=config.sst_phi_w),
        grid.latitudes,
    )
    ocean = OceanState(surface_temperature=sst_init)
    surface = SurfaceState(ocean=ocean)

    filt = exponential_filter(transform.arrays, dt)
    init_fn, step_fn = build_coupled_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        forcing=forcing,
        ocean_config=ocean_config,
        q_flux=q_flux,
        spectral_filter=filt,
    )

    steps_per_day = int(86400 / dt)
    days_per_year = EARTH_ORBIT.days_per_year

    def scan_body(carry, _):
        p, c, sfc = carry
        p, c, sfc = step_fn(p, c, sfc)
        return (p, c, sfc), None

    # Initialize from warm state
    forcing.day_of_year = jnp.float64(0.0)
    forcing.prescribed_sst = surface.ocean.surface_temperature
    prev, curr, surface = init_fn(warm_state, surface)

    t_start = time.perf_counter()
    for day in range(1, n_days + 1):
        forcing.day_of_year = jnp.float64(day % days_per_year)
        forcing.prescribed_sst = surface.ocean.surface_temperature
        (prev, curr, surface), _ = jax.lax.scan(
            scan_body,
            (prev, curr, surface),
            None,
            length=steps_per_day,
        )

        if day % 25 == 0 or day == n_days:
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            sst = np.asarray(surface.ocean.surface_temperature)
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
    sst = np.asarray(surface.ocean.surface_temperature)
    q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.humidity))
    lat_deg = np.asarray(grid.latitudes_deg)
    eq_idx = np.argmin(np.abs(lat_deg))

    jet_level = max(0, n_levels // 4)
    u_grid, _v = grid_winds_at_level(curr, jet_level, transform)
    u_grid = np.asarray(u_grid)

    ps = np.asarray(grid_surface_pressure(curr, transform, EARTH))

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
    parser.add_argument("--days", type=int, default=100, help="Coupled integration days")
    parser.add_argument("--spinup-days", type=int, default=100, help="Prescribed-SST spinup days")
    parser.add_argument("--averaging-days", type=int, default=200, help="Q-flux averaging days")
    parser.add_argument("--scheme", type=str, default="byrne", choices=["byrne", "speedy"])
    parser.add_argument("--clouds", action="store_true", help="Enable diagnostic clouds (speedy)")
    parser.add_argument(
        "--q-flux-file",
        type=str,
        default=None,
        help="Pre-computed Q-flux .npz (skip in-memory diagnosis)",
    )
    parser.add_argument(
        "--restart-file",
        type=str,
        default=None,
        help="Pre-computed restart .npz (skip in-memory spinup)",
    )
    args = parser.parse_args()

    n_days = args.days

    # --- Setup ---
    grid = GaussianGrid(truncation=21)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(20)

    state, ref_temps, surface_phi = moist_aquaplanet_initial_state(
        transform,
        EARTH,
        levels,
        initial_rh=0.7,
        seed=42,
    )

    # --- Spinup + Q-flux: in-memory or from disk ---
    if args.restart_file is not None and args.q_flux_file is not None:
        # Both provided: load from disk
        warm_state, _ = load_restart(args.restart_file)
        data = np.load(args.q_flux_file)
        q_flux = jnp.array(data["q_flux"])
        print(f"Loaded restart from {args.restart_file}")
        print(f"Loaded Q-flux from {args.q_flux_file}")
        print(f"  Q range: [{float(jnp.min(q_flux)):.1f}, {float(jnp.max(q_flux)):.1f}] W/m^2")
    else:
        # In-memory spinup + Q-flux diagnosis
        sd, ad = args.spinup_days, args.averaging_days
        print(f"Running prescribed-SST spinup ({sd}d spinup + {ad}d averaging)...")
        if args.scheme == "speedy":
            from notus.physics.clouds import CloudConfig

            base_config = PhysicsSuiteConfig(
                radiation=SpeedyRadiation(clouds=CloudConfig() if args.clouds else None),
                orbital=EARTH_ORBIT,
            )
        else:
            base_config = PhysicsSuiteConfig(
                radiation=ByrneRadiation(sw_tau_0=0.22),
                orbital=EARTH_ORBIT,
            )
        spinup_forcing = PhysicsSuite(transform, EARTH, levels, config=base_config)
        result = spinup_prescribed_sst(
            state,
            spinup_forcing,
            transform,
            EARTH,
            levels,
            ref_temps,
            surface_phi,
            dt=900.0,
            spinup_days=args.spinup_days,
            averaging_days=args.averaging_days,
        )
        warm_state = result.state
        q_flux = result.q_flux
        print()

    print(f"=== MO Verification: {n_days}-day slab ocean aquaplanet (T21 L20) ===\n")

    # Baseline: constant C_D
    print("--- Baseline (constant C_D=0.0015) ---")
    if args.scheme == "speedy":
        from notus.physics.clouds import CloudConfig

        baseline_cfg = PhysicsSuiteConfig(
            radiation=SpeedyRadiation(clouds=CloudConfig() if args.clouds else None),
            orbital=EARTH_ORBIT,
        )
    else:
        baseline_cfg = PhysicsSuiteConfig(
            radiation=ByrneRadiation(sw_tau_0=0.22),
            orbital=EARTH_ORBIT,
        )
    r_base = run_one("BASE", baseline_cfg, n_days, warm_state, q_flux, ref_temps, surface_phi)

    print()

    # MO-enabled
    print("--- Monin-Obukhov (Louis 1979, z0=1e-4) ---")
    if args.scheme == "speedy":
        mo_cfg = PhysicsSuiteConfig(
            radiation=SpeedyRadiation(clouds=CloudConfig() if args.clouds else None),
            orbital=EARTH_ORBIT,
            surface_layer=SurfaceLayerConfig(z0_momentum=1e-4),
        )
    else:
        mo_cfg = PhysicsSuiteConfig(
            radiation=ByrneRadiation(sw_tau_0=0.22),
            orbital=EARTH_ORBIT,
            surface_layer=SurfaceLayerConfig(z0_momentum=1e-4),
        )
    r_mo = run_one("MO", mo_cfg, n_days, warm_state, q_flux, ref_temps, surface_phi)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'':>20s}  {'Baseline':>12s}  {'MO':>12s}")
    print("-" * 60)
    for key in [
        "stable",
        "sst_eq",
        "sst_min",
        "sst_max",
        "t_mean",
        "t_min",
        "t_max",
        "q_mean_gkg",
        "jet_max",
        "ps_min",
        "ps_max",
    ]:
        if key in r_base and key in r_mo:
            v1 = r_base[key]
            v2 = r_mo[key]
            if isinstance(v1, bool):
                print(f"  {key:>18s}  {v1!s:>12s}  {v2!s:>12s}")
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
