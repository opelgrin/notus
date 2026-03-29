#!/usr/bin/env python3
"""Slab ocean aquaplanet with seasonal insolation.

Runs a moist aquaplanet with Byrne two-band radiation, seasonal solar
forcing (orbital parameters, daily-mean insolation), and an interactive
slab ocean replacing the prescribed SST.

The slab ocean evolves via surface energy balance:
    dT_s/dt = (SW_abs + LW_down - sigma*T_s^4 - H - LE + Q_flux) / C_ocean

**Warm start recommended**: Cold-starting from an isothermal atmosphere
creates violent transients that can destabilize the integration,
especially with Monin-Obukhov surface layer. Use ``diagnose_qflux.py
--save-restart`` to produce a spun-up atmospheric state, then pass it
via ``--restart-file``.

Usage
-----
    # Step 1: Diagnose Q-flux and save restart
    uv run python examples/diagnose_qflux.py --days 300 --save-restart restart.npz

    # Step 2: Run slab ocean from warm start
    uv run python examples/slab_ocean_aquaplanet.py \
        --restart-file restart.npz --q-flux-file qflux.npz
"""

from __future__ import annotations

import argparse
import logging
import sys

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
    SurfaceState,
    ZonalMeanState,
    build_coupled_pe_stepper,
    compute_sst,
    compute_zonal_mean_state,
    exponential_filter,
    grid_winds_at_level,
    load_restart,
    moist_aquaplanet_initial_state,
    run_simulation,
    standard_sigma_levels,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_slab_ocean_aquaplanet(
    n_days: int = 300,
    spinup_days: int = 100,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    q_flux_amplitude: float = 30.0,
    q_flux_file: str | None = None,
    restart_file: str | None = None,
    scheme: str = "byrne",
    clouds: bool = False,
) -> bool:
    """Run a slab ocean aquaplanet with seasonal insolation.

    Returns True if the integration completes without blowup.
    """
    days_per_year = EARTH_ORBIT.days_per_year
    print(f"Slab ocean aquaplanet: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")
    print(f"  Seasonal cycle: obliquity={np.degrees(EARTH_ORBIT.obliquity):.1f} deg")
    if q_flux_file is not None:
        print(f"  Q-flux: from {q_flux_file}")
    else:
        print(f"  Slab ocean: 50m mixed layer, Q-flux amplitude={q_flux_amplitude:.0f} W/m^2")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # --- Setup ---
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(n_levels)

    # Warm start from restart, or cold start from isothermal IC
    if restart_file is not None:
        state, _ = load_restart(restart_file)
        print(f"  Warm start from {restart_file}")
    else:
        state, _, _ = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
            initial_rh=0.7,
            seed=42,
        )

    # ref_temps and surface_phi always needed for semi-implicit solver
    _, ref_temps, surface_phi = moist_aquaplanet_initial_state(
        transform,
        EARTH,
        levels,
        initial_rh=0.7,
        seed=42,
    )

    if scheme == "speedy":
        from notus.physics.clouds import CloudConfig

        config = PhysicsSuiteConfig(
            radiation=SpeedyRadiation(clouds=CloudConfig() if clouds else None),
            orbital=EARTH_ORBIT,
        )
    else:
        config = PhysicsSuiteConfig(radiation=ByrneRadiation(sw_tau_0=0.22), orbital=EARTH_ORBIT)
    forcing = PhysicsSuite(transform, EARTH, levels, config=config)

    # Slab ocean
    ocean_config = SlabOceanConfig(mixed_layer_depth=50.0)
    sin_lat = grid.sin_lat

    # Q-flux: from diagnosed file or analytic fallback
    if q_flux_file is not None:
        data = np.load(q_flux_file)
        q_flux = jnp.array(data["q_flux"])
        print(
            f"  Loaded Q-flux: [{float(jnp.min(q_flux)):.1f}, {float(jnp.max(q_flux)):.1f}] W/m^2"
        )
    else:
        # Simple analytic: poleward heat transport
        q_flux = q_flux_amplitude * (1.0 - 2.0 * sin_lat**2)

    # Initialize SST from the Frierson profile
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

    # --- Accumulator for time-averaged zonal means ---
    n_averaging_samples = 0
    accum_zm: ZonalMeanState | None = None
    blew_up = False

    def on_day(day: int, curr_state: object, sfc: object) -> None:
        nonlocal n_averaging_samples, accum_zm, blew_up

        if day > spinup_days:
            zm = compute_zonal_mean_state(curr_state, transform)  # type: ignore[arg-type]
            n_averaging_samples += 1
            accum_zm = zm if accum_zm is None else jax.tree.map(np.add, accum_zm, zm)

        if day <= 10 or day % 50 == 0 or day == n_days:
            t_grid = np.asarray(
                jax.vmap(transform.spectral_to_grid)(curr_state.temperature)  # type: ignore[union-attr]
            )
            sst = np.asarray(sfc.ocean.surface_temperature)  # type: ignore[union-attr]

            jet_level = max(0, n_levels // 4)
            u_grid, _v = grid_winds_at_level(curr_state, jet_level, transform)  # type: ignore[arg-type]
            u_grid = np.asarray(u_grid)

            t_mean = float(np.mean(t_grid))
            phase = "spinup" if day <= spinup_days else "averaging"

            q_str = ""
            if curr_state.has_humidity:  # type: ignore[union-attr]
                q_grid = np.asarray(
                    jax.vmap(transform.spectral_to_grid)(curr_state.humidity)  # type: ignore[union-attr]
                )
                q_mean_gkg = float(np.mean(q_grid)) * 1000
                q_str = f"  q={q_mean_gkg:.2f}g/kg"

            print(
                f"  Day {day:5d} [{phase:>9s}]: "
                f"SST=[{np.min(sst):.1f}, {np.max(sst):.1f}] K  "
                f"T_atm=[{np.min(t_grid):.1f}, {np.max(t_grid):.1f}] K  "
                f"|U|={np.max(np.abs(u_grid)):.1f}"
                f"{q_str}"
            )

            if not np.isfinite(t_mean) or not np.all(np.isfinite(sst)):
                print("ERROR: Integration has blown up!")
                blew_up = True

    # --- Run ---
    result = run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=state,
        dt=dt,
        n_days=n_days,
        surface=surface,
        forcing=forcing,
        days_per_year=days_per_year,
        on_day=on_day,
        verbose=False,
    )

    if blew_up:
        return False

    print(f"\nDone. Total wall time: {result.wall_time:.0f}s ({result.wall_time / 3600:.1f}h)")
    print(f"Averaged over {n_averaging_samples} daily samples")

    # --- Print summary ---
    final_surface = result.surface
    if final_surface is None:
        return False
    sst_final = np.asarray(final_surface.ocean.surface_temperature)
    lat_deg = np.asarray(grid.latitudes_deg)
    eq_idx = np.argmin(np.abs(lat_deg))
    pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))

    print("\n--- Final state ---")
    print(f"  SST(equator)  = {sst_final[eq_idx]:.1f} K")
    print(f"  SST(pole)     = {sst_final[pole_idx]:.1f} K")
    print(f"  SST range     = [{np.min(sst_final):.1f}, {np.max(sst_final):.1f}] K")

    if accum_zm is not None and n_averaging_samples > 0:
        mean_zm = jax.tree.map(lambda x: x / n_averaging_samples, accum_zm)

        sigma_full = np.asarray(levels.sigma_full)
        surface_idx = np.argmax(sigma_full)
        upper_mask = sigma_full < 0.4

        print("\n--- Atmospheric climatology ---")
        print(f"  T(equator, sfc) = {mean_zm.temperature[surface_idx, eq_idx]:.1f} K")
        print(f"  T(pole, sfc)    = {mean_zm.temperature[surface_idx, pole_idx]:.1f} K")
        jet_max = float(np.max(np.abs(mean_zm.u[upper_mask, :])))
        print(f"  Jet max |U|     = {jet_max:.1f} m/s")
        eke = mean_zm.eke
        print(f"  EKE max         = {float(np.max(eke)):.1f} m^2/s^2")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Slab ocean aquaplanet with seasonal cycle")
    parser.add_argument("--days", type=int, default=300, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--scheme", type=str, default="byrne", choices=["byrne", "speedy"])
    parser.add_argument("--clouds", action="store_true", help="Enable diagnostic clouds (speedy)")
    parser.add_argument("--q-flux", type=float, default=30.0, help="Q-flux amplitude [W/m^2]")
    parser.add_argument(
        "--q-flux-file",
        type=str,
        default=None,
        help="Q-flux .npz file from diagnose_qflux.py",
    )
    parser.add_argument(
        "--restart-file",
        type=str,
        default=None,
        help="Warm-start from restart .npz (from diagnose_qflux.py --save-restart)",
    )
    args = parser.parse_args()

    passed = run_slab_ocean_aquaplanet(
        n_days=args.days,
        spinup_days=args.spinup,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        q_flux_amplitude=args.q_flux,
        q_flux_file=args.q_flux_file,
        restart_file=args.restart_file,
        scheme=args.scheme,
        clouds=args.clouds,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
