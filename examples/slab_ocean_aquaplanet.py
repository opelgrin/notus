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
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

from notus.constants import EARTH
from notus.diagnostics import ZonalMeanState, compute_zonal_mean_state
from notus.grid import GaussianGrid
from notus.initial_conditions import load_restart, moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.solar import EARTH_ORBIT
from notus.physics.surface import (
    OceanState,
    PrescribedSST,
    SlabOceanConfig,
    SurfaceState,
    compute_sst,
)
from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


def run_slab_ocean_aquaplanet(
    n_days: int = 300,
    spinup_days: int = 100,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    q_flux_amplitude: float = 30.0,
    q_flux_file: str | None = None,
    restart_file: str | None = None,
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

    # Physics: Byrne LW + SW + seasonal
    config = SimplePhysicsConfig(
        radiation_scheme="byrne",
        sw_tau_0=0.22,
        orbital=EARTH_ORBIT,
    )
    forcing = SimplePhysics(transform, EARTH, levels, config=config)

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

    steps_per_day = int(86400 / dt)

    # --- Build a scan function for one day ---
    def one_day(
        carry: tuple,
        day_of_year: jnp.ndarray,
    ) -> tuple[tuple, None]:
        prev, curr, sfc = carry
        # Set day_of_year for seasonal insolation
        forcing.day_of_year = day_of_year

        def step(
            carry: tuple,
            _: None,
        ) -> tuple[tuple, None]:
            p, c, s = carry
            p, c, s = step_fn(p, c, s)
            return (p, c, s), None

        (prev, curr, sfc), _ = jax.lax.scan(step, (prev, curr, sfc), None, length=steps_per_day)
        return (prev, curr, sfc), None

    one_day_jit = jax.jit(one_day)

    # --- Initialize ---
    print("Initializing...")
    t0 = time.perf_counter()
    prev, curr, surface = init_fn(state, surface)

    # Force compilation on first day
    (prev, curr, surface), _ = one_day_jit((prev, curr, surface), jnp.float64(0.0))
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
        ocean_state: OceanState,
        elapsed: float,
    ) -> bool:
        """Print diagnostics and return False if blowup detected."""
        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr_state.temperature))
        sst = np.asarray(ocean_state.surface_temperature)

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

        q_str = ""
        if curr_state.has_humidity:
            q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr_state.humidity))
            q_mean_gkg = float(np.mean(q_grid)) * 1000
            q_str = f"  q={q_mean_gkg:.2f}g/kg"

        print(
            f"  Day {day:5d} [{phase:>9s}]: "
            f"SST=[{np.min(sst):.1f}, {np.max(sst):.1f}] K  "
            f"T_atm=[{np.min(t_grid):.1f}, {np.max(t_grid):.1f}] K  "
            f"|U|={np.max(np.abs(u_grid)):.1f}"
            f"{q_str}"
            f"  [{days_per_sec:.1f} d/s]"
        )

        if not np.isfinite(t_mean) or not np.all(np.isfinite(sst)):
            print("ERROR: Integration has blown up!")
            return False
        return True

    # --- Main integration loop ---
    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        day_of_year = jnp.float64(day % days_per_year)
        # Sync forcing SST with ocean for consistent LW radiation
        forcing.sst = surface.ocean.surface_temperature
        (prev, curr, surface), _ = one_day_jit((prev, curr, surface), day_of_year)

        if day > spinup_days:
            _accumulate(curr)

        if day <= 10 or day % 50 == 0 or day == n_days:
            elapsed = time.perf_counter() - t_start
            if not _print_status(day, curr, surface.ocean, elapsed):
                return False

    total_time = time.perf_counter() - t0
    print(f"\nDone. Total wall time: {total_time:.0f}s ({total_time / 3600:.1f}h)")
    print(f"Averaged over {n_averaging_samples} daily samples")

    # --- Print summary ---
    sst_final = np.asarray(surface.ocean.surface_temperature)
    lat_deg = np.degrees(np.asarray(grid.latitudes))
    eq_idx = np.argmin(np.abs(lat_deg))
    pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))

    print("\n--- Final state ---")
    print(f"  SST(equator)  = {sst_final[eq_idx]:.1f} K")
    print(f"  SST(pole)     = {sst_final[pole_idx]:.1f} K")
    print(f"  SST range     = [{np.min(sst_final):.1f}, {np.max(sst_final):.1f}] K")

    if accum_zm is not None and n_averaging_samples > 0:
        mean_zm = ZonalMeanState(
            u=accum_zm.u / n_averaging_samples,
            v=accum_zm.v / n_averaging_samples,
            temperature=accum_zm.temperature / n_averaging_samples,
            u_prime_sq=accum_zm.u_prime_sq / n_averaging_samples,
            v_prime_sq=accum_zm.v_prime_sq / n_averaging_samples,
            uv_prime=accum_zm.uv_prime / n_averaging_samples,
            vt_prime=accum_zm.vt_prime / n_averaging_samples,
        )

        sigma_full = np.asarray(levels.sigma_full)
        surface_idx = np.argmax(sigma_full)
        upper_mask = sigma_full < 0.4

        print("\n--- Atmospheric climatology ---")
        print(f"  T(equator, sfc) = {mean_zm.temperature[surface_idx, eq_idx]:.1f} K")
        print(f"  T(pole, sfc)    = {mean_zm.temperature[surface_idx, pole_idx]:.1f} K")
        jet_max = float(np.max(np.abs(mean_zm.u[upper_mask, :])))
        print(f"  Jet max |U|     = {jet_max:.1f} m/s")
        eke = 0.5 * (mean_zm.u_prime_sq + mean_zm.v_prime_sq)
        print(f"  EKE max         = {float(np.max(eke)):.1f} m^2/s^2")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Slab ocean aquaplanet with seasonal cycle")
    parser.add_argument("--days", type=int, default=300, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=100, help="Spinup days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--q-flux", type=float, default=30.0, help="Q-flux amplitude [W/m^2]")
    parser.add_argument(
        "--q-flux-file", type=str, default=None, help="Q-flux .npz file from diagnose_qflux.py"
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
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
