#!/usr/bin/env python3
"""Moist aquaplanet with an isolated mountain.

Demonstrates a full-physics Notus simulation over topography: moist
dynamics with SPEEDY multi-band radiation, Betts-Miller convection,
large-scale condensation, surface fluxes, and a 2500 m Gaussian mountain.

The mountain forces spatially varying precipitation, flow deflection,
and surface pressure patterns — all emerging from the resolved dynamics
without any explicit orographic parameterization.

Usage
-----
    uv run python examples/mountain_aquaplanet.py
    uv run python examples/mountain_aquaplanet.py --days 100 --spinup 30
    uv run python examples/mountain_aquaplanet.py --truncation 42 --dt 600
    uv run python examples/mountain_aquaplanet.py --height 4000 --lat 45
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
    ByrneRadiation,
    FriersonRadiation,
    GaussianGrid,
    PhysicsSuite,
    PhysicsSuiteConfig,
    PrimitiveEquationState,
    SpectralTransform,
    SpeedyRadiation,
    build_pe_stepper,
    compute_zonal_mean_state,
    exponential_filter,
    gaussian_mountain,
    grid_surface_pressure,
    grid_winds_at_level,
    moist_aquaplanet_initial_state,
    orographic_log_surface_pressure,
    run_simulation,
    smooth_orography,
    standard_sigma_levels,
)
from notus.physics.surface import diagnose_precipitation


logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_mountain_aquaplanet(
    n_days: int = 100,
    spinup_days: int = 30,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    mountain_height: float = 2500.0,
    mountain_lat: float = 30.0,
    mountain_lon: float = 180.0,
    mountain_width: float = 20.0,
    scheme: str = "speedy",
) -> bool:
    """Run a moist aquaplanet with a mountain.

    Parameters
    ----------
    n_days : int
        Total simulation days.
    spinup_days : int
        Days before diagnostics begin.
    truncation : int
        Spectral truncation (T21, T42, ...).
    n_levels : int
        Number of vertical levels.
    dt : float
        Timestep [s].
    mountain_height : float
        Mountain peak height [m].
    mountain_lat : float
        Mountain centre latitude [degrees].
    mountain_lon : float
        Mountain centre longitude [degrees].
    mountain_width : float
        Mountain half-width [degrees].
    scheme : str
        Radiation scheme: "frierson", "byrne", or "speedy".

    Returns
    -------
    bool
        True if the integration completes successfully.
    """
    print(f"Mountain aquaplanet: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Mountain: {mountain_height:.0f} m at ({mountain_lat:.0f}N, {mountain_lon:.0f}E)")
    print(f"  Spinup: {spinup_days} days, averaging: {n_days - spinup_days} days")
    print(f"  Radiation: {scheme}")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

    # ----------------------------------------------------------------
    # 1. Grid, transform, vertical levels
    # ----------------------------------------------------------------
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(n_levels)

    # ----------------------------------------------------------------
    # 2. Topography: Gaussian mountain with spectral smoothing
    # ----------------------------------------------------------------
    center_lat = np.radians(mountain_lat)
    center_lon = np.radians(mountain_lon)
    half_width = np.radians(mountain_width)

    surface_phi = gaussian_mountain(
        transform,
        EARTH,
        height=mountain_height,
        center_lat=center_lat,
        center_lon=center_lon,
        half_width=half_width,
    )
    # Lanczos smoothing suppresses Gibbs ringing from spectral truncation
    surface_phi = smooth_orography(surface_phi, transform.arrays, method="lanczos")

    # Verify the smoothed peak height
    z_grid = np.asarray(transform.spectral_to_grid(surface_phi)) / EARTH.gravity
    print(f"  Smoothed peak height: {z_grid.max():.0f} m")

    # ----------------------------------------------------------------
    # 3. Initial conditions: moist atmosphere + orographic pressure
    # ----------------------------------------------------------------
    ref_temp = 264.0
    state, ref_temps, _ = moist_aquaplanet_initial_state(
        transform, EARTH, levels, initial_temperature=ref_temp, initial_rh=0.7, seed=42
    )
    # Adjust surface pressure for hydrostatic balance over the mountain
    ln_ps = orographic_log_surface_pressure(surface_phi, transform, EARTH, ref_temp)
    state = state.replace(log_surface_pressure=ln_ps)

    # ----------------------------------------------------------------
    # 4. Physics and time stepper
    # ----------------------------------------------------------------
    radiation_configs = {
        "frierson": FriersonRadiation(),
        "byrne": ByrneRadiation(sw_tau_0=0.22),
        "speedy": SpeedyRadiation(),
    }
    config = PhysicsSuiteConfig(radiation=radiation_configs[scheme])
    forcing = PhysicsSuite(transform, EARTH, levels, config=config)
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

    # ----------------------------------------------------------------
    # 5. Diagnostic callback
    # ----------------------------------------------------------------
    n_averaging_samples = 0
    accum_u: np.ndarray | None = None
    accum_t: np.ndarray | None = None
    accum_eke: np.ndarray | None = None
    accum_precip: np.ndarray | None = None
    blew_up = False

    def on_day(day: int, curr_state: PrimitiveEquationState) -> None:
        nonlocal n_averaging_samples, accum_u, accum_t, accum_eke, accum_precip, blew_up

        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr_state.temperature))
        t_mean = float(np.mean(t_grid))
        if not np.isfinite(t_mean):
            print("ERROR: Integration has blown up!")
            blew_up = True
            return

        # Accumulate zonal-mean diagnostics after spinup
        if day > spinup_days:
            zm = compute_zonal_mean_state(curr_state, transform)
            n_averaging_samples += 1
            u = np.asarray(zm.u)
            t = np.asarray(zm.temperature)
            eke = np.asarray(zm.eke)
            accum_u = u if accum_u is None else accum_u + u
            accum_t = t if accum_t is None else accum_t + t
            accum_eke = eke if accum_eke is None else accum_eke + eke

            # Diagnose precipitation
            q_grid = jax.vmap(transform.spectral_to_grid)(curr_state.humidity)
            ps_grid = EARTH.reference_pressure * jnp.exp(
                transform.spectral_to_grid(curr_state.log_surface_pressure)
            )
            sigma_full = jnp.array(levels.sigma_full)
            pressure = sigma_full[:, None, None] * ps_grid[None, :, :]
            precip = np.asarray(
                diagnose_precipitation(
                    t_grid,
                    q_grid,
                    pressure,
                    levels.dsigma,
                    ps_grid,
                    gravity=EARTH.gravity,
                    epsilon=EARTH.epsilon_moisture,
                    latent_heat=EARTH.latent_heat_vaporization,
                    specific_heat_cp=EARTH.specific_heat_cp,
                    gas_constant=EARTH.gas_constant,
                )
            )
            accum_precip = precip if accum_precip is None else accum_precip + precip

        # Status output
        if day <= 5 or day % 20 == 0 or day == n_days:
            ps_grid = np.asarray(grid_surface_pressure(curr_state, transform, EARTH))
            jet_level = max(0, n_levels // 4)
            u_grid, _ = grid_winds_at_level(curr_state, jet_level, transform)
            u_grid = np.asarray(u_grid)
            q_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr_state.humidity))
            q_mean = float(np.mean(q_grid)) * 1000
            phase = "spinup" if day <= spinup_days else "averaging"

            print(
                f"  Day {day:4d} [{phase:>9s}]: "
                f"ps=[{np.min(ps_grid) / 100:.1f}, {np.max(ps_grid) / 100:.1f}] hPa  "
                f"T=[{np.min(t_grid):.1f}, {np.max(t_grid):.1f}] K  "
                f"|U|={np.max(np.abs(u_grid)):.1f} m/s  "
                f"q={q_mean:.2f} g/kg"
            )

    # ----------------------------------------------------------------
    # 6. Run
    # ----------------------------------------------------------------
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

    print(f"\nDone. Wall time: {result.wall_time:.0f}s ({result.wall_time / 3600:.1f}h)")
    print(f"Averaged over {n_averaging_samples} daily samples (days {spinup_days + 1}-{n_days})")

    # ----------------------------------------------------------------
    # 7. Print climatology summary
    # ----------------------------------------------------------------
    if accum_u is None or n_averaging_samples == 0:
        print("ERROR: No averaging samples collected")
        return False

    mean_u = accum_u / n_averaging_samples
    mean_t = accum_t / n_averaging_samples  # type: ignore[operator]
    mean_eke = accum_eke / n_averaging_samples  # type: ignore[operator]
    mean_precip = accum_precip / n_averaging_samples  # type: ignore[operator]

    lat_deg = np.asarray(grid.latitudes_deg)
    lon_deg = np.degrees(np.asarray(grid.longitudes))
    sigma_full = np.asarray(levels.sigma_full)
    surface_idx = np.argmax(sigma_full)
    equator_idx = np.argmin(np.abs(lat_deg))

    print("\n--- Atmospheric climatology ---")
    print(f"  T(equator, surface) = {mean_t[surface_idx, equator_idx]:.1f} K")
    upper_mask = sigma_full < 0.4
    jet_max = float(np.max(np.abs(mean_u[upper_mask, :])))
    print(f"  Jet max |U|         = {jet_max:.1f} m/s")
    print(f"  EKE max             = {float(np.max(mean_eke)):.1f} m^2/s^2")

    # Precipitation near the mountain
    ilat = np.argmin(np.abs(lat_deg - mountain_lat))
    ilon = np.argmin(np.abs(lon_deg - mountain_lon))
    mm_day = mean_precip * 86400  # kg/m²/s → mm/day
    print("\n--- Precipitation (mm/day) ---")
    print(f"  Global mean         = {mm_day.mean():.2f}")
    print(f"  Global max          = {mm_day.max():.2f}")
    print(f"  At mountain peak    = {mm_day[ilat, ilon]:.2f}")

    # Zonal profile at mountain latitude
    print(f"\n  Zonal profile at {lat_deg[ilat]:.1f}N:")
    n_lon = grid.n_lon
    for j in range(0, n_lon, max(1, n_lon // 8)):
        dlon = ((lon_deg[j] - mountain_lon + 180) % 360) - 180
        print(
            f"    lon={lon_deg[j]:5.1f} (dlon={dlon:+6.1f})  "
            f"z={z_grid[ilat, j]:5.0f} m  "
            f"P={mm_day[ilat, j]:.2f} mm/day"
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moist aquaplanet with topography",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=100, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=30, help="Spinup days before averaging")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--height", type=float, default=2500.0, help="Mountain height [m]")
    parser.add_argument("--lat", type=float, default=30.0, help="Mountain latitude [deg N]")
    parser.add_argument("--lon", type=float, default=180.0, help="Mountain longitude [deg E]")
    parser.add_argument("--width", type=float, default=20.0, help="Mountain half-width [deg]")
    parser.add_argument(
        "--scheme", type=str, default="speedy", choices=["frierson", "byrne", "speedy"]
    )
    args = parser.parse_args()

    passed = run_mountain_aquaplanet(
        n_days=args.days,
        spinup_days=args.spinup,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        mountain_height=args.height,
        mountain_lat=args.lat,
        mountain_lon=args.lon,
        mountain_width=args.width,
        scheme=args.scheme,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
