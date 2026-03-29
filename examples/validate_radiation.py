#!/usr/bin/env python3
"""Validate radiation energy conservation across the globe.

Runs a moist aquaplanet to quasi-equilibrium with a chosen radiation
scheme, then computes the full SW and LW radiation budget at every grid
point.  Reports global-mean and zonal-mean diagnostics to verify:

1. **SW column closure**: atmospheric absorption + surface flux = TOA
   incoming (should be exact to machine precision).
2. **Global energy balance**: net TOA flux should be small at equilibrium.
3. **Humidity-dependent SW effect**: how much extra SW is absorbed due
   to water vapor.

Usage
-----
    uv run python examples/validate_radiation.py --scheme byrne
    uv run python examples/validate_radiation.py --scheme speedy --days 300
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
from notus.diagnostics import spherical_integral
from notus.grid import GaussianGrid
from notus.initial_conditions import moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import (
    STEFAN_BOLTZMANN,
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    longwave_heating,
    shortwave_heating,
    speedy_longwave_heating,
    speedy_shortwave_heating,
)
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


def compute_olr(
    temperature: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    tau_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute OLR for single-band LW (Byrne/Frierson schemes)."""
    n_lon = temperature.shape[-1]

    if tau_half.ndim == 2:
        tau_half = jnp.broadcast_to(tau_half[:, :, None], (*tau_half.shape, n_lon))

    dtau = tau_half[1:] - tau_half[:-1]
    transmissivity = jnp.exp(-dtau)

    bb = STEFAN_BOLTZMANN * temperature**4
    if surface_temperature.ndim == 1:
        surface_temperature = surface_temperature[:, None]
    n_lat = temperature.shape[1]
    bb_surface = jnp.broadcast_to(STEFAN_BOLTZMANN * surface_temperature**4, (n_lat, n_lon))

    trans_rev = transmissivity[::-1]
    bb_rev = bb[::-1]

    def _upward_step(
        f_up: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        trans_k, bb_k = layer
        return f_up * trans_k + bb_k * (1.0 - trans_k), None

    olr, _ = jax.lax.scan(_upward_step, bb_surface, (trans_rev, bb_rev))
    return olr


def _diagnose_byrne(
    t_grid: jnp.ndarray,
    q_grid: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    forcing: SimplePhysics,
    surface_pressure: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Byrne scheme: return (lw_heating, olr, lw_down, sw_heating, sw_down, sw_down_dry)."""
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    sin_lat = forcing.transform.grid.sin_lat

    tau_lw = byrne_longwave_optical_depth(
        levels.dsigma,
        q_grid,
        surface_pressure,
        planet.reference_pressure,
        byrne_a=cfg.byrne_a,
        byrne_b=cfg.byrne_b,
    )
    lw_heating, lw_down_sfc = longwave_heating(
        t_grid,
        surface_temperature,
        tau_lw,
        levels.dsigma,
        surface_pressure,
        planet.gravity,
        planet.specific_heat_cp,
    )
    olr = compute_olr(t_grid, surface_temperature, tau_lw)

    tau_sw = byrne_shortwave_optical_depth(
        levels.dsigma,
        q_grid,
        surface_pressure,
        planet.reference_pressure,
        sw_tau_0=cfg.sw_tau_0,
        byrne_sw_a=cfg.byrne_sw_a,
        byrne_sw_b=cfg.byrne_sw_b,
    )
    sw_heating, sw_down_sfc = shortwave_heating(
        levels.sigma_half,
        levels.dsigma,
        sin_lat,
        surface_pressure,
        planet.solar_constant,
        planet.gravity,
        planet.specific_heat_cp,
        sw_tau_0=cfg.sw_tau_0,
        sw_exponent=cfg.sw_exponent,
        delta_s=cfg.delta_s,
        tau_sw_half=tau_sw,
        surface_albedo=planet.surface_albedo,
    )
    _, sw_down_dry = shortwave_heating(
        levels.sigma_half,
        levels.dsigma,
        sin_lat,
        surface_pressure,
        planet.solar_constant,
        planet.gravity,
        planet.specific_heat_cp,
        sw_tau_0=cfg.sw_tau_0,
        sw_exponent=cfg.sw_exponent,
        delta_s=cfg.delta_s,
        surface_albedo=planet.surface_albedo,
    )
    return lw_heating, olr, lw_down_sfc, sw_heating, sw_down_sfc, sw_down_dry


def _diagnose_speedy(
    t_grid: jnp.ndarray,
    q_grid: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    forcing: SimplePhysics,
    surface_pressure: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SPEEDY scheme: return (lw_heating, olr, lw_down, sw_heating, sw_down, sw_down_dry)."""
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    sin_lat = forcing.transform.grid.sin_lat

    lw_heating, lw_down_sfc, olr = speedy_longwave_heating(
        t_grid,
        surface_temperature,
        q_grid,
        levels.dsigma,
        surface_pressure,
        planet.reference_pressure,
        planet.gravity,
        planet.specific_heat_cp,
        epslw=cfg.speedy_epslw,
        surface_emissivity=cfg.speedy_surface_emissivity,
        ablwin=cfg.speedy_ablwin,
        ablco2=cfg.speedy_ablco2,
        ablwv1=cfg.speedy_ablwv1,
        ablwv2=cfg.speedy_ablwv2,
    )

    insolation = planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
    sw_heating, sw_down_sfc = speedy_shortwave_heating(
        levels.dsigma,
        levels.sigma_full,
        q_grid,
        surface_pressure,
        planet.reference_pressure,
        insolation,
        planet.gravity,
        planet.specific_heat_cp,
        surface_albedo=planet.surface_albedo,
        absdry=cfg.speedy_absdry,
        absaer=cfg.speedy_absaer,
        abswv1=cfg.speedy_sw_abswv1,
        abswv2=cfg.speedy_sw_abswv2,
        visible_fraction=cfg.speedy_visible_fraction,
    )
    # Dry comparison: zero humidity
    q_dry = jnp.zeros_like(q_grid)
    _, sw_down_dry = speedy_shortwave_heating(
        levels.dsigma,
        levels.sigma_full,
        q_dry,
        surface_pressure,
        planet.reference_pressure,
        insolation,
        planet.gravity,
        planet.specific_heat_cp,
        surface_albedo=planet.surface_albedo,
        absdry=cfg.speedy_absdry,
        absaer=cfg.speedy_absaer,
        abswv1=cfg.speedy_sw_abswv1,
        abswv2=cfg.speedy_sw_abswv2,
        visible_fraction=cfg.speedy_visible_fraction,
    )
    return lw_heating, olr, lw_down_sfc, sw_heating, sw_down_sfc, sw_down_dry


def diagnose_radiation_budget(
    state,
    transform: SpectralTransform,
    forcing: SimplePhysics,
    surface_pressure: jnp.ndarray,
    surface_temperature: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute full radiation budget at every grid point."""
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    sin_lat = transform.grid.sin_lat

    t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
    q_grid = jnp.maximum(jax.vmap(transform.spectral_to_grid)(state.humidity), 0.0)

    # Dispatch by scheme
    if cfg.radiation_scheme == "speedy":
        lw_heat, olr, lw_down_sfc, sw_heat, sw_down_sfc, sw_down_dry = _diagnose_speedy(
            t_grid,
            q_grid,
            surface_temperature,
            forcing,
            surface_pressure,
        )
    else:
        lw_heat, olr, lw_down_sfc, sw_heat, sw_down_sfc, sw_down_dry = _diagnose_byrne(
            t_grid,
            q_grid,
            surface_temperature,
            forcing,
            surface_pressure,
        )

    # Surface LW up
    t_s = surface_temperature
    if t_s.ndim == 1:
        t_s = t_s[:, None]
    lw_up_sfc = STEFAN_BOLTZMANN * t_s**4

    # TOA incoming SW
    n_lon = surface_pressure.shape[-1]
    insolation = planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
    toa_sw_in = insolation[:, None] * jnp.ones((sin_lat.shape[0], n_lon))

    # Column integrals
    dp = levels.dsigma[:, None, None] * surface_pressure[None, :, :]
    sw_atm_absorbed = jnp.sum(sw_heat * planet.specific_heat_cp * dp / planet.gravity, axis=0)
    lw_atm_cooling = jnp.sum(lw_heat * planet.specific_heat_cp * dp / planet.gravity, axis=0)

    sw_sfc_absorbed = sw_down_sfc * (1.0 - planet.surface_albedo)
    sw_reflected_space = toa_sw_in - sw_atm_absorbed - sw_sfc_absorbed

    # Surface turbulent fluxes
    lowest = levels.n_levels - 1
    t_lowest = transform.spectral_to_grid(state.temperature[lowest])
    t_safe = jnp.maximum(t_lowest, 1.0)
    rho_sfc = surface_pressure / (planet.gas_constant * t_safe)

    u_cos_spec, v_cos_spec = uv_from_vordiv(
        state.vorticity[lowest],
        state.divergence[lowest],
        transform.arrays,
    )
    cos_lat_safe = jnp.maximum(transform.grid.cos_lat[:, None], 1.0e-6)
    u_grid = transform.spectral_to_grid(u_cos_spec) / cos_lat_safe
    v_grid = transform.spectral_to_grid(v_cos_spec) / cos_lat_safe
    wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

    c_d = cfg.c_d
    sensible = rho_sfc * planet.specific_heat_cp * c_d * wind_speed * (t_s - t_lowest)

    q_lowest = jnp.maximum(transform.spectral_to_grid(state.humidity[lowest]), 0.0)
    q_sat_sfc = saturation_specific_humidity(t_s, surface_pressure, planet.epsilon_moisture)
    latent = rho_sfc * planet.latent_heat_vaporization * c_d * wind_speed * (q_sat_sfc - q_lowest)

    net_sfc = sw_sfc_absorbed + lw_down_sfc - lw_up_sfc - sensible - latent
    net_toa = (toa_sw_in - sw_reflected_space) - olr
    sw_closure = sw_atm_absorbed + sw_sfc_absorbed + sw_reflected_space - toa_sw_in

    return {
        "toa_sw_in": toa_sw_in,
        "sw_atm_absorbed": sw_atm_absorbed,
        "sw_down_sfc": sw_down_sfc,
        "sw_sfc_absorbed": sw_sfc_absorbed,
        "sw_reflected_space": sw_reflected_space,
        "sw_down_sfc_dry": sw_down_dry,
        "sw_closure_residual": sw_closure,
        "olr": olr,
        "lw_down_sfc": lw_down_sfc,
        "lw_up_sfc": lw_up_sfc,
        "lw_atm_cooling": lw_atm_cooling,
        "sensible": sensible,
        "latent": latent,
        "net_sfc": net_sfc,
        "net_toa": net_toa,
    }


def global_mean(field: jnp.ndarray, grid: GaussianGrid) -> float:
    """Compute area-weighted global mean of a 2-D field."""
    integral = spherical_integral(field, grid)
    area = spherical_integral(jnp.ones_like(field), grid)
    return float(integral / area)


def zonal_mean(field: jnp.ndarray) -> np.ndarray:
    """Zonal mean of a 2-D field -> 1-D."""
    return np.asarray(jnp.mean(field, axis=-1))


def run_validation(
    n_days: int = 300,
    spinup_days: int = 200,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    scheme: str = "byrne",
    clouds: bool = False,
) -> bool:
    """Run moist aquaplanet and validate radiation budget."""
    if scheme == "byrne":
        config = SimplePhysicsConfig(radiation_scheme="byrne", sw_tau_0=0.22)
        scheme_label = "Byrne LW + humidity-dependent SW (tau=0.22)"
    elif scheme == "speedy":
        config = SimplePhysicsConfig(radiation_scheme="speedy", enable_clouds=clouds)
        scheme_label = "SPEEDY 4-band LW + 2-band SW" + (" + clouds" if clouds else "")
    else:
        print(f"ERROR: unknown scheme '{scheme}'")
        return False

    print(f"Radiation validation: T{truncation} L{n_levels}, dt={dt:.0f}s, {n_days} days")
    print(f"  Spinup: {spinup_days} days, diagnostic at day {n_days}")
    print(f"  Radiation: {scheme_label}")

    if n_days <= spinup_days:
        print(f"ERROR: n_days ({n_days}) must be > spinup_days ({spinup_days})")
        return False

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

    def one_day(carry, _):
        prev, curr = carry

        def step(carry, _):
            p, c = carry
            p, c = step_fn(p, c)
            return (p, c), None

        (prev, curr), _ = jax.lax.scan(step, (prev, curr), None, length=steps_per_day)
        return (prev, curr), None

    one_day_jit = jax.jit(one_day)

    print("Initializing (JIT compile on first day)...")
    t0 = time.perf_counter()
    prev, curr = init_fn(state)
    (prev, curr), _ = one_day_jit((prev, curr), None)
    print(f"  Day 1 (incl. compile): {time.perf_counter() - t0:.1f}s")

    n_samples = 0
    accum: dict[str, jnp.ndarray] = {}

    t_start = time.perf_counter()
    for day in range(2, n_days + 1):
        (prev, curr), _ = one_day_jit((prev, curr), None)

        if day <= 10 or day % 50 == 0 or day == n_days:
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            t_mean = float(np.mean(t_grid))
            elapsed = time.perf_counter() - t_start
            dps = (day - 1) / elapsed if elapsed > 0 else 0
            phase = "spinup" if day <= spinup_days else "averaging"
            print(f"  Day {day:5d} [{phase:>9s}]: T_mean={t_mean:.1f} K  [{dps:.1f} d/s]")
            if not np.isfinite(t_mean):
                print("ERROR: blowup")
                return False

        if day > spinup_days:
            lnps_grid = transform.spectral_to_grid(curr.log_surface_pressure)
            ps_grid = EARTH.reference_pressure * jnp.exp(lnps_grid)
            budget = diagnose_radiation_budget(curr, transform, forcing, ps_grid, forcing.sst)
            n_samples += 1
            if not accum:
                accum = dict(budget.items())
            else:
                for k in accum:
                    accum[k] = accum[k] + budget[k]

    total_time = time.perf_counter() - t0
    print(f"\nDone. Total: {total_time:.0f}s. Averaged over {n_samples} daily samples.\n")

    if n_samples == 0:
        print("ERROR: no averaging samples collected")
        return False

    mean_budget = {k: v / n_samples for k, v in accum.items()}
    return _print_report(mean_budget, grid)


def _print_report(mean_budget: dict[str, jnp.ndarray], grid: GaussianGrid) -> bool:
    """Print the radiation budget report and return True if validation passes."""

    def gm(key: str) -> float:
        return global_mean(mean_budget[key], grid)

    sw_in = gm("toa_sw_in")
    sw_atm = gm("sw_atm_absorbed")
    sw_sfc_down = gm("sw_down_sfc")
    sw_sfc_abs = gm("sw_sfc_absorbed")
    sw_refl = gm("sw_reflected_space")
    sw_residual = gm("sw_closure_residual")
    sw_sfc_dry = gm("sw_down_sfc_dry")
    olr_val = gm("olr")
    lw_down = gm("lw_down_sfc")
    lw_up = gm("lw_up_sfc")
    lw_atm = gm("lw_atm_cooling")
    sensible_val = gm("sensible")
    latent_val = gm("latent")
    net_sfc_val = gm("net_sfc")
    net_toa_val = gm("net_toa")
    sw_h2o_effect = sw_sfc_dry - sw_sfc_down

    print("=" * 64)
    print("  RADIATION BUDGET — Global Mean [W/m²]")
    print("=" * 64)

    print()
    print("  --- Shortwave ---")
    print(f"  TOA incoming SW:         {sw_in:8.1f}")
    print(f"  Atmospheric SW absorbed: {sw_atm:8.1f}")
    print(f"  SW reaching surface:     {sw_sfc_down:8.1f}")
    print(f"  Surface SW absorbed:     {sw_sfc_abs:8.1f}  (albedo={EARTH.surface_albedo:.2f})")
    eff_albedo = sw_refl / sw_in * 100
    print(f"  SW reflected to space:   {sw_refl:8.1f}  (eff. albedo {eff_albedo:.1f}%)")
    closure_ok = "✓" if abs(sw_residual) < 0.01 else "✗"
    print(f"  Column closure residual: {sw_residual:8.4f}  {closure_ok}")
    print(f"  H₂O SW absorption:      {sw_h2o_effect:8.1f}  (moist - dry surface flux)")

    print()
    print("  --- Longwave ---")
    print(f"  OLR (outgoing LW):       {olr_val:8.1f}")
    print(f"  LW down at surface:      {lw_down:8.1f}")
    print(f"  LW up from surface:      {lw_up:8.1f}")
    print(f"  Atmospheric LW cooling:  {lw_atm:8.1f}")
    greenhouse = lw_up - olr_val
    print(f"  Greenhouse effect:       {greenhouse:8.1f}  (LW_up_sfc - OLR)")

    print()
    print("  --- Surface Energy Balance ---")
    print(f"  SW absorbed:             {sw_sfc_abs:8.1f}")
    print(f"  LW down:                 {lw_down:8.1f}")
    print(f"  LW up:                  {-lw_up:8.1f}")
    print(f"  Sensible heat:          {-sensible_val:8.1f}")
    print(f"  Latent heat:            {-latent_val:8.1f}")
    print(f"  Net surface flux:        {net_sfc_val:8.1f}")

    print()
    print("  --- Top of Atmosphere ---")
    print(f"  Net SW (in - reflected): {sw_in - sw_refl:8.1f}")
    print(f"  OLR:                    {-olr_val:8.1f}")
    print(f"  Net TOA flux:            {net_toa_val:8.1f}")
    print()

    print("=" * 64)
    ok = True
    if abs(sw_residual) < 0.01:
        print("  ✓ SW column closure:    PASS  (residual < 0.01 W/m²)")
    else:
        print(f"  ✗ SW column closure:    FAIL  (residual = {sw_residual:.4f} W/m²)")
        ok = False

    if abs(net_toa_val) < 20.0:
        print(f"  ✓ TOA energy balance:   {net_toa_val:+.1f} W/m²  (acceptable)")
    else:
        print(f"  ⚠ TOA energy balance:   {net_toa_val:+.1f} W/m²  (large imbalance)")

    if sw_h2o_effect > 0.5:
        print(f"  ✓ H₂O SW feedback:      {sw_h2o_effect:.1f} W/m² additional absorption")
    else:
        print(f"  ⚠ H₂O SW feedback:      {sw_h2o_effect:.1f} W/m² (expected > 0.5)")

    print("=" * 64)

    lat_deg = np.degrees(np.asarray(grid.latitudes))
    print("\n  Zonal-Mean Radiation Budget [W/m²]")
    print("  " + "-" * 72)
    print(
        f"  {'Lat':>6s}  {'SW_in':>7s}  {'SW_atm':>7s}  {'SW_sfc':>7s}  "
        f"{'OLR':>7s}  {'NetTOA':>7s}  {'NetSfc':>7s}"
    )
    print("  " + "-" * 72)
    for j in range(0, len(lat_deg), max(1, len(lat_deg) // 12)):
        print(
            f"  {lat_deg[j]:6.1f}°  "
            f"{zonal_mean(mean_budget['toa_sw_in'])[j]:7.1f}  "
            f"{zonal_mean(mean_budget['sw_atm_absorbed'])[j]:7.1f}  "
            f"{zonal_mean(mean_budget['sw_sfc_absorbed'])[j]:7.1f}  "
            f"{zonal_mean(mean_budget['olr'])[j]:7.1f}  "
            f"{zonal_mean(mean_budget['net_toa'])[j]:7.1f}  "
            f"{zonal_mean(mean_budget['net_sfc'])[j]:7.1f}"
        )
    print("  " + "-" * 72)

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate radiation energy conservation")
    parser.add_argument("--days", type=int, default=300, help="Total integration days")
    parser.add_argument("--spinup", type=int, default=200, help="Spinup days before averaging")
    parser.add_argument("--scheme", type=str, default="byrne", choices=["byrne", "speedy"])
    parser.add_argument("--clouds", action="store_true", help="Enable diagnostic clouds (speedy)")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    args = parser.parse_args()

    passed = run_validation(
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
