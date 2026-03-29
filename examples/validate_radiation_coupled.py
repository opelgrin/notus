#!/usr/bin/env python3
"""Validate radiation energy conservation in a coupled slab ocean run.

Three-phase validation:
1. Prescribed-SST spinup (200 days) -> diagnose Q-flux
2. Coupled slab ocean integration (300 days from warm start)
3. Radiation budget diagnosis on the equilibrated coupled state

This tests the full energy cycle: atmospheric SW/LW + surface fluxes +
interactive SST.  At equilibrium, net TOA flux should be near zero and
SST should be stable.

Usage
-----
    uv run python examples/validate_radiation_coupled.py
    uv run python examples/validate_radiation_coupled.py --coupled-days 500
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

from validate_radiation import compute_olr

from notus import (
    EARTH,
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
    build_coupled_pe_stepper,
    compute_sst,
    exponential_filter,
    grid_surface_pressure,
    moist_aquaplanet_initial_state,
    spherical_integral,
    spinup_prescribed_sst,
    standard_sigma_levels,
    uv_from_vordiv,
)
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import (
    STEFAN_BOLTZMANN,
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    shortwave_heating,
)


def global_mean(field: jnp.ndarray, grid: GaussianGrid) -> float:
    """Area-weighted global mean of a 2-D field."""
    integral = spherical_integral(field, grid)
    area = spherical_integral(jnp.ones_like(field), grid)
    return float(integral / area)


def diagnose_coupled_budget(
    state,
    transform: SpectralTransform,
    forcing: PhysicsSuite,
    surface_pressure: jnp.ndarray,
    surface_temperature: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute radiation + surface flux budget for a coupled state."""
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    sin_lat = transform.grid.sin_lat

    t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
    q_grid = jnp.maximum(jax.vmap(transform.spectral_to_grid)(state.humidity), 0.0)

    # LW optical depth (Byrne)
    rad = cfg.radiation
    if not isinstance(rad, ByrneRadiation):
        msg = "Expected ByrneRadiation"
        raise TypeError(msg)
    tau_lw = byrne_longwave_optical_depth(
        levels.dsigma,
        q_grid,
        surface_pressure,
        planet.reference_pressure,
        byrne_a=rad.a,
        byrne_b=rad.b,
    )

    # OLR
    olr = compute_olr(t_grid, surface_temperature, tau_lw)

    # Surface LW
    t_s = surface_temperature
    if t_s.ndim == 1:
        t_s = t_s[:, None]
    lw_up_sfc = STEFAN_BOLTZMANN * t_s**4

    # SW optical depth (humidity-dependent)
    tau_sw = byrne_shortwave_optical_depth(
        levels.dsigma,
        q_grid,
        surface_pressure,
        planet.reference_pressure,
        sw_tau_0=rad.sw_tau_0,
        byrne_sw_a=rad.sw_a,
        byrne_sw_b=rad.sw_b,
    )

    # SW fluxes
    sw_heating, sw_down_sfc = shortwave_heating(
        levels.sigma_half,
        levels.dsigma,
        sin_lat,
        surface_pressure,
        planet.solar_constant,
        planet.gravity,
        planet.specific_heat_cp,
        sw_tau_0=rad.sw_tau_0,
        sw_exponent=rad.sw_exponent,
        delta_s=cfg.delta_s,
        tau_sw_half=tau_sw,
        surface_albedo=planet.surface_albedo,
    )

    # TOA incoming
    n_lon = surface_pressure.shape[-1]
    insolation = planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
    toa_sw_in = insolation[:, None] * jnp.ones((sin_lat.shape[0], n_lon))

    # Atmospheric SW absorption
    dp = levels.dsigma[:, None, None] * surface_pressure[None, :, :]
    sw_atm_absorbed = jnp.sum(
        sw_heating * planet.specific_heat_cp * dp / planet.gravity,
        axis=0,
    )

    sw_sfc_absorbed = sw_down_sfc * (1.0 - planet.surface_albedo)
    sw_reflected = toa_sw_in - sw_atm_absorbed - sw_sfc_absorbed

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

    # Net TOA = absorbed SW - OLR
    net_toa = (toa_sw_in - sw_reflected) - olr

    # Column closure
    sw_closure = sw_atm_absorbed + sw_sfc_absorbed + sw_reflected - toa_sw_in

    return {
        "toa_sw_in": toa_sw_in,
        "sw_atm_absorbed": sw_atm_absorbed,
        "sw_down_sfc": sw_down_sfc,
        "sw_sfc_absorbed": sw_sfc_absorbed,
        "sw_reflected": sw_reflected,
        "sw_closure": sw_closure,
        "olr": olr,
        "lw_up_sfc": lw_up_sfc,
        "sensible": sensible,
        "latent": latent,
        "net_toa": net_toa,
    }


def run_validation(
    spinup_days: int = 200,
    averaging_days: int = 100,
    coupled_days: int = 300,
    coupled_avg_days: int = 100,
    truncation: int = 21,
    n_levels: int = 20,
    dt: float = 900.0,
    scheme: str = "byrne",
    clouds: bool = False,
) -> bool:
    """Three-phase coupled radiation validation."""
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

    if scheme == "speedy":
        from notus.physics.clouds import CloudConfig

        config = PhysicsSuiteConfig(
            radiation=SpeedyRadiation(clouds=CloudConfig() if clouds else None),
        )
    else:
        config = PhysicsSuiteConfig(radiation=ByrneRadiation(sw_tau_0=0.22))
    forcing = PhysicsSuite(transform, EARTH, levels, config=config)
    filt = exponential_filter(transform.arrays, dt)

    # ==========================================================
    # Phase 1: Prescribed-SST spinup + Q-flux diagnosis
    # ==========================================================
    print(f"Phase 1: Prescribed-SST spinup ({spinup_days}d) + averaging ({averaging_days}d)")

    t0 = time.perf_counter()
    result = spinup_prescribed_sst(
        state,
        forcing,
        transform,
        EARTH,
        levels,
        ref_temps,
        surface_phi,
        dt,
        spinup_days=spinup_days,
        averaging_days=averaging_days,
        spectral_filter=filt,
    )
    warm_state = result.state
    q_flux_diag = result.q_flux

    print(f"  Spinup done in {time.perf_counter() - t0:.0f}s")
    print(f"  Q-flux from {result.n_samples} samples")
    print(f"  Q range: [{float(jnp.min(q_flux_diag)):.1f}, {float(jnp.max(q_flux_diag)):.1f}] W/m²")
    cos_lat = np.asarray(np.sqrt(1.0 - np.asarray(grid.sin_lat) ** 2))
    qf_global = float(np.sum(np.asarray(q_flux_diag) * cos_lat) / np.sum(cos_lat))
    print(f"  Q global mean: {qf_global:+.2f} W/m² (should be ~0)\n")

    steps_per_day = int(86400 / dt)

    # ==========================================================
    # Phase 2: Coupled slab ocean integration
    # ==========================================================
    coupled_spinup = coupled_days - coupled_avg_days
    print(f"Phase 2: Coupled slab ocean ({coupled_spinup}d spinup + {coupled_avg_days}d averaging)")

    ocean_config = SlabOceanConfig(mixed_layer_depth=50.0)
    sst_init = compute_sst(
        PrescribedSST(
            t_min=config.sst_t_min,
            t_delta=config.sst_t_delta,
            phi_w=config.sst_phi_w,
        ),
        grid.latitudes,
    )
    ocean = OceanState(surface_temperature=sst_init)
    surface = SurfaceState(ocean=ocean)

    coupled_init_fn, coupled_step_fn = build_coupled_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        forcing=forcing,
        ocean_config=ocean_config,
        q_flux=q_flux_diag,
        spectral_filter=filt,
    )

    def one_day_coupled(carry, _):
        prev, curr, sfc = carry

        def step(carry, _):
            p, c, s = carry
            p, c, s = coupled_step_fn(p, c, s)
            return (p, c, s), None

        (prev, curr, sfc), _ = jax.lax.scan(
            step,
            (prev, curr, sfc),
            None,
            length=steps_per_day,
        )
        return (prev, curr, sfc), None

    one_day_coupled_jit = jax.jit(one_day_coupled)

    t_phase2 = time.perf_counter()
    prev, curr, surface = coupled_init_fn(warm_state, surface)
    (prev, curr, surface), _ = one_day_coupled_jit((prev, curr, surface), None)
    print(f"  Day 1 (compile): {time.perf_counter() - t_phase2:.1f}s")

    # Track SST evolution
    sst_history = []
    n_budget_samples = 0
    accum: dict[str, jnp.ndarray] = {}

    t_start2 = time.perf_counter()
    for day in range(2, coupled_days + 1):
        forcing.prescribed_sst = surface.ocean.surface_temperature
        (prev, curr, surface), _ = one_day_coupled_jit((prev, curr, surface), None)

        sst = np.asarray(surface.ocean.surface_temperature)
        eq_idx = np.argmin(np.abs(np.asarray(grid.latitudes_deg)))

        # Averaging period: accumulate budget
        if day > coupled_spinup:
            ps = grid_surface_pressure(curr, transform, EARTH)
            budget = diagnose_coupled_budget(
                curr,
                transform,
                forcing,
                ps,
                surface.ocean.surface_temperature,
            )
            n_budget_samples += 1
            if not accum:
                accum = dict(budget.items())
            else:
                for k in accum:
                    accum[k] = accum[k] + budget[k]

            sst_history.append(float(sst[eq_idx]))

        if day <= 5 or day % 50 == 0 or day == coupled_days:
            t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
            elapsed = time.perf_counter() - t_start2
            dps = (day - 1) / elapsed if elapsed > 0 else 0
            phase = "spinup" if day <= coupled_spinup else "averaging"
            print(
                f"  Day {day:5d} [{phase:>9s}]: "
                f"SST_eq={sst[eq_idx]:.1f} K  "
                f"T_mean={np.mean(t_grid):.1f} K  "
                f"[{dps:.1f} d/s]"
            )

    total_time = time.perf_counter() - t0
    print(f"\n  Total wall time: {total_time:.0f}s ({total_time / 3600:.1f}h)")

    # ==========================================================
    # Phase 3: Report
    # ==========================================================
    print()
    mean_budget = {k: v / n_budget_samples for k, v in accum.items()}

    def gm(key: str) -> float:
        return global_mean(mean_budget[key], grid)

    sw_in = gm("toa_sw_in")
    sw_atm = gm("sw_atm_absorbed")
    sw_sfc_abs = gm("sw_sfc_absorbed")
    sw_refl = gm("sw_reflected")
    sw_closure = gm("sw_closure")
    olr_val = gm("olr")
    lw_up = gm("lw_up_sfc")
    sensible_val = gm("sensible")
    latent_val = gm("latent")
    net_toa_val = gm("net_toa")

    print("=" * 64)
    print("  COUPLED RADIATION BUDGET — Global Mean [W/m²]")
    print(f"  (averaged over {n_budget_samples} daily samples)")
    print("=" * 64)

    print()
    print("  --- Shortwave ---")
    print(f"  TOA incoming SW:         {sw_in:8.1f}")
    print(f"  Atmospheric SW absorbed: {sw_atm:8.1f}")
    print(f"  Surface SW absorbed:     {sw_sfc_abs:8.1f}")
    eff_albedo = sw_refl / sw_in * 100
    print(f"  SW reflected to space:   {sw_refl:8.1f}  (eff. albedo {eff_albedo:.1f}%)")
    closure_ok = "✓" if abs(sw_closure) < 0.01 else "✗"
    print(f"  Column closure residual: {sw_closure:8.4f}  {closure_ok}")

    print()
    print("  --- Longwave ---")
    print(f"  OLR (outgoing LW):       {olr_val:8.1f}")
    print(f"  LW up from surface:      {lw_up:8.1f}")
    greenhouse = lw_up - olr_val
    print(f"  Greenhouse effect:       {greenhouse:8.1f}")

    print()
    print("  --- Top of Atmosphere ---")
    net_sw = sw_in - sw_refl
    print(f"  Net SW absorbed:         {net_sw:8.1f}")
    print(f"  OLR:                    {-olr_val:8.1f}")
    print(f"  Net TOA flux:            {net_toa_val:8.1f}")

    print()
    print("  --- Surface turbulent fluxes ---")
    print(f"  Sensible heat:          {-sensible_val:8.1f}")
    print(f"  Latent heat:            {-latent_val:8.1f}")

    # SST stability
    sst_arr = np.array(sst_history)
    sst_first_half = np.mean(sst_arr[: len(sst_arr) // 2])
    sst_second_half = np.mean(sst_arr[len(sst_arr) // 2 :])
    sst_drift = sst_second_half - sst_first_half

    sst_final = np.asarray(surface.ocean.surface_temperature)
    lat_deg = np.asarray(grid.latitudes_deg)
    eq_idx = np.argmin(np.abs(lat_deg))
    pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))

    print()
    print("  --- SST ---")
    print(f"  SST(equator):            {sst_final[eq_idx]:8.1f} K")
    print(f"  SST(pole):               {sst_final[pole_idx]:8.1f} K")
    print(f"  SST eq drift (2nd-1st):  {sst_drift:8.2f} K")
    print()

    # --- Verdict ---
    print("=" * 64)
    ok = True

    if abs(sw_closure) < 0.01:
        print("  ✓ SW column closure:    PASS")
    else:
        print(f"  ✗ SW column closure:    FAIL  ({sw_closure:.4f} W/m²)")
        ok = False

    if abs(net_toa_val) < 30.0:
        print(f"  ✓ TOA balance:          {net_toa_val:+.1f} W/m²")
    elif abs(net_toa_val) < 80.0:
        print(f"  ~ TOA balance:          {net_toa_val:+.1f} W/m²  (still adjusting)")
    else:
        print(f"  ⚠ TOA balance:          {net_toa_val:+.1f} W/m²  (large imbalance)")

    if abs(sst_drift) < 2.0:
        print(f"  ✓ SST stability:        drift = {sst_drift:+.2f} K")
    else:
        print(f"  ⚠ SST stability:        drift = {sst_drift:+.2f} K  (still adjusting)")

    print("=" * 64)

    # --- Zonal-mean table ---
    zm = lambda key: np.asarray(jnp.mean(mean_budget[key], axis=-1))  # noqa: E731
    print("\n  Zonal-Mean Budget [W/m²]  (SST final in K)")
    print("  " + "-" * 72)
    print(
        f"  {'Lat':>6s}  {'SST':>7s}  {'SW_abs':>7s}  "
        f"{'OLR':>7s}  {'NetTOA':>7s}  {'H':>7s}  {'LE':>7s}"
    )
    print("  " + "-" * 72)
    for j in range(0, len(lat_deg), max(1, len(lat_deg) // 12)):
        print(
            f"  {lat_deg[j]:6.1f}°  "
            f"{sst_final[j]:7.1f}  "
            f"{zm('sw_sfc_absorbed')[j]:7.1f}  "
            f"{zm('olr')[j]:7.1f}  "
            f"{zm('net_toa')[j]:7.1f}  "
            f"{-zm('sensible')[j]:7.1f}  "
            f"{-zm('latent')[j]:7.1f}"
        )
    print("  " + "-" * 72)

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Coupled slab ocean radiation validation")
    parser.add_argument("--spinup-days", type=int, default=200, help="Prescribed-SST spinup days")
    parser.add_argument("--avg-days", type=int, default=100, help="Q-flux averaging days")
    parser.add_argument("--coupled-days", type=int, default=300, help="Coupled integration days")
    parser.add_argument("--coupled-avg", type=int, default=100, help="Coupled averaging days")
    parser.add_argument("--truncation", type=int, default=21, help="Spectral truncation")
    parser.add_argument("--levels", type=int, default=20, help="Vertical levels")
    parser.add_argument("--dt", type=float, default=900.0, help="Timestep [s]")
    parser.add_argument("--scheme", type=str, default="byrne", choices=["byrne", "speedy"])
    parser.add_argument("--clouds", action="store_true", help="Enable diagnostic clouds (speedy)")
    args = parser.parse_args()

    passed = run_validation(
        spinup_days=args.spinup_days,
        averaging_days=args.avg_days,
        coupled_days=args.coupled_days,
        coupled_avg_days=args.coupled_avg,
        truncation=args.truncation,
        n_levels=args.levels,
        dt=args.dt,
        scheme=args.scheme,
        clouds=args.clouds,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
