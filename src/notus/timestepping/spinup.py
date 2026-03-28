"""Prescribed-SST spinup and Q-flux diagnosis.

Runs an atmospheric integration under prescribed SST to produce:
1. A spun-up atmospheric state for warm-starting coupled slab ocean runs
2. A diagnosed Q-flux (the implied ocean heat transport needed to maintain
   the prescribed SST pattern)

The Q-flux diagnosis uses the exact same surface flux computation as the
coupled slab ocean stepper, ensuring self-consistency.

Typical usage
-------------
    # In-memory (recommended for scripts)
    result = spinup_prescribed_sst(state, forcing, transform, levels, ...)
    ocean = OceanState(surface_temperature=forcing.sst)
    init_fn, step_fn = build_coupled_pe_stepper(..., q_flux=result.q_flux)
    prev, curr, ocean = init_fn(result.state, ocean)

    # With disk I/O (for separate CLI invocations)
    save_restart("restart.npz", result.state)
    np.savez("qflux.npz", q_flux=result.q_flux, ...)
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.forcing import Forcing
from notus.physics.radiation import (
    byrne_longwave_optical_depth,
    longwave_optical_depth,
    lw_down_surface,
)
from notus.physics.surface import compute_net_surface_flux
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


@dataclasses.dataclass(frozen=True)
class SpinupResult:
    """Result of a prescribed-SST spinup integration.

    Attributes
    ----------
    state : PrimitiveEquationState
        Spun-up atmospheric state (spectral space).
    q_flux : jnp.ndarray
        Diagnosed Q-flux [W/m²], shape ``(n_lat,)``.
        ``Q = -<F_net>`` where ``<F_net>`` is the time-mean net surface flux.
    mean_net_flux : jnp.ndarray
        Time-mean net downward surface flux [W/m²], shape ``(n_lat,)``.
    n_samples : int
        Number of daily samples in the time average.
    """

    state: PrimitiveEquationState
    q_flux: jnp.ndarray
    mean_net_flux: jnp.ndarray
    n_samples: int


def _diagnose_surface_flux(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    forcing: Forcing,
    surface_pressure: jnp.ndarray,
) -> jnp.ndarray:
    """Compute zonal-mean net downward surface energy flux [W/m²].

    Uses ``compute_net_surface_flux`` — the same function as the coupled
    slab ocean stepper — for self-consistent Q-flux diagnosis.
    """
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    lowest = levels.n_levels - 1
    sin_lat = transform.grid.sin_lat

    # Full temperature field (needed for two-stream LW)
    t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
    t_lowest = t_grid[lowest]

    # Surface winds
    u_cos_spec, v_cos_spec = uv_from_vordiv(
        state.vorticity[lowest],
        state.divergence[lowest],
        transform.arrays,
    )
    cos_lat_safe = jnp.maximum(transform.grid.cos_lat[:, None], 1.0e-6)
    u_grid = transform.spectral_to_grid(u_cos_spec) / cos_lat_safe
    v_grid = transform.spectral_to_grid(v_cos_spec) / cos_lat_safe
    wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

    # Insolation
    if cfg.sw_tau_0 > 0.0:
        insolation = (
            planet.solar_constant / 4.0
            * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
        )
    else:
        insolation = jnp.zeros_like(sin_lat)

    # Downward LW from two-stream solver
    if cfg.radiation_scheme == "byrne" and state.humidity is not None:
        q_grid = jnp.maximum(
            jax.vmap(transform.spectral_to_grid)(state.humidity), 0.0,
        )
        tau_half = byrne_longwave_optical_depth(
            levels.dsigma, q_grid, surface_pressure,
            planet.reference_pressure,
            byrne_a=cfg.byrne_a, byrne_b=cfg.byrne_b,
        )
    else:
        tau_half = longwave_optical_depth(
            levels.sigma_half, sin_lat,
            tau_equator=cfg.tau_equator, tau_pole=cfg.tau_pole,
            linear_fraction=cfg.linear_fraction, alpha=cfg.alpha,
        )
    lw_down = lw_down_surface(t_grid, tau_half)

    # Humidity at lowest level
    q_lowest = jnp.zeros_like(t_lowest)
    if state.humidity is not None:
        q_lowest = jnp.maximum(
            transform.spectral_to_grid(state.humidity[lowest]), 0.0,
        )

    # Net flux (same function as coupled stepper)
    net_flux = compute_net_surface_flux(
        forcing.sst, t_lowest, q_lowest, wind_speed, surface_pressure,
        insolation, lw_down,
        gravity=planet.gravity, gas_constant=planet.gas_constant,
        specific_heat_cp=planet.specific_heat_cp,
        epsilon=planet.epsilon_moisture,
        latent_heat=planet.latent_heat_vaporization,
        drag_coefficient=cfg.c_d, surface_albedo=planet.surface_albedo,
        sw_tau_0=cfg.sw_tau_0,
    )

    return jnp.mean(net_flux, axis=-1)


def spinup_prescribed_sst(
    state: PrimitiveEquationState,
    forcing: Forcing,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    reference_temperature: np.ndarray,
    surface_geopotential: jnp.ndarray,
    dt: float,
    *,
    spinup_days: int = 100,
    averaging_days: int = 200,
    spectral_filter: jnp.ndarray | None = None,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
    robert_coeff: float = 0.05,
    alpha: float = 0.5,
    verbose: bool = True,
) -> SpinupResult:
    """Spin up the atmosphere under prescribed SST and diagnose Q-flux.

    Runs a prescribed-SST integration for ``spinup_days + averaging_days``,
    then returns the spun-up atmospheric state and diagnosed Q-flux.

    Parameters
    ----------
    state : PrimitiveEquationState
        Initial atmospheric state (can be cold isothermal).
    forcing : Forcing
        Physics forcing (must have ``config``, ``sst``, ``planet``, ``levels``).
    transform : SpectralTransform
        Spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    reference_temperature : np.ndarray
        Reference temperature profile for semi-implicit solver.
    surface_geopotential : jnp.ndarray
        Surface geopotential (spectral).
    dt : float
        Timestep [s].
    spinup_days : int
        Days to discard before averaging.
    averaging_days : int
        Days to average for Q-flux diagnosis.
    spectral_filter : jnp.ndarray or None
        Spectral filter array.
    verbose : bool
        Print progress.

    Returns
    -------
    SpinupResult
        Spun-up state, diagnosed Q-flux, and diagnostics.
    """
    if spectral_filter is None:
        spectral_filter = exponential_filter(transform.arrays, dt)

    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=planet,
        levels=levels,
        reference_temperature=reference_temperature,
        surface_geopotential=surface_geopotential,
        dt=dt,
        forcing=forcing,
        spectral_filter=spectral_filter,
        diffusion_order=diffusion_order,
        diffusion_timescale=diffusion_timescale,
        robert_coeff=robert_coeff,
        alpha=alpha,
    )

    steps_per_day = int(86400 / dt)
    n_days = spinup_days + averaging_days
    n_lat = transform.grid.n_lat

    def one_day(carry, _):
        prev, curr = carry
        def step(carry, _):
            p, c = carry
            p, c = step_fn(p, c)
            return (p, c), None
        (prev, curr), _ = jax.lax.scan(step, (prev, curr), None, length=steps_per_day)
        return (prev, curr), None

    one_day_jit = jax.jit(one_day)
    diagnose_jit = jax.jit(
        lambda s, ps: _diagnose_surface_flux(s, transform, forcing, ps),
    )

    # Initialize
    prev, curr = init_fn(state)
    (prev, curr), _ = one_day_jit((prev, curr), None)

    # Integration
    flux_accum = np.zeros(n_lat)
    n_samples = 0
    spinup_state = curr

    for day in range(2, n_days + 1):
        (prev, curr), _ = one_day_jit((prev, curr), None)

        if day == spinup_days + 1:
            spinup_state = curr

        if day > spinup_days:
            lnps_grid = transform.spectral_to_grid(curr.log_surface_pressure)
            ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)
            flux_accum += np.asarray(diagnose_jit(curr, ps_grid))
            n_samples += 1

        if verbose and (day <= 10 or day % 50 == 0 or day == n_days):
            t_grid = np.asarray(
                jax.vmap(transform.spectral_to_grid)(curr.temperature),
            )
            phase = "spinup" if day <= spinup_days else "averaging"
            print(
                f"  Day {day:5d} [{phase:>9s}]: "
                f"T_mean={np.mean(t_grid):.1f} K",
            )

    mean_flux = flux_accum / n_samples
    q_flux = jnp.array(-mean_flux)

    if verbose:
        lat_deg = np.degrees(np.asarray(transform.grid.latitudes))
        eq_idx = np.argmin(np.abs(lat_deg))
        print(f"\n  Q-flux(equator) = {float(q_flux[eq_idx]):+.1f} W/m^2")
        print(f"  Q-flux range = [{float(jnp.min(q_flux)):.1f}, {float(jnp.max(q_flux)):.1f}] W/m^2")

    return SpinupResult(
        state=spinup_state,
        q_flux=q_flux,
        mean_net_flux=jnp.array(mean_flux),
        n_samples=n_samples,
    )
