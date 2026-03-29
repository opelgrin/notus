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
    ocean = OceanState(surface_temperature=forcing.prescribed_sst)
    init_fn, step_fn = build_coupled_pe_stepper(..., q_flux=result.q_flux)
    prev, curr, ocean = init_fn(result.state, ocean)

    # With disk I/O (for separate CLI invocations)
    save_restart("restart.npz", result.state)
    np.savez("qflux.npz", q_flux=result.q_flux, ...)
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators import exponential_filter
from notus.operators.vector import uv_from_vordiv
from notus.physics.clouds import diagnose_clouds
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.physics_suite import PhysicsSuite
from notus.physics.radiation import (
    ByrneRadiation,
    FriersonRadiation,
    SpeedyRadiation,
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    longwave_optical_depth,
    lw_down_surface,
    shortwave_heating,
    speedy_lw_down_surface,
    speedy_shortwave_heating,
)
from notus.physics.surface import compute_net_surface_flux
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


logger = logging.getLogger(__name__)


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
    forcing: PhysicsSuite,
    surface_pressure: jnp.ndarray,
    surface_albedo: float | None = None,
) -> jnp.ndarray:
    """Compute zonal-mean net downward surface energy flux [W/m²].

    Uses ``compute_net_surface_flux`` — the same function as the coupled
    slab ocean stepper — for self-consistent Q-flux diagnosis.
    """
    planet = forcing.planet
    levels = forcing.levels
    cfg = forcing.config
    rad = cfg.radiation
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

    # SW surface flux: consistent with atmospheric absorption
    effective_albedo = surface_albedo if surface_albedo is not None else planet.surface_albedo
    n_lat, n_lon = surface_pressure.shape

    if isinstance(rad, SpeedyRadiation) and state.humidity is not None:
        q_grid_sp = jnp.maximum(
            jax.vmap(transform.spectral_to_grid)(state.humidity),
            0.0,
        )
        # Cloud diagnosis for SPEEDY
        sp_cloud = None
        if rad.clouds is not None:
            pressure = levels.sigma_full[:, None, None] * surface_pressure[None, :, :]
            q_sat_sp = saturation_specific_humidity(t_grid, pressure, planet.epsilon_moisture)
            rh_sp = q_grid_sp / jnp.maximum(q_sat_sp, 1e-10)
            geop = planet.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t_grid)
            sp_cloud = diagnose_clouds(
                rh_sp,
                q_grid_sp,
                t_grid,
                geop,
                precipitation_rate=jnp.zeros((n_lat, n_lon)),
                convective_mask=jnp.zeros_like(t_grid, dtype=bool),
                gravity=planet.gravity,
                specific_heat_cp=planet.specific_heat_cp,
                config=rad.clouds,
            )
        insol = planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
        _, sw_down_sfc = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q_grid_sp,
            surface_pressure,
            planet.reference_pressure,
            insol,
            planet.gravity,
            planet.specific_heat_cp,
            surface_albedo=effective_albedo,
            absdry=rad.absdry,
            absaer=rad.absaer,
            abswv1=rad.sw_abswv1,
            abswv2=rad.sw_abswv2,
            visible_fraction=rad.visible_fraction,
            cloud=sp_cloud,
        )
    elif isinstance(rad, (FriersonRadiation, ByrneRadiation)) and rad.sw_tau_0 > 0.0:
        tau_sw: jnp.ndarray | None = None
        if isinstance(rad, ByrneRadiation) and state.humidity is not None:
            q_grid_sw = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            tau_sw = byrne_shortwave_optical_depth(
                levels.dsigma,
                q_grid_sw,
                surface_pressure,
                planet.reference_pressure,
                sw_tau_0=rad.sw_tau_0,
                byrne_sw_a=rad.sw_a,
                byrne_sw_b=rad.sw_b,
            )
        _, sw_down_sfc = shortwave_heating(
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
            surface_albedo=effective_albedo,
        )
    else:
        sw_down_sfc = jnp.zeros((n_lat, n_lon))

    # Downward LW
    if isinstance(rad, SpeedyRadiation) and state.humidity is not None:
        q_grid_lw = jnp.maximum(
            jax.vmap(transform.spectral_to_grid)(state.humidity),
            0.0,
        )
        lw_down = speedy_lw_down_surface(
            t_grid,
            forcing.prescribed_sst,
            q_grid_lw,
            levels.dsigma,
            surface_pressure,
            planet.reference_pressure,
            epslw=rad.epslw,
            surface_emissivity=rad.surface_emissivity,
            ablwin=rad.ablwin,
            ablco2=rad.ablco2,
            ablwv1=rad.ablwv1,
            ablwv2=rad.ablwv2,
            cloud=sp_cloud,
        )
    elif isinstance(rad, ByrneRadiation) and state.humidity is not None:
        q_grid = jnp.maximum(
            jax.vmap(transform.spectral_to_grid)(state.humidity),
            0.0,
        )
        tau_half = byrne_longwave_optical_depth(
            levels.dsigma,
            q_grid,
            surface_pressure,
            planet.reference_pressure,
            byrne_a=rad.a,
            byrne_b=rad.b,
        )
        lw_down = lw_down_surface(t_grid, tau_half)
    else:
        fri = rad if isinstance(rad, FriersonRadiation) else FriersonRadiation()
        tau_half = longwave_optical_depth(
            levels.sigma_half,
            sin_lat,
            tau_equator=fri.tau_equator,
            tau_pole=fri.tau_pole,
            linear_fraction=fri.linear_fraction,
            alpha=fri.alpha,
        )
        lw_down = lw_down_surface(t_grid, tau_half)

    # Humidity at lowest level
    q_lowest = jnp.zeros_like(t_lowest)
    if state.humidity is not None:
        q_lowest = jnp.maximum(
            transform.spectral_to_grid(state.humidity[lowest]),
            0.0,
        )

    # Net flux (same function as coupled stepper)
    net_flux = compute_net_surface_flux(
        forcing.prescribed_sst,
        t_lowest,
        q_lowest,
        wind_speed,
        surface_pressure,
        sw_down_sfc,
        lw_down,
        gravity=planet.gravity,
        gas_constant=planet.gas_constant,
        specific_heat_cp=planet.specific_heat_cp,
        epsilon=planet.epsilon_moisture,
        latent_heat=planet.latent_heat_vaporization,
        drag_coefficient=cfg.c_d,
        surface_albedo=effective_albedo,
    )

    return jnp.mean(net_flux, axis=-1)


def spinup_prescribed_sst(
    state: PrimitiveEquationState,
    forcing: PhysicsSuite,
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
    surface_albedo: float | None = None,
    verbose: bool = True,
) -> SpinupResult:
    """Spin up the atmosphere under prescribed SST and diagnose Q-flux.

    Runs a prescribed-SST integration for ``spinup_days + averaging_days``,
    then returns the spun-up atmospheric state and diagnosed Q-flux.

    Parameters
    ----------
    state : PrimitiveEquationState
        Initial atmospheric state (can be cold isothermal).
    forcing : PhysicsSuite
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
    surface_albedo : float or None
        Surface albedo for the Q-flux diagnosis.  When ``None``
        (default), uses ``planet.surface_albedo``.  Set this explicitly
        to match the albedo used by the coupled stepper (e.g. 0.06 for
        an aquaplanet ocean) so the diagnosed Q-flux is self-consistent.
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

    def one_day(
        carry: tuple[PrimitiveEquationState, PrimitiveEquationState],
        _: None,
    ) -> tuple[tuple[PrimitiveEquationState, PrimitiveEquationState], None]:
        prev, curr = carry

        def step(
            carry: tuple[PrimitiveEquationState, PrimitiveEquationState],
            _: None,
        ) -> tuple[tuple[PrimitiveEquationState, PrimitiveEquationState], None]:
            p, c = carry
            p, c = step_fn(p, c)
            return (p, c), None

        (prev, curr), _ = jax.lax.scan(step, (prev, curr), None, length=steps_per_day)
        return (prev, curr), None

    one_day_jit = jax.jit(one_day)
    diagnose_jit = jax.jit(
        lambda s, ps: _diagnose_surface_flux(s, transform, forcing, ps, surface_albedo),
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

        log_interval = 50
        if verbose and (day <= log_interval // 5 or day % log_interval == 0 or day == n_days):
            t_grid = np.asarray(
                jax.vmap(transform.spectral_to_grid)(curr.temperature),
            )
            phase = "spinup" if day <= spinup_days else "averaging"
            logger.info(
                "  Day %5d [%9s]: T_mean=%.1f K",
                day,
                phase,
                np.mean(t_grid),
            )

    mean_flux = flux_accum / n_samples
    q_flux = jnp.array(-mean_flux)

    if verbose:
        lat_deg = np.degrees(np.asarray(transform.grid.latitudes))
        eq_idx = np.argmin(np.abs(lat_deg))
        logger.info("  Q-flux(equator) = %+.1f W/m^2", float(q_flux[eq_idx]))
        logger.info(
            "  Q-flux range = [%.1f, %.1f] W/m^2",
            float(jnp.min(q_flux)),
            float(jnp.max(q_flux)),
        )

    return SpinupResult(
        state=spinup_state,
        q_flux=q_flux,
        mean_net_flux=jnp.array(mean_flux),
        n_samples=n_samples,
    )
