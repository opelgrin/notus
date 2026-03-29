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
from notus.physics.physics_suite import PhysicsSuite
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
    cfg = forcing.config
    lowest = forcing.levels.n_levels - 1

    effective_albedo = surface_albedo if surface_albedo is not None else planet.surface_albedo

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

    # Radiation surface fluxes via PhysicsSuite methods
    cloud = forcing.compute_clouds(state, surface_pressure)
    sw_down_sfc = forcing.compute_sw_down_surface(
        state,
        surface_pressure,
        effective_albedo=effective_albedo,
        cloud=cloud,
    )
    lw_down = forcing.compute_lw_down_surface(state, surface_pressure, cloud=cloud)

    # Humidity at lowest level
    t_lowest = transform.spectral_to_grid(state.temperature[lowest])
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
