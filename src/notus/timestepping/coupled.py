"""Coupled atmosphere-ocean time stepping.

Wraps the atmospheric IMEX stepper with slab ocean thermodynamics.
The core IMEX machinery (euler_init, imex_leapfrog_step) is reused
unchanged — this module only adds the ocean state threading and
coupled implicit surface treatment.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators.vector import uv_from_vordiv
from notus.physics.boundary_layer import SurfaceLayerConfig, compute_transfer_coefficients
from notus.physics.forcing import Forcing
from notus.physics.surface_types import SurfaceProperties
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import (
    byrne_longwave_optical_depth,
    longwave_optical_depth,
    lw_down_surface,
)
from notus.physics.solar import daily_mean_insolation
from notus.physics.surface import (
    OceanState,
    SlabOceanConfig,
    compute_net_surface_flux,
    step_slab_ocean_implicit,
    surface_flux_derivative,
)
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


class _ExplicitOnlyForcing:
    """Wrapper that suppresses ``apply_implicit`` from a forcing.

    This is used by ``build_coupled_pe_stepper`` so that the core IMEX
    stepper does not auto-detect and apply implicit physics — the coupled
    stepper handles it with the ocean update instead.
    """

    def __init__(self, forcing: Forcing) -> None:
        self._forcing = forcing
        # Forward compute_reference_humidity if present
        if hasattr(forcing, "compute_reference_humidity"):
            self.compute_reference_humidity = forcing.compute_reference_humidity

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        return self._forcing(state, surface_pressure)


def build_coupled_pe_stepper(  # noqa: PLR0915
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    reference_temperature: np.ndarray,
    surface_geopotential: jnp.ndarray,
    dt: float,
    forcing: Forcing,
    ocean_config: SlabOceanConfig,
    q_flux: jnp.ndarray,
    surface_properties: SurfaceProperties | None = None,
    spectral_filter: jnp.ndarray | None = None,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
    robert_coeff: float = 0.05,
    alpha: float = 0.5,
) -> tuple[
    Callable[
        [PrimitiveEquationState, OceanState],
        tuple[PrimitiveEquationState, PrimitiveEquationState, OceanState],
    ],
    Callable[
        [PrimitiveEquationState, PrimitiveEquationState, OceanState],
        tuple[PrimitiveEquationState, PrimitiveEquationState, OceanState],
    ],
]:
    """Build init and step functions for coupled atmosphere-ocean integration.

    Wraps ``build_pe_stepper`` and adds slab ocean thermodynamics.
    The atmospheric IMEX step runs first (without implicit physics),
    then the coupled post-step updates both the ocean SST and the
    implicit atmospheric surface treatment together.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    reference_temperature : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential in spectral space.
    dt : float
        Timestep [s].
    forcing : Forcing
        Physics forcing (must expose ``apply_implicit`` and
        ``compute_reference_humidity`` for full functionality).
    ocean_config : SlabOceanConfig
        Slab ocean parameters.
    q_flux : jnp.ndarray
        Prescribed ocean heat transport [W/m²], shape ``(n_lat,)``.
    spectral_filter : jnp.ndarray or None
        Spectral filter array.
    diffusion_order : int
        Hyperdiffusion order.
    diffusion_timescale : float
        Hyperdiffusion timescale [s].
    robert_coeff : float
        Robert-Asselin filter coefficient.
    alpha : float
        Implicit weighting.

    Returns
    -------
    tuple[init_fn, step_fn]
        ``init_fn(state, ocean) -> (previous, current, ocean)``
        ``step_fn(previous, current, ocean) -> (filtered_current, future, ocean)``
    """
    # Build atmospheric stepper WITHOUT implicit physics
    explicit_forcing = _ExplicitOnlyForcing(forcing)
    atm_init_fn, atm_step_fn = build_pe_stepper(
        transform=transform,
        planet=planet,
        levels=levels,
        reference_temperature=reference_temperature,
        surface_geopotential=surface_geopotential,
        dt=dt,
        forcing=explicit_forcing,
        spectral_filter=spectral_filter,
        diffusion_order=diffusion_order,
        diffusion_timescale=diffusion_timescale,
        robert_coeff=robert_coeff,
        alpha=alpha,
    )

    lowest = levels.n_levels - 1
    dsigma_lowest = float(np.asarray(levels.dsigma)[-1])
    cfg = forcing.config if hasattr(forcing, "config") else None
    c_d = cfg.c_d if cfg is not None else 0.0015
    surface_layer_cfg: SurfaceLayerConfig | None = (
        cfg.surface_layer if cfg is not None and hasattr(cfg, "surface_layer") else None
    )
    sw_tau_0 = cfg.sw_tau_0 if cfg is not None else 0.0
    radiation_scheme = cfg.radiation_scheme if cfg is not None else "frierson"
    lw_tau_equator = cfg.tau_equator if cfg is not None else 6.0
    lw_tau_pole = cfg.tau_pole if cfg is not None else 0.1
    lw_linear_fraction = cfg.linear_fraction if cfg is not None else 0.1
    lw_alpha = cfg.alpha if cfg is not None else 4.0
    lw_byrne_a = cfg.byrne_a if cfg is not None else 0.8678
    lw_byrne_b = cfg.byrne_b if cfg is not None else 1997.9
    heat_capacity = ocean_config.heat_capacity
    sigma_lowest_val = 1.0 - 0.5 * dsigma_lowest
    # Surface properties: spatially varying albedo/roughness, or scalar defaults
    sfc_albedo = surface_properties.albedo if surface_properties is not None else planet.surface_albedo

    def _compute_insolation() -> jnp.ndarray:
        """Compute TOA insolation (seasonal or fixed)."""
        sin_lat = transform.grid.sin_lat
        if (
            hasattr(forcing, "day_of_year")
            and forcing.day_of_year is not None
            and cfg is not None
            and cfg.orbital is not None
        ):
            return daily_mean_insolation(
                sin_lat, forcing.day_of_year, planet.solar_constant, cfg.orbital,
            )
        delta_s = cfg.delta_s if cfg is not None else 1.4
        return planet.solar_constant / 4.0 * (
            1.0 + delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0
        )

    def _surface_state(
        state: PrimitiveEquationState,
        ocean: OceanState,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Extract surface winds, pressure, temperature, humidity, exchange coeff, and drag."""
        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[lowest], state.divergence[lowest], transform.arrays,
        )
        cos_lat_safe = jnp.maximum(transform.grid.cos_lat[:, None], 1.0e-6)
        u_grid = transform.spectral_to_grid(u_cos_spec) / cos_lat_safe
        v_grid = transform.spectral_to_grid(v_cos_spec) / cos_lat_safe
        wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

        lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
        ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)
        t_lowest_grid = transform.spectral_to_grid(state.temperature[lowest])

        t_safe = jnp.maximum(t_lowest_grid, 1.0)
        dp_safe = jnp.maximum(dsigma_lowest * ps_grid, 1.0)
        rho_sfc = ps_grid * sigma_lowest_val / (planet.gas_constant * t_safe)

        # Transfer coefficient: MO stability-dependent or constant
        if surface_layer_cfg is not None:
            sst = ocean.surface_temperature
            sst_bc = sst[:, None] if sst.ndim == 1 else sst
            t_sfc_2d = sst_bc * jnp.ones_like(t_lowest_grid)
            _c_d_m, c_h = compute_transfer_coefficients(
                t_sfc_2d, t_lowest_grid, wind_speed,
                dsigma_lowest, planet.gravity, planet.gas_constant,
                surface_layer_cfg,
            )
        else:
            c_h = c_d

        k_sfc = planet.gravity * rho_sfc * c_h * wind_speed / dp_safe

        q_lowest = jnp.zeros_like(t_lowest_grid)
        if state.humidity is not None:
            q_lowest = jnp.maximum(
                transform.spectral_to_grid(state.humidity[lowest]), 0.0,
            )

        return wind_speed, ps_grid, t_lowest_grid, q_lowest, k_sfc, c_h

    def _coupled_post_step(
        state: PrimitiveEquationState,
        ocean: OceanState,
        dt_implicit: float,
    ) -> tuple[PrimitiveEquationState, OceanState]:
        """Apply coupled implicit physics and update ocean SST."""
        # --- Rayleigh friction (exact exponential decay) ---
        if hasattr(forcing, "k_v"):
            damp = jnp.exp(-dt_implicit * forcing.k_v[:, None])
            state = state.replace(
                vorticity=state.vorticity * damp,
                divergence=state.divergence * damp,
            )

        wind_speed, ps_grid, t_lowest_grid, q_lowest, k_sfc, c_h = _surface_state(state, ocean)
        insolation = _compute_insolation()

        # Downward LW flux at surface from the two-stream radiation solver
        t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
        if radiation_scheme == "byrne" and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity), 0.0,
            )
            tau_half = byrne_longwave_optical_depth(
                levels.dsigma, q_grid, ps_grid, planet.reference_pressure,
                byrne_a=lw_byrne_a, byrne_b=lw_byrne_b,
            )
        else:
            tau_half = longwave_optical_depth(
                levels.sigma_half, transform.grid.sin_lat,
                tau_equator=lw_tau_equator, tau_pole=lw_tau_pole,
                linear_fraction=lw_linear_fraction, alpha=lw_alpha,
            )
        lw_down = lw_down_surface(t_grid, tau_half)

        net_flux = compute_net_surface_flux(
            ocean.surface_temperature, t_lowest_grid, q_lowest,
            wind_speed, ps_grid, insolation, lw_down,
            gravity=planet.gravity, gas_constant=planet.gas_constant,
            specific_heat_cp=planet.specific_heat_cp,
            epsilon=planet.epsilon_moisture,
            latent_heat=planet.latent_heat_vaporization,
            drag_coefficient=c_h, surface_albedo=sfc_albedo,
            sw_tau_0=sw_tau_0,
        )

        # Derivative of net flux w.r.t. SST (for implicit step)
        dflux_dt = surface_flux_derivative(
            ocean.surface_temperature, wind_speed, ps_grid,
            gas_constant=planet.gas_constant,
            specific_heat_cp=planet.specific_heat_cp,
            epsilon=planet.epsilon_moisture,
            latent_heat=planet.latent_heat_vaporization,
            drag_coefficient=c_h,
        )

        # Update ocean SST (linearized implicit, zonal-mean for 1-D SST)
        ocean = step_slab_ocean_implicit(
            ocean,
            jnp.mean(net_flux, axis=-1),
            jnp.mean(dflux_dt, axis=-1),
            q_flux,
            heat_capacity,
            dt_implicit,
        )
        new_sst = ocean.surface_temperature

        # Implicit atmospheric decay toward new SST
        decay_sfc = jnp.exp(-dt_implicit * k_sfc)
        sst_bc = new_sst[:, None] if new_sst.ndim == 1 else new_sst
        t_corrected = sst_bc + (t_lowest_grid - sst_bc) * decay_sfc
        new_temp = state.temperature.at[lowest].set(
            transform.grid_to_spectral(t_corrected),
        )

        # Implicit latent heat flux toward q_sat(new SST)
        new_humidity = state.humidity
        if state.humidity is not None:
            q_lowest_grid = transform.spectral_to_grid(state.humidity[lowest])
            q_sat_sfc = saturation_specific_humidity(sst_bc, ps_grid, planet.epsilon_moisture)
            q_corrected = q_sat_sfc + (q_lowest_grid - q_sat_sfc) * decay_sfc
            new_humidity = state.humidity.at[lowest].set(
                transform.grid_to_spectral(q_corrected),
            )

        state = PrimitiveEquationState(
            vorticity=state.vorticity,
            divergence=state.divergence,
            temperature=new_temp,
            log_surface_pressure=state.log_surface_pressure,
            humidity=new_humidity,
        )

        return state, ocean

    @jax.jit
    def init_fn(
        state: PrimitiveEquationState,
        ocean: OceanState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, OceanState]:
        previous, current = atm_init_fn(state)
        current, ocean = _coupled_post_step(current, ocean, dt)
        return previous, current, ocean

    @jax.jit
    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
        ocean: OceanState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, OceanState]:
        filtered_current, future = atm_step_fn(previous, current)
        future, ocean = _coupled_post_step(future, ocean, 2.0 * dt)
        return filtered_current, future, ocean

    return init_fn, step_fn
