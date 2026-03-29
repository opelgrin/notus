"""Coupled atmosphere-ocean-land time stepping.

Wraps the atmospheric IMEX stepper with slab ocean thermodynamics and
optional bucket land surface model.  The core IMEX machinery
(euler_init, imex_leapfrog_step) is reused unchanged — this module only
adds the surface state threading and coupled implicit treatment.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators.vector import uv_from_vordiv
from notus.physics.boundary_layer import SurfaceLayerConfig, compute_transfer_coefficients
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.physics_suite import PhysicsSuite
from notus.physics.surface import (
    BucketLandConfig,
    SlabOceanConfig,
    SurfaceState,
    beta_function,
    compute_net_land_flux,
    compute_net_surface_flux,
    diagnose_precipitation,
    land_flux_derivative,
    step_bucket_hydrology,
    step_land_implicit,
    step_slab_ocean_implicit,
    surface_flux_derivative,
)
from notus.physics.surface_types import SurfaceProperties, moisture_dependent_albedo
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

    def __init__(self, forcing: PhysicsSuite) -> None:
        self._forcing = forcing
        self.compute_reference_humidity = forcing.compute_reference_humidity

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        return self._forcing(state, surface_pressure)


def build_coupled_pe_stepper(  # noqa: C901, PLR0915
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    reference_temperature: np.ndarray,
    surface_geopotential: jnp.ndarray,
    dt: float,
    forcing: PhysicsSuite,
    ocean_config: SlabOceanConfig,
    q_flux: jnp.ndarray,
    surface_properties: SurfaceProperties | None = None,
    land_config: BucketLandConfig | None = None,
    spectral_filter: jnp.ndarray | None = None,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
    robert_coeff: float = 0.05,
    alpha: float = 0.5,
) -> tuple[
    Callable[
        [PrimitiveEquationState, SurfaceState],
        tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState],
    ],
    Callable[
        [PrimitiveEquationState, PrimitiveEquationState, SurfaceState],
        tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState],
    ],
]:
    """Build init and step functions for coupled atmosphere-surface integration.

    Wraps ``build_pe_stepper`` and adds slab ocean thermodynamics and
    optional bucket land surface model.  The atmospheric IMEX step runs
    first (without implicit physics), then the coupled post-step updates
    ocean SST, land soil temperature / bucket hydrology, and the implicit
    atmospheric surface treatment together.

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
    forcing : PhysicsSuite
        Physics forcing (provides radiation, SST, implicit surface
        physics, and reference humidity).
    ocean_config : SlabOceanConfig
        Slab ocean parameters.
    q_flux : jnp.ndarray
        Prescribed ocean heat transport [W/m²], shape ``(n_lat,)``.
    surface_properties : SurfaceProperties or None
        Spatially varying albedo and roughness.  Required when
        ``land_config`` is not None.
    land_config : BucketLandConfig or None
        Bucket land model parameters.  If ``None``, pure ocean mode.
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
        ``init_fn(state, surface) -> (previous, current, surface)``
        ``step_fn(previous, current, surface) -> (filtered_current, future, surface)``
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

    # Pre-extract build-time constants
    lowest = levels.n_levels - 1
    dsigma_lowest = float(np.asarray(levels.dsigma)[-1])
    cfg = forcing.config
    c_d = cfg.c_d
    surface_layer_cfg: SurfaceLayerConfig | None = cfg.surface_layer
    ocean_heat_capacity = ocean_config.heat_capacity
    sigma_lowest_val = 1.0 - 0.5 * dsigma_lowest

    # Surface properties
    sfc_albedo = (
        surface_properties.albedo if surface_properties is not None else planet.surface_albedo
    )
    has_land = land_config is not None and surface_properties is not None
    land_frac = surface_properties.land_fraction if surface_properties is not None else None

    # Physics config for precipitation diagnostic (land branch only)
    tau_bm = cfg.tau_bm
    rh_ref = cfg.rh_ref
    n_cond_iter = cfg.n_condensation_iterations
    rh_cond = cfg.rh_condensation
    tau_adj = cfg.tau_adjustment

    # ------------------------------------------------------------------
    # Helper closures
    # ------------------------------------------------------------------

    def _apply_rayleigh_friction(
        state: PrimitiveEquationState,
        dt_implicit: float,
    ) -> PrimitiveEquationState:
        """Apply Rayleigh friction via exact exponential decay."""
        damp = jnp.exp(-dt_implicit * forcing.k_v[:, None])
        return state.replace(
            vorticity=state.vorticity * damp,
            divergence=state.divergence * damp,
        )

    def _surface_state(
        state: PrimitiveEquationState,
        surface: SurfaceState,
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float | jnp.ndarray
    ]:
        """Extract surface winds, pressure, temperature, humidity, exchange coeff, and drag."""
        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[lowest],
            state.divergence[lowest],
            transform.arrays,
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

        # Surface temperature for MO: blended if land is present
        sst = surface.ocean.surface_temperature
        sst_bc = sst[:, None] if sst.ndim == 1 else sst
        if has_land and surface.land is not None and land_frac is not None:
            t_sfc_2d = (1.0 - land_frac) * sst_bc + land_frac * surface.land.soil_temperature
        else:
            t_sfc_2d = sst_bc * jnp.ones_like(t_lowest_grid)

        # Transfer coefficient: MO stability-dependent or constant
        c_h: float | jnp.ndarray
        if surface_layer_cfg is not None:
            z0_m_arr = surface_properties.z0_momentum if surface_properties is not None else None
            z0_h_arr = surface_properties.z0_heat if surface_properties is not None else None
            _c_d_m, c_h = compute_transfer_coefficients(
                t_sfc_2d,
                t_lowest_grid,
                wind_speed,
                dsigma_lowest,
                planet.gravity,
                planet.gas_constant,
                surface_layer_cfg,
                z0_momentum_override=z0_m_arr,
                z0_heat_override=z0_h_arr,
            )
        else:
            c_h = c_d

        k_sfc = planet.gravity * rho_sfc * c_h * wind_speed / dp_safe

        q_lowest = jnp.zeros_like(t_lowest_grid)
        if state.humidity is not None:
            q_lowest = jnp.maximum(
                transform.spectral_to_grid(state.humidity[lowest]),
                0.0,
            )

        return wind_speed, ps_grid, t_lowest_grid, q_lowest, k_sfc, c_h

    def _compute_effective_albedo(surface: SurfaceState) -> float | jnp.ndarray:
        """Compute surface albedo (moisture-dependent over land, or constant)."""
        if (
            has_land
            and land_config is not None
            and land_config.moisture_dependent_albedo
            and surface.land is not None
            and land_frac is not None
        ):
            ocean_alb = (
                surface_properties.albedo
                if surface_properties is not None
                else planet.surface_albedo
            )
            return moisture_dependent_albedo(
                surface.land.bucket_depth,
                land_config.bucket_capacity,
                land_config.albedo_dry,
                land_config.albedo_wet,
                land_frac,
                ocean_alb,
            )
        return sfc_albedo

    def _apply_implicit_decay(
        state: PrimitiveEquationState,
        t_target: jnp.ndarray,
        q_target: jnp.ndarray,
        t_lowest_grid: jnp.ndarray,
        ps_grid: jnp.ndarray,
        k_sfc: jnp.ndarray,
        dt_implicit: float,
    ) -> PrimitiveEquationState:
        """Implicit atmospheric decay toward surface temperature/humidity targets."""
        decay_sfc = jnp.exp(-dt_implicit * k_sfc)
        t_corrected = t_target + (t_lowest_grid - t_target) * decay_sfc
        new_temp = state.temperature.at[lowest].set(
            transform.grid_to_spectral(t_corrected),
        )

        new_humidity = state.humidity
        if state.humidity is not None:
            q_lowest_grid = transform.spectral_to_grid(state.humidity[lowest])
            q_corrected = q_target + (q_lowest_grid - q_target) * decay_sfc
            new_humidity = state.humidity.at[lowest].set(
                transform.grid_to_spectral(q_corrected),
            )

        return PrimitiveEquationState(
            vorticity=state.vorticity,
            divergence=state.divergence,
            temperature=new_temp,
            log_surface_pressure=state.log_surface_pressure,
            humidity=new_humidity,
        )

    # ------------------------------------------------------------------
    # Unified coupled post-step
    # ------------------------------------------------------------------

    def _coupled_post_step(
        state: PrimitiveEquationState,
        surface: SurfaceState,
        dt_implicit: float,
    ) -> tuple[PrimitiveEquationState, SurfaceState]:
        """Post-step: update surface + implicit atmospheric decay."""
        # Rayleigh friction
        state = _apply_rayleigh_friction(state, dt_implicit)

        # Surface state extraction
        wind_speed, ps_grid, t_lowest_grid, q_lowest, k_sfc, c_h = _surface_state(
            state,
            surface,
        )

        # Radiation surface fluxes (delegated to PhysicsSuite)
        effective_albedo = _compute_effective_albedo(surface)
        cloud = forcing.compute_clouds(state, ps_grid)
        sw_down_sfc = forcing.compute_sw_down_surface(
            state,
            ps_grid,
            effective_albedo=effective_albedo,
            day_of_year=forcing.day_of_year,
            cloud=cloud,
        )
        lw_down = forcing.compute_lw_down_surface(
            state,
            ps_grid,
            sst=surface.ocean.surface_temperature,
            cloud=cloud,
        )

        # --- Ocean SST update ---
        ocean = surface.ocean
        ocean_net_flux = compute_net_surface_flux(
            ocean.surface_temperature,
            t_lowest_grid,
            q_lowest,
            wind_speed,
            ps_grid,
            sw_down_sfc,
            lw_down,
            gravity=planet.gravity,
            gas_constant=planet.gas_constant,
            specific_heat_cp=planet.specific_heat_cp,
            epsilon=planet.epsilon_moisture,
            latent_heat=planet.latent_heat_vaporization,
            drag_coefficient=c_h,
            surface_albedo=effective_albedo,
        )
        ocean_dflux = surface_flux_derivative(
            ocean.surface_temperature,
            wind_speed,
            ps_grid,
            gas_constant=planet.gas_constant,
            specific_heat_cp=planet.specific_heat_cp,
            epsilon=planet.epsilon_moisture,
            latent_heat=planet.latent_heat_vaporization,
            drag_coefficient=c_h,
        )
        ocean = step_slab_ocean_implicit(
            ocean,
            jnp.mean(ocean_net_flux, axis=-1),
            jnp.mean(ocean_dflux, axis=-1),
            q_flux,
            ocean_heat_capacity,
            dt_implicit,
        )

        # --- Land update (conditional) ---
        land = surface.land
        if has_land and land_config is not None and land is not None and land_frac is not None:
            lc = land_config
            lf = land_frac
            beta = beta_function(land.bucket_depth, lc.w_crit)

            land_net_flux, land_evap = compute_net_land_flux(
                land.soil_temperature,
                t_lowest_grid,
                q_lowest,
                wind_speed,
                ps_grid,
                sw_down_sfc,
                lw_down,
                beta,
                gravity=planet.gravity,
                gas_constant=planet.gas_constant,
                specific_heat_cp=planet.specific_heat_cp,
                epsilon=planet.epsilon_moisture,
                latent_heat=planet.latent_heat_vaporization,
                drag_coefficient=c_h,
                surface_albedo=effective_albedo,
            )
            land_dflux = land_flux_derivative(
                land.soil_temperature,
                wind_speed,
                ps_grid,
                beta,
                gas_constant=planet.gas_constant,
                specific_heat_cp=planet.specific_heat_cp,
                epsilon=planet.epsilon_moisture,
                latent_heat=planet.latent_heat_vaporization,
                drag_coefficient=c_h,
            )
            land = step_land_implicit(
                land,
                land_net_flux,
                land_dflux,
                lc.soil_heat_capacity,
                dt_implicit,
            )

            # Precipitation diagnostic for bucket hydrology
            t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
            q_grid: jnp.ndarray | None = None
            if state.humidity is not None:
                q_grid = jnp.maximum(
                    jax.vmap(transform.spectral_to_grid)(state.humidity),
                    0.0,
                )
            if q_grid is not None:
                sigma = levels.sigma_full[:, None, None]
                pressure = sigma * ps_grid[None, :, :]
                precip = diagnose_precipitation(
                    t_grid,
                    q_grid,
                    pressure,
                    levels.dsigma,
                    ps_grid,
                    gravity=planet.gravity,
                    epsilon=planet.epsilon_moisture,
                    latent_heat=planet.latent_heat_vaporization,
                    specific_heat_cp=planet.specific_heat_cp,
                    gas_constant=planet.gas_constant,
                    tau_bm=tau_bm,
                    rh_ref=rh_ref,
                    n_condensation_iterations=n_cond_iter,
                    rh_condensation=rh_cond,
                    tau_adjustment=tau_adj,
                )
            else:
                precip = jnp.zeros_like(land.soil_temperature)

            land = step_bucket_hydrology(
                land,
                precip,
                land_evap,
                lc.bucket_capacity,
                dt_implicit,
            )

        # --- Implicit atmospheric decay ---
        new_sst = ocean.surface_temperature
        sst_bc = new_sst[:, None] if new_sst.ndim == 1 else new_sst

        if has_land and land is not None and land_config is not None and land_frac is not None:
            lf = land_frac
            t_land = land.soil_temperature
            t_target = (1.0 - lf) * sst_bc + lf * t_land
            q_sat_ocean = saturation_specific_humidity(
                sst_bc,
                ps_grid,
                planet.epsilon_moisture,
            )
            beta_new = beta_function(land.bucket_depth, land_config.w_crit)
            q_sat_land = saturation_specific_humidity(
                t_land,
                ps_grid,
                planet.epsilon_moisture,
            )
            q_target = (1.0 - lf) * q_sat_ocean + lf * beta_new * q_sat_land
        else:
            t_target = sst_bc * jnp.ones_like(t_lowest_grid)
            q_target = saturation_specific_humidity(
                sst_bc,
                ps_grid,
                planet.epsilon_moisture,
            )

        state = _apply_implicit_decay(
            state,
            t_target,
            q_target,
            t_lowest_grid,
            ps_grid,
            k_sfc,
            dt_implicit,
        )

        return state, SurfaceState(ocean=ocean, land=land)

    # ------------------------------------------------------------------
    # Public init / step functions
    # ------------------------------------------------------------------

    @jax.jit
    def init_fn(
        state: PrimitiveEquationState,
        surface: SurfaceState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState]:
        previous, current = atm_init_fn(state)
        current, surface = _coupled_post_step(current, surface, dt)
        return previous, current, surface

    @jax.jit
    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
        surface: SurfaceState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState]:
        filtered_current, future = atm_step_fn(previous, current)
        future, surface = _coupled_post_step(future, surface, 2.0 * dt)
        return filtered_current, future, surface

    return init_fn, step_fn
