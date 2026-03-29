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
from notus.physics.clouds import CloudDiagnostic, diagnose_clouds
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import (
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    longwave_optical_depth,
    lw_down_surface,
    shortwave_heating,
    speedy_lw_down_surface,
    speedy_shortwave_heating,
)
from notus.physics.simple_physics import SimplePhysics
from notus.physics.solar import daily_mean_insolation
from notus.physics.surface import (
    BucketLandConfig,
    LandState,
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

    def __init__(self, forcing: SimplePhysics) -> None:
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
    forcing: SimplePhysics,
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
    forcing : SimplePhysics
        Physics forcing (provides radiation config, SST, implicit
        surface physics, and reference humidity).
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

    lowest = levels.n_levels - 1
    dsigma_lowest = float(np.asarray(levels.dsigma)[-1])
    cfg = forcing.config
    c_d = cfg.c_d
    surface_layer_cfg: SurfaceLayerConfig | None = cfg.surface_layer
    sw_tau_0 = cfg.sw_tau_0
    sw_exponent = cfg.sw_exponent
    delta_s = cfg.delta_s
    radiation_scheme = cfg.radiation_scheme
    lw_tau_equator = cfg.tau_equator
    lw_tau_pole = cfg.tau_pole
    lw_linear_fraction = cfg.linear_fraction
    lw_alpha = cfg.alpha
    lw_byrne_a = cfg.byrne_a
    lw_byrne_b = cfg.byrne_b
    byrne_sw_a = cfg.byrne_sw_a
    byrne_sw_b = cfg.byrne_sw_b
    sp_epslw = cfg.speedy_epslw
    sp_emisfc = cfg.speedy_surface_emissivity
    sp_ablwin = cfg.speedy_ablwin
    sp_ablco2 = cfg.speedy_ablco2
    sp_ablwv1 = cfg.speedy_ablwv1
    sp_ablwv2 = cfg.speedy_ablwv2
    sp_absdry = cfg.speedy_absdry
    sp_absaer = cfg.speedy_absaer
    sp_sw_abswv1 = cfg.speedy_sw_abswv1
    sp_sw_abswv2 = cfg.speedy_sw_abswv2
    sp_vis_frac = cfg.speedy_visible_fraction
    clouds_enabled = cfg.enable_clouds
    cloud_cfg = cfg.cloud_config
    ocean_heat_capacity = ocean_config.heat_capacity
    sigma_lowest_val = 1.0 - 0.5 * dsigma_lowest

    # Surface properties: spatially varying albedo/roughness, or scalar defaults
    sfc_albedo = (
        surface_properties.albedo if surface_properties is not None else planet.surface_albedo
    )
    has_land = land_config is not None and surface_properties is not None
    land_frac = surface_properties.land_fraction if surface_properties is not None else None

    # Pre-extract physics config for precipitation diagnostic
    tau_bm = cfg.tau_bm if cfg is not None else 7200.0
    rh_ref = cfg.rh_ref if cfg is not None else 0.7
    n_cond_iter = cfg.n_condensation_iterations if cfg is not None else 3
    rh_cond = cfg.rh_condensation if cfg is not None else 1.0
    tau_adj = cfg.tau_adjustment if cfg is not None else 43200.0

    def _compute_insolation() -> jnp.ndarray:
        """Compute TOA insolation (seasonal or fixed Frierson profile)."""
        sin_lat = transform.grid.sin_lat
        if forcing.day_of_year is not None and cfg.orbital is not None:
            return daily_mean_insolation(
                sin_lat,
                forcing.day_of_year,
                planet.solar_constant,
                cfg.orbital,
            )
        return planet.solar_constant / 4.0 * (1.0 + delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)

    def _diagnose_speedy_clouds(
        state: PrimitiveEquationState,
        t_grid: jnp.ndarray,
        q_grid: jnp.ndarray,
        ps_grid: jnp.ndarray,
    ) -> CloudDiagnostic | None:
        """Diagnose clouds for SPEEDY scheme (returns None if disabled)."""
        if not clouds_enabled:
            return None
        pressure = levels.sigma_full[:, None, None] * ps_grid[None, :, :]
        q_sat = saturation_specific_humidity(t_grid, pressure, planet.epsilon_moisture)
        rh = q_grid / jnp.maximum(q_sat, 1e-10)
        geopotential = planet.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t_grid)
        n_lat, n_lon = ps_grid.shape
        return diagnose_clouds(
            rh,
            q_grid,
            t_grid,
            geopotential,
            precipitation_rate=jnp.zeros((n_lat, n_lon)),
            convective_mask=jnp.zeros_like(t_grid, dtype=bool),
            gravity=planet.gravity,
            specific_heat_cp=planet.specific_heat_cp,
            config=cloud_cfg,
        )

    def _compute_sw_down_surface(
        state: PrimitiveEquationState,
        ps_grid: jnp.ndarray,
        effective_albedo: float | jnp.ndarray,
        cloud: CloudDiagnostic | None = None,
    ) -> jnp.ndarray:
        """Compute SW flux reaching the surface [W/m²]."""
        sin_lat = transform.grid.sin_lat

        if radiation_scheme == "speedy" and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            _, sw_down_sfc = speedy_shortwave_heating(
                levels.dsigma,
                levels.sigma_full,
                q_grid,
                ps_grid,
                planet.reference_pressure,
                _compute_insolation(),
                planet.gravity,
                planet.specific_heat_cp,
                surface_albedo=effective_albedo,
                absdry=sp_absdry,
                absaer=sp_absaer,
                abswv1=sp_sw_abswv1,
                abswv2=sp_sw_abswv2,
                visible_fraction=sp_vis_frac,
                cloud=cloud,
            )
            return sw_down_sfc

        if sw_tau_0 <= 0.0:
            n_lat, n_lon = ps_grid.shape
            return jnp.zeros((n_lat, n_lon))

        insolation = _compute_insolation()

        # Humidity-dependent SW optical depth for Byrne scheme
        tau_sw: jnp.ndarray | None = None
        if radiation_scheme == "byrne" and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            tau_sw = byrne_shortwave_optical_depth(
                levels.dsigma,
                q_grid,
                ps_grid,
                planet.reference_pressure,
                sw_tau_0=sw_tau_0,
                byrne_sw_a=byrne_sw_a,
                byrne_sw_b=byrne_sw_b,
            )

        _, sw_down_sfc = shortwave_heating(
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            ps_grid,
            planet.solar_constant,
            planet.gravity,
            planet.specific_heat_cp,
            sw_tau_0=sw_tau_0,
            sw_exponent=sw_exponent,
            delta_s=delta_s,
            insolation=insolation,
            tau_sw_half=tau_sw,
            surface_albedo=effective_albedo,
        )
        return sw_down_sfc

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
        if has_land and surface.land is not None:
            t_sfc_2d = (1.0 - land_frac) * sst_bc + land_frac * surface.land.soil_temperature  # type: ignore[operator]
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

    def _compute_lw_down(
        state: PrimitiveEquationState,
        ps_grid: jnp.ndarray,
        sst: jnp.ndarray | None = None,
        cloud: CloudDiagnostic | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Compute downward LW flux and return grid-space fields."""
        t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)
        q_grid: jnp.ndarray | None = None

        if radiation_scheme == "speedy" and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            t_sfc = sst if sst is not None else forcing.prescribed_sst
            lw_down = speedy_lw_down_surface(
                t_grid,
                t_sfc,
                q_grid,
                levels.dsigma,
                ps_grid,
                planet.reference_pressure,
                epslw=sp_epslw,
                surface_emissivity=sp_emisfc,
                ablwin=sp_ablwin,
                ablco2=sp_ablco2,
                ablwv1=sp_ablwv1,
                ablwv2=sp_ablwv2,
                cloud=cloud,
            )
            return lw_down, t_grid, q_grid

        if radiation_scheme == "byrne" and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            tau_half = byrne_longwave_optical_depth(
                levels.dsigma,
                q_grid,
                ps_grid,
                planet.reference_pressure,
                byrne_a=lw_byrne_a,
                byrne_b=lw_byrne_b,
            )
        else:
            tau_half = longwave_optical_depth(
                levels.sigma_half,
                transform.grid.sin_lat,
                tau_equator=lw_tau_equator,
                tau_pole=lw_tau_pole,
                linear_fraction=lw_linear_fraction,
                alpha=lw_alpha,
            )
        lw_down = lw_down_surface(t_grid, tau_half)
        return lw_down, t_grid, q_grid

    def _ocean_only_post_step(
        state: PrimitiveEquationState,
        surface: SurfaceState,
        dt_implicit: float,
    ) -> tuple[PrimitiveEquationState, SurfaceState]:
        """Post-step for ocean-only mode (backward compatible)."""
        # Rayleigh friction
        damp = jnp.exp(-dt_implicit * forcing.k_v[:, None])
        state = state.replace(
            vorticity=state.vorticity * damp,
            divergence=state.divergence * damp,
        )

        wind_speed, ps_grid, t_lowest_grid, q_lowest, k_sfc, c_h = _surface_state(
            state,
            surface,
        )
        # Cloud diagnosis for speedy scheme
        cloud_diag = None
        if radiation_scheme == "speedy" and clouds_enabled and state.humidity is not None:
            t_for_cloud = jax.vmap(transform.spectral_to_grid)(state.temperature)
            q_for_cloud = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            cloud_diag = _diagnose_speedy_clouds(state, t_for_cloud, q_for_cloud, ps_grid)

        sw_down_sfc = _compute_sw_down_surface(state, ps_grid, sfc_albedo, cloud=cloud_diag)
        lw_down, _t_grid, _q_grid = _compute_lw_down(state, ps_grid, cloud=cloud_diag)

        ocean = surface.ocean
        net_flux = compute_net_surface_flux(
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
            surface_albedo=sfc_albedo,
        )
        dflux_dt = surface_flux_derivative(
            ocean.surface_temperature,
            wind_speed,
            ps_grid,
            gas_constant=planet.gas_constant,
            specific_heat_cp=planet.specific_heat_cp,
            epsilon=planet.epsilon_moisture,
            latent_heat=planet.latent_heat_vaporization,
            drag_coefficient=c_h,
        )

        # Update ocean SST (zonal-mean for 1-D SST)
        ocean = step_slab_ocean_implicit(
            ocean,
            jnp.mean(net_flux, axis=-1),
            jnp.mean(dflux_dt, axis=-1),
            q_flux,
            ocean_heat_capacity,
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

        new_humidity = state.humidity
        if state.humidity is not None:
            q_lowest_grid = transform.spectral_to_grid(state.humidity[lowest])
            q_sat_sfc = saturation_specific_humidity(
                sst_bc,
                ps_grid,
                planet.epsilon_moisture,
            )
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
        return state, SurfaceState(ocean=ocean, land=surface.land)

    def _land_ocean_post_step(  # noqa: PLR0915
        state: PrimitiveEquationState,
        surface: SurfaceState,
        dt_implicit: float,
    ) -> tuple[PrimitiveEquationState, SurfaceState]:
        """Post-step with land + ocean coupling."""
        # land_config, land_frac, and surface.land are guaranteed non-None
        # by the has_land check at build time.
        lc: BucketLandConfig = land_config  # type: ignore[assignment]
        lf: jnp.ndarray = land_frac  # type: ignore[assignment]
        land: LandState = surface.land  # type: ignore[assignment]

        # Rayleigh friction
        damp = jnp.exp(-dt_implicit * forcing.k_v[:, None])
        state = state.replace(
            vorticity=state.vorticity * damp,
            divergence=state.divergence * damp,
        )

        wind_speed, ps_grid, t_lowest_grid, q_lowest, k_sfc, c_h = _surface_state(
            state,
            surface,
        )
        # Cloud diagnosis for speedy scheme
        cloud_diag_lo = None
        if radiation_scheme == "speedy" and clouds_enabled and state.humidity is not None:
            t_for_cloud = jax.vmap(transform.spectral_to_grid)(state.temperature)
            q_for_cloud = jnp.maximum(
                jax.vmap(transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            cloud_diag_lo = _diagnose_speedy_clouds(state, t_for_cloud, q_for_cloud, ps_grid)

        lw_down, t_grid, q_grid = _compute_lw_down(state, ps_grid, cloud=cloud_diag_lo)

        ocean = surface.ocean

        # --- Albedo: moisture-dependent if configured ---
        effective_albedo: jnp.ndarray | float
        if lc.moisture_dependent_albedo:
            ocean_alb = (
                surface_properties.albedo
                if surface_properties is not None
                else planet.surface_albedo
            )
            effective_albedo = moisture_dependent_albedo(
                land.bucket_depth,
                lc.bucket_capacity,
                lc.albedo_dry,
                lc.albedo_wet,
                lf,
                ocean_alb,
            )
        else:
            effective_albedo = sfc_albedo

        sw_down_sfc = _compute_sw_down_surface(
            state,
            ps_grid,
            effective_albedo,
            cloud=cloud_diag_lo,
        )

        # --- Ocean branch ---
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

        # --- Land branch ---
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
        # Over ocean: decay toward SST / q_sat(SST).
        # Over land: decay toward T_land / beta*q_sat(T_land).
        # The implicit decay keeps the lowest-level air temperature
        # close to the surface, preventing dynamical instability from
        # large air-surface temperature contrasts.
        new_sst = ocean.surface_temperature
        sst_bc = new_sst[:, None] if new_sst.ndim == 1 else new_sst
        t_land = land.soil_temperature

        # Blended surface temperature target
        t_target = (1.0 - lf) * sst_bc + lf * t_land

        # Blended humidity target: ocean=q_sat(SST), land=beta*q_sat(T_land)
        q_sat_ocean = saturation_specific_humidity(
            sst_bc,
            ps_grid,
            planet.epsilon_moisture,
        )
        beta_new = beta_function(land.bucket_depth, lc.w_crit)
        q_sat_land = saturation_specific_humidity(
            t_land,
            ps_grid,
            planet.epsilon_moisture,
        )
        q_target = (1.0 - lf) * q_sat_ocean + lf * beta_new * q_sat_land

        # Implicit atmospheric decay toward blended target
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

        state = PrimitiveEquationState(
            vorticity=state.vorticity,
            divergence=state.divergence,
            temperature=new_temp,
            log_surface_pressure=state.log_surface_pressure,
            humidity=new_humidity,
        )
        return state, SurfaceState(ocean=ocean, land=land)

    # Select the appropriate post-step based on configuration
    coupled_post_step = _land_ocean_post_step if has_land else _ocean_only_post_step

    @jax.jit
    def init_fn(
        state: PrimitiveEquationState,
        surface: SurfaceState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState]:
        previous, current = atm_init_fn(state)
        current, surface = coupled_post_step(current, surface, dt)
        return previous, current, surface

    @jax.jit
    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
        surface: SurfaceState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState]:
        filtered_current, future = atm_step_fn(previous, current)
        future, surface = coupled_post_step(future, surface, 2.0 * dt)
        return filtered_current, future, surface

    return init_fn, step_fn
