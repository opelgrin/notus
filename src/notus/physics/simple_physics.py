"""Simple physics forcing following Frierson et al. (2006, 2007).

Composes gray longwave radiation, convective adjustment, bulk surface
fluxes (sensible + latent heat), large-scale condensation, and
Rayleigh boundary-layer drag into a single ``Forcing`` implementation
suitable for aquaplanet experiments.

When the model state includes humidity, the moist physics pathway is
activated: surface evaporation, large-scale condensation, and
simplified Betts-Miller convection.  Without humidity, the scheme
falls back to the dry configuration (Phase 5).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators.vector import uv_from_vordiv
from notus.physics.boundary_layer import SurfaceLayerConfig, compute_transfer_coefficients
from notus.physics.clouds import CloudConfig, diagnose_clouds
from notus.physics.convection import (
    betts_miller_convection,
    dry_convective_adjustment,
    large_scale_condensation,
)
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import (
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    longwave_heating,
    longwave_optical_depth,
    shortwave_heating,
    speedy_longwave_heating,
    speedy_shortwave_heating,
)
from notus.physics.solar import OrbitalParameters, daily_mean_insolation
from notus.physics.surface import (
    PrescribedSST,
    compute_sst,
    surface_latent_heat_flux,
    surface_sensible_heat_flux,
)
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


@dataclasses.dataclass(frozen=True, slots=True)
class SimplePhysicsConfig:
    """Configuration for simple physics (Frierson et al. 2006/2007).

    Parameters
    ----------
    radiation_scheme : str
        Longwave radiation scheme: ``"frierson"`` (gray, prescribed tau)
        or ``"byrne"`` (humidity-dependent tau with water vapor feedback).
    tau_equator : float
        Equatorial longwave optical depth (Frierson scheme only).
    tau_pole : float
        Polar longwave optical depth (Frierson scheme only).
    linear_fraction : float
        Fraction of LW optical depth with linear pressure dependence
        (Frierson scheme only).
    alpha : float
        Pressure exponent for the nonlinear part of LW optical depth
        (Frierson scheme only).
    byrne_a : float
        Well-mixed gas LW absorption coefficient (Byrne scheme only).
    byrne_b : float
        Water vapor LW absorption coefficient (Byrne scheme only).
    byrne_sw_a : float
        Well-mixed gas SW absorption coefficient (Byrne scheme only).
    byrne_sw_b : float
        Water vapor SW absorption coefficient (Byrne scheme only).
    sw_tau_0 : float
        Shortwave optical depth.  Zero disables atmospheric SW absorption
        (Frierson convention).
    sw_exponent : float
        Shortwave pressure exponent for Beer-Lambert absorption.
    delta_s : float
        Insolation meridional distribution parameter.
    speedy_epslw : float
        LW PBL emission fraction (SPEEDY scheme only).
    speedy_surface_emissivity : float
        Surface LW emissivity (SPEEDY scheme only).
    speedy_ablwin : float
        Window-band absorptivity (SPEEDY LW).
    speedy_ablco2 : float
        CO₂-band absorptivity (SPEEDY LW).
    speedy_ablwv1 : float
        H₂O weak-band absorptivity coefficient (SPEEDY LW).
    speedy_ablwv2 : float
        H₂O strong-band absorptivity coefficient (SPEEDY LW).
    speedy_absdry : float
        Dry-air absorptivity (SPEEDY SW band 1).
    speedy_absaer : float
        Aerosol absorptivity coefficient (SPEEDY SW band 1).
    speedy_sw_abswv1 : float
        Water vapor absorptivity, visible band (SPEEDY SW band 1).
    speedy_sw_abswv2 : float
        Water vapor absorptivity, near-IR band (SPEEDY SW band 2).
    speedy_visible_fraction : float
        Fraction of solar irradiance in the visible band (SPEEDY SW).
    orbital : OrbitalParameters or None
        Orbital parameters for seasonal insolation.  When provided
        together with a ``day_of_year`` set on the forcing object,
        replaces the fixed Frierson insolation with time-varying
        daily-mean insolation from solar geometry.
    sst_t_min : float
        Minimum SST (temperature floor) [K].
    sst_t_delta : float
        Equator-to-floor SST difference [K].
    sst_phi_w : float
        SST latitude width parameter [rad].
    k_f : float
        Boundary-layer Rayleigh friction rate [s-1].
    sigma_b : float
        Boundary-layer top (sigma coordinate).
    c_d : float
        Surface drag coefficient for heat/moisture fluxes.
    n_adjustment_iterations : int
        Number of dry convective adjustment sweeps.
    tau_adjustment : float
        Relaxation timescale for convective adjustment tendency [s].
        Must exceed 2*dt to avoid leapfrog instability.
    tau_bm : float
        Betts-Miller convective relaxation timescale [s].
    rh_ref : float
        Betts-Miller reference relative humidity (0-1).
    n_condensation_iterations : int
        Number of implicit condensation iterations.
    rh_condensation : float
        Relative humidity threshold for large-scale condensation.
    implicit_surface : bool
        When True, Rayleigh friction and surface fluxes (sensible +
        latent heat) are excluded from the explicit tendencies and
        should instead be applied implicitly via :meth:`apply_implicit`.
    """

    radiation_scheme: str = "frierson"
    tau_equator: float = 6.0
    tau_pole: float = 1.5
    linear_fraction: float = 0.1
    alpha: float = 4.0
    byrne_a: float = 0.8678
    byrne_b: float = 1997.9
    byrne_sw_a: float = 0.0
    byrne_sw_b: float = 0.2
    sw_tau_0: float = 0.0
    sw_exponent: float = 2.0
    delta_s: float = 1.4
    speedy_epslw: float = 0.05
    speedy_surface_emissivity: float = 0.98
    speedy_ablwin: float = 0.3
    speedy_ablco2: float = 6.0
    speedy_ablwv1: float = 0.7
    speedy_ablwv2: float = 50.0
    speedy_absdry: float = 0.033
    speedy_absaer: float = 0.033
    speedy_sw_abswv1: float = 0.022
    speedy_sw_abswv2: float = 15.0
    speedy_visible_fraction: float = 0.95
    enable_clouds: bool = False
    cloud_config: CloudConfig = dataclasses.field(default_factory=CloudConfig)
    orbital: OrbitalParameters | None = None
    sst_t_min: float = 271.0
    sst_t_delta: float = 29.0
    sst_phi_w: float = 26.0 * jnp.pi / 180.0
    k_f: float = 1.0 / (1.0 * 86400.0)
    sigma_b: float = 0.7
    c_d: float = 0.0015
    n_adjustment_iterations: int = 3
    tau_adjustment: float = 43200.0
    tau_bm: float = 7200.0
    rh_ref: float = 0.7
    n_condensation_iterations: int = 3
    rh_condensation: float = 1.0
    implicit_surface: bool = True
    surface_layer: SurfaceLayerConfig | None = None

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.radiation_scheme not in {"frierson", "byrne", "speedy"}:
            msg = (
                f"radiation_scheme must be 'frierson', 'byrne', or 'speedy',"
                f" got '{self.radiation_scheme}'"
            )
            raise ValueError(msg)
        if self.sigma_b >= 1.0:
            msg = f"sigma_b must be < 1.0, got {self.sigma_b}"
            raise ValueError(msg)
        if self.tau_adjustment <= 0.0:
            msg = f"tau_adjustment must be > 0, got {self.tau_adjustment}"
            raise ValueError(msg)
        if self.tau_bm <= 0.0:
            msg = f"tau_bm must be > 0, got {self.tau_bm}"
            raise ValueError(msg)
        if not 0.0 < self.rh_ref <= 1.0:
            msg = f"rh_ref must be in (0, 1], got {self.rh_ref}"
            raise ValueError(msg)
        if self.n_condensation_iterations < 1:
            msg = f"n_condensation_iterations must be >= 1, got {self.n_condensation_iterations}"
            raise ValueError(msg)
        if not 0.0 < self.rh_condensation <= 1.0:
            msg = f"rh_condensation must be in (0, 1], got {self.rh_condensation}"
            raise ValueError(msg)


class SimplePhysics:
    """Frierson et al. (2006, 2007) simple physics forcing.

    Composes gray longwave radiation, convective adjustment, bulk
    surface fluxes, and Rayleigh boundary-layer drag for an aquaplanet
    with prescribed SST.

    When the state includes humidity, adds surface evaporation,
    large-scale condensation, and simplified Betts-Miller convection.

    Implements the ``Forcing`` protocol.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    config : SimplePhysicsConfig | None
        Scheme parameters.  Uses Frierson defaults if ``None``.
    """

    def __init__(
        self,
        transform: SpectralTransform,
        planet: PlanetaryConstants,
        levels: SigmaLevels,
        config: SimplePhysicsConfig | None = None,
    ) -> None:
        self.transform = transform
        self.planet = planet
        self.levels = levels
        self.config = config if config is not None else SimplePhysicsConfig()

        # Day of year for seasonal insolation (set by the driver loop)
        self.day_of_year: jnp.ndarray | None = None

        # Pre-compute prescribed SST profile: (n_lat,)
        sst_config = PrescribedSST(
            t_min=self.config.sst_t_min,
            t_delta=self.config.sst_t_delta,
            phi_w=self.config.sst_phi_w,
        )
        self.sst: jnp.ndarray = compute_sst(sst_config, transform.grid.latitudes)

        # Pre-compute Rayleigh friction coefficient per level
        sigma_full = np.asarray(levels.sigma_full)
        sigma_frac = np.maximum(
            0.0,
            (sigma_full - self.config.sigma_b) / (1.0 - self.config.sigma_b),
        )
        self.k_v = jnp.array(self.config.k_f * sigma_frac)  # (n_levels,)

        # Pre-compute lowest-level dsigma as a Python float (JIT-safe)
        self.dsigma_lowest = float(np.asarray(levels.dsigma)[-1])

    def compute_reference_humidity(
        self,
        reference_temperature: np.ndarray,
    ) -> np.ndarray:
        """Compute a reference humidity profile for the semi-implicit solver.

        Returns an RH-based profile matching the initial condition convention:
        ~70 % RH tapered to zero above σ = 0.3.  Used by ``build_pe_stepper``
        to construct the virtual reference temperature ``T_v_ref``.

        Parameters
        ----------
        reference_temperature : np.ndarray
            Dry reference temperature profile, shape ``(n_levels,)``.

        Returns
        -------
        np.ndarray
            Reference specific humidity, shape ``(n_levels,)``.
        """
        sigma_full = np.asarray(self.levels.sigma_full)
        p_ref = sigma_full * self.planet.reference_pressure
        rh_profile = 0.7 * np.minimum(1.0, sigma_full / 0.3)
        q_sat_ref = np.asarray(
            saturation_specific_humidity(
                jnp.array(reference_temperature),
                jnp.array(p_ref),
                self.planet.epsilon_moisture,
            )
        )
        return rh_profile * q_sat_ref

    @property
    def _sst_2d(self) -> jnp.ndarray:
        """Broadcast SST to 2-D ``(n_lat, n_lon)`` if needed."""
        if self.sst.ndim == 1:
            return self.sst[:, None]
        return self.sst

    def _compute_radiation_heating(
        self,
        t_grid: jnp.ndarray,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute total radiative heating rate [K/s]."""
        cfg = self.config
        levels = self.levels
        planet = self.planet
        sin_lat = self.transform.grid.sin_lat

        if cfg.radiation_scheme == "speedy":
            return self._compute_speedy_radiation(
                t_grid,
                state,
                surface_pressure,
            )

        # --- Frierson / Byrne LW ---
        if cfg.radiation_scheme == "byrne" and state.humidity is not None:
            q_grid_for_rad = jax.vmap(self.transform.spectral_to_grid)(state.humidity)
            q_grid_for_rad = jnp.maximum(q_grid_for_rad, 0.0)
            tau_half = byrne_longwave_optical_depth(
                levels.dsigma,
                q_grid_for_rad,
                surface_pressure,
                planet.reference_pressure,
                byrne_a=cfg.byrne_a,
                byrne_b=cfg.byrne_b,
            )
        else:
            tau_half = longwave_optical_depth(
                levels.sigma_half,
                sin_lat,
                tau_equator=cfg.tau_equator,
                tau_pole=cfg.tau_pole,
                linear_fraction=cfg.linear_fraction,
                alpha=cfg.alpha,
            )

        q_lw, _lw_down_sfc = longwave_heating(
            t_grid,
            self.sst,
            tau_half,
            levels.dsigma,
            surface_pressure,
            planet.gravity,
            planet.specific_heat_cp,
        )

        # --- Frierson / Byrne SW ---
        if cfg.sw_tau_0 > 0.0:
            sw_insolation = None
            if self.day_of_year is not None and cfg.orbital is not None:
                sw_insolation = daily_mean_insolation(
                    sin_lat,
                    self.day_of_year,
                    planet.solar_constant,
                    cfg.orbital,
                )

            tau_sw: jnp.ndarray | None = None
            if cfg.radiation_scheme == "byrne" and state.humidity is not None:
                q_grid_for_rad = jax.vmap(self.transform.spectral_to_grid)(state.humidity)
                q_grid_for_rad = jnp.maximum(q_grid_for_rad, 0.0)
                tau_sw = byrne_shortwave_optical_depth(
                    levels.dsigma,
                    q_grid_for_rad,
                    surface_pressure,
                    planet.reference_pressure,
                    sw_tau_0=cfg.sw_tau_0,
                    byrne_sw_a=cfg.byrne_sw_a,
                    byrne_sw_b=cfg.byrne_sw_b,
                )

            q_sw, _sw_down_sfc = shortwave_heating(
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
                insolation=sw_insolation,
                tau_sw_half=tau_sw,
            )
            q_lw += q_sw

        return q_lw

    def _compute_speedy_radiation(
        self,
        t_grid: jnp.ndarray,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute radiation heating with the SPEEDY multi-band scheme."""
        cfg = self.config
        levels = self.levels
        planet = self.planet
        sin_lat = self.transform.grid.sin_lat

        # Humidity (required for SPEEDY — always used with moist state)
        if state.humidity is None:
            msg = "SPEEDY radiation scheme requires humidity"
            raise ValueError(msg)
        q_grid = jax.vmap(self.transform.spectral_to_grid)(state.humidity)
        q_grid = jnp.maximum(q_grid, 0.0)

        # Cloud diagnosis (RH-based, no precipitation in explicit path)
        cloud = None
        if cfg.enable_clouds:
            pressure = levels.sigma_full[:, None, None] * surface_pressure[None, :, :]
            q_sat = saturation_specific_humidity(t_grid, pressure, planet.epsilon_moisture)
            rh = q_grid / jnp.maximum(q_sat, 1e-10)
            geopotential = (
                planet.gravity
                * levels.sigma_full[:, None, None]
                * jnp.ones_like(
                    t_grid,
                )
            )
            cloud = diagnose_clouds(
                rh,
                q_grid,
                t_grid,
                geopotential,
                precipitation_rate=jnp.zeros(surface_pressure.shape),
                convective_mask=jnp.zeros_like(t_grid, dtype=bool),
                gravity=planet.gravity,
                specific_heat_cp=planet.specific_heat_cp,
                config=cfg.cloud_config,
            )

        # LW heating
        q_lw, _lw_down_sfc, _olr = speedy_longwave_heating(
            t_grid,
            self.sst,
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
            cloud=cloud,
        )

        # SW heating (always active for SPEEDY)
        sw_insolation = None
        if self.day_of_year is not None and cfg.orbital is not None:
            sw_insolation = daily_mean_insolation(
                sin_lat,
                self.day_of_year,
                planet.solar_constant,
                cfg.orbital,
            )
        if sw_insolation is None:
            sw_insolation = (
                planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
            )

        q_sw, _sw_down_sfc = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q_grid,
            surface_pressure,
            planet.reference_pressure,
            sw_insolation,
            planet.gravity,
            planet.specific_heat_cp,
            surface_albedo=planet.surface_albedo,
            absdry=cfg.speedy_absdry,
            absaer=cfg.speedy_absaer,
            abswv1=cfg.speedy_sw_abswv1,
            abswv2=cfg.speedy_sw_abswv2,
            visible_fraction=cfg.speedy_visible_fraction,
            cloud=cloud,
        )

        return q_lw + q_sw

    def _compute_surface_exchange(
        self,
        state: PrimitiveEquationState,
        t_grid: jnp.ndarray,
        surface_pressure: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float | jnp.ndarray]:
        """Compute surface winds, sensible heat flux, and drag coefficient.

        Returns
        -------
        tuple[wind_speed, q_sfc, c_h]
        """
        cfg = self.config
        planet = self.planet
        lowest = self.levels.n_levels - 1

        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[lowest],
            state.divergence[lowest],
            self.transform.arrays,
        )
        cos_lat_safe = jnp.maximum(self.transform.grid.cos_lat[:, None], 1.0e-6)
        u_grid = self.transform.spectral_to_grid(u_cos_spec) / cos_lat_safe
        v_grid = self.transform.spectral_to_grid(v_cos_spec) / cos_lat_safe
        wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

        if cfg.implicit_surface:
            return wind_speed, jnp.zeros_like(t_grid[lowest]), cfg.c_d

        # Transfer coefficient: MO or constant
        c_h: float | jnp.ndarray
        if cfg.surface_layer is not None:
            _c_d, c_h = compute_transfer_coefficients(
                self._sst_2d * jnp.ones_like(t_grid[lowest]),
                t_grid[lowest],
                wind_speed,
                self.dsigma_lowest,
                planet.gravity,
                planet.gas_constant,
                cfg.surface_layer,
            )
        else:
            c_h = cfg.c_d

        q_sfc = surface_sensible_heat_flux(
            self.sst,
            t_grid[lowest],
            wind_speed,
            surface_pressure,
            planet.gravity,
            planet.specific_heat_cp,
            planet.gas_constant,
            self.dsigma_lowest,
            drag_coefficient=c_h,
        )
        return wind_speed, q_sfc, c_h

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        """Compute simple physics tendencies.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field ps (grid space), shape ``(n_lat, n_lon)``.

        Returns
        -------
        PrimitiveEquationState
            Tendencies due to simple physics forcing (spectral coefficients).
        """
        cfg = self.config
        implicit = cfg.implicit_surface

        # --- Rayleigh friction (spectral space, diagonal) ---
        if implicit:
            dvort_spec = jnp.zeros_like(state.vorticity)
            ddiv_spec = jnp.zeros_like(state.divergence)
        else:
            k_v = self.k_v[:, None]
            dvort_spec = -k_v * state.vorticity
            ddiv_spec = -k_v * state.divergence

        # --- Grid-space temperature ---
        t_grid = jax.vmap(self.transform.spectral_to_grid)(state.temperature)

        # --- Radiation ---
        q_lw = self._compute_radiation_heating(t_grid, state, surface_pressure)

        # --- Surface exchange ---
        wind_speed, q_sfc, c_h = self._compute_surface_exchange(
            state,
            t_grid,
            surface_pressure,
        )

        # --- Moist or dry pathway ---
        if state.humidity is not None:
            dt_grid, dq_grid = self._moist_physics(
                t_grid,
                state.humidity,
                surface_pressure,
                wind_speed,
                q_lw,
                q_sfc,
                drag_coefficient=c_h,
            )
        else:
            dt_grid = self._dry_physics(t_grid, q_lw, q_sfc)
            dq_grid = None

        # Transform to spectral
        dt_spec = jax.vmap(self.transform.grid_to_spectral)(dt_grid)
        zero_lnps = jnp.zeros_like(state.log_surface_pressure)
        humidity_tend: jnp.ndarray | None = None
        if dq_grid is not None:
            humidity_tend = jax.vmap(self.transform.grid_to_spectral)(dq_grid)

        return PrimitiveEquationState(
            vorticity=dvort_spec,
            divergence=ddiv_spec,
            temperature=dt_spec,
            log_surface_pressure=zero_lnps,
            humidity=humidity_tend,
        )

    def _dry_physics(
        self,
        t_grid: jnp.ndarray,
        q_lw: jnp.ndarray,
        q_sfc: jnp.ndarray,
    ) -> jnp.ndarray:
        """Dry physics pathway (Phase 5 behavior)."""
        cfg = self.config
        levels = self.levels
        planet = self.planet
        lowest = levels.n_levels - 1

        t_adjusted = dry_convective_adjustment(
            t_grid,
            levels.sigma_full,
            levels.dsigma,
            planet.kappa,
            n_iterations=cfg.n_adjustment_iterations,
        )
        q_adj = (t_adjusted - t_grid) / cfg.tau_adjustment

        return (q_lw + q_adj).at[lowest].add(q_sfc)

    def _moist_physics(
        self,
        t_grid: jnp.ndarray,
        humidity_spec: jnp.ndarray,
        surface_pressure: jnp.ndarray,
        wind_speed: jnp.ndarray,
        q_lw: jnp.ndarray,
        q_sfc_sensible: jnp.ndarray,
        *,
        drag_coefficient: float | jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Moist physics pathway with condensation and convection."""
        cfg = self.config
        levels = self.levels
        planet = self.planet
        lowest = levels.n_levels - 1
        c_d = drag_coefficient if drag_coefficient is not None else cfg.c_d

        # Transform humidity to grid
        q_grid = jax.vmap(self.transform.spectral_to_grid)(
            humidity_spec,
        )  # (n_levels, n_lat, n_lon)

        # Pressure at full levels
        sigma = levels.sigma_full[:, None, None]
        pressure = sigma * surface_pressure[None, :, :]

        # --- Surface evaporation ---
        # When implicit_surface=True, evaporation is handled via apply_implicit.
        if cfg.implicit_surface:
            q_evap = jnp.zeros_like(q_grid[lowest])
        else:
            q_evap = surface_latent_heat_flux(
                self.sst,
                q_grid[lowest],
                wind_speed,
                surface_pressure,
                planet.gravity,
                planet.gas_constant,
                self.dsigma_lowest,
                planet.epsilon_moisture,
                drag_coefficient=c_d,
            )

        # --- Betts-Miller convection ---
        dt_bm, dq_bm = betts_miller_convection(
            t_grid,
            q_grid,
            pressure,
            levels.dsigma,
            planet.epsilon_moisture,
            planet.latent_heat_vaporization,
            planet.specific_heat_cp,
            planet.gas_constant,
            tau_bm=cfg.tau_bm,
            rh_ref=cfg.rh_ref,
        )

        # --- Large-scale condensation (on the current state) ---
        t_cond, q_cond, _condensate = large_scale_condensation(
            t_grid,
            q_grid,
            pressure,
            planet.epsilon_moisture,
            planet.latent_heat_vaporization,
            planet.specific_heat_cp,
            planet.gas_constant,
            n_iterations=cfg.n_condensation_iterations,
            rh_threshold=cfg.rh_condensation,
        )
        # Express condensation as a relaxation tendency (same lesson
        # as dry convection: avoid instantaneous adjustment with leapfrog)
        dt_cond = (t_cond - t_grid) / cfg.tau_adjustment
        dq_cond = (q_cond - q_grid) / cfg.tau_adjustment

        # --- Total temperature tendency ---
        dt_grid = q_lw + dt_bm + dt_cond
        dq_grid = dq_bm + dq_cond

        dt_grid = dt_grid.at[lowest].add(q_sfc_sensible)
        dq_grid = dq_grid.at[lowest].add(q_evap)

        return dt_grid, dq_grid

    def apply_implicit(
        self,
        state: PrimitiveEquationState,
        dt_implicit: float,
    ) -> PrimitiveEquationState:
        """Apply implicit physics corrections after the IMEX step.

        Treats stiff boundary-layer terms with exact exponential decay
        for unconditional stability and zero systematic bias:

        - **Rayleigh friction**: ``ζ_new = ζ · exp(-dt·k_v)``
        - **Sensible heat flux**: decay toward SST at lowest level
        - **Latent heat flux**: decay toward q_sat(SST) at lowest level

        For a relaxation ``dX/dt = (X_ref - X) / τ``, the exact solution is::

            X_new = X_ref + (X - X_ref) · exp(-dt / τ)

        This is unconditionally stable AND unbiased — unlike backward Euler
        (which undershoots) or forward Euler (which overshoots).

        Betts-Miller convection remains in the explicit pathway to avoid
        operator-splitting errors that alter the equilibrium climate.

        Should be called AFTER the IMEX time step with
        ``dt_implicit = 2·dt`` for leapfrog or ``dt`` for the Euler init.

        Parameters
        ----------
        state : PrimitiveEquationState
            Post-IMEX state (spectral coefficients).
        dt_implicit : float
            Effective implicit timestep [s].

        Returns
        -------
        PrimitiveEquationState
            State with implicit corrections applied.
        """
        cfg = self.config
        planet = self.planet
        levels = self.levels
        transform = self.transform
        lowest = levels.n_levels - 1

        # --- Rayleigh friction (spectral, exact exponential decay) ---
        damp = jnp.exp(-dt_implicit * self.k_v[:, None])
        new_vort = state.vorticity * damp
        new_div = state.divergence * damp

        # --- Surface winds for drag computation ---
        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[lowest],
            state.divergence[lowest],
            transform.arrays,
        )
        u_cos_grid = transform.spectral_to_grid(u_cos_spec)
        v_cos_grid = transform.spectral_to_grid(v_cos_spec)
        cos_lat = transform.grid.cos_lat[:, None]
        cos_lat_safe = jnp.maximum(cos_lat, 1.0e-6)
        u_grid = u_cos_grid / cos_lat_safe
        v_grid = v_cos_grid / cos_lat_safe
        wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

        # --- Surface pressure and density ---
        lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
        ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)
        dp = self.dsigma_lowest * ps_grid
        dp_safe = jnp.maximum(dp, 1.0)
        sigma_lowest = 1.0 - 0.5 * self.dsigma_lowest
        t_lowest_grid = transform.spectral_to_grid(state.temperature[lowest])
        t_safe = jnp.maximum(t_lowest_grid, 1.0)
        rho_sfc = ps_grid * sigma_lowest / (planet.gas_constant * t_safe)

        # Transfer coefficient: MO stability-dependent or constant
        c_h: float | jnp.ndarray
        if cfg.surface_layer is not None:
            _c_d, c_h = compute_transfer_coefficients(
                self._sst_2d * jnp.ones_like(t_lowest_grid),
                t_lowest_grid,
                wind_speed,
                self.dsigma_lowest,
                planet.gravity,
                planet.gas_constant,
                cfg.surface_layer,
            )
        else:
            c_h = cfg.c_d

        k_sfc = planet.gravity * rho_sfc * c_h * wind_speed / dp_safe

        # --- Sensible heat flux (exact exponential decay at lowest level) ---
        decay_sfc = jnp.exp(-dt_implicit * k_sfc)
        t_corrected = self._sst_2d + (t_lowest_grid - self._sst_2d) * decay_sfc
        new_temp = state.temperature.at[lowest].set(transform.grid_to_spectral(t_corrected))

        # --- Latent heat flux (exact exponential decay at lowest level) ---
        new_humidity: jnp.ndarray | None = None
        if state.humidity is not None:
            q_lowest_grid = transform.spectral_to_grid(state.humidity[lowest])
            q_sat_sfc = saturation_specific_humidity(
                self._sst_2d,
                ps_grid,
                planet.epsilon_moisture,
            )
            q_corrected = q_sat_sfc + (q_lowest_grid - q_sat_sfc) * decay_sfc
            new_humidity = state.humidity.at[lowest].set(transform.grid_to_spectral(q_corrected))

        return PrimitiveEquationState(
            vorticity=new_vort,
            divergence=new_div,
            temperature=new_temp,
            log_surface_pressure=state.log_surface_pressure,
            humidity=new_humidity if new_humidity is not None else state.humidity,
        )
