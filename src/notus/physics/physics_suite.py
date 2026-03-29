"""Composable physics suite for aquaplanet and coupled experiments.

Composes radiation, convective adjustment, bulk surface fluxes
(sensible + latent heat), large-scale condensation, and Rayleigh
boundary-layer drag into a single ``Forcing`` implementation.

Multiple radiation schemes are supported via typed configuration
objects: :class:`FriersonRadiation`, :class:`ByrneRadiation`, and
:class:`SpeedyRadiation`.

When the model state includes humidity, the moist physics pathway is
activated: surface evaporation, large-scale condensation, and
simplified Betts-Miller convection.  Without humidity, the scheme
falls back to the dry configuration.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators.vector import uv_from_vordiv
from notus.physics.boundary_layer import SurfaceLayerConfig, compute_transfer_coefficients
from notus.physics.clouds import CloudDiagnostic, diagnose_clouds
from notus.physics.convection import (
    betts_miller_convection,
    dry_convective_adjustment,
    large_scale_condensation,
)
from notus.physics.forcing import PhysicsDiagnostics
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import (
    ByrneRadiation,
    FriersonRadiation,
    RadiationConfig,
    SpeedyRadiation,
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    longwave_heating,
    longwave_optical_depth,
    lw_down_surface,
    shortwave_heating,
    speedy_longwave_heating,
    speedy_lw_down_surface,
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
class PhysicsSuiteConfig:
    """Configuration for the composable physics suite.

    Parameters
    ----------
    radiation : RadiationConfig
        Radiation scheme configuration.  One of :class:`FriersonRadiation`
        (default), :class:`ByrneRadiation`, or :class:`SpeedyRadiation`.
    orbital : OrbitalParameters or None
        Orbital parameters for seasonal insolation.  When provided
        together with a ``day_of_year`` set on the forcing object,
        replaces the fixed insolation profile with time-varying
        daily-mean insolation from solar geometry.
    delta_s : float
        Insolation meridional distribution parameter.
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
    surface_layer : SurfaceLayerConfig or None
        Optional Monin-Obukhov stability-dependent transfer coefficients.
    """

    radiation: RadiationConfig = dataclasses.field(default_factory=FriersonRadiation)
    orbital: OrbitalParameters | None = None
    delta_s: float = 1.4
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


class PhysicsSuite:
    """Composable physics suite for aquaplanet and coupled experiments.

    Composes radiation, convective adjustment, bulk surface fluxes,
    and Rayleigh boundary-layer drag.  The radiation scheme is selected
    via the typed ``radiation`` field on :class:`PhysicsSuiteConfig`.

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
    config : PhysicsSuiteConfig or None
        Scheme parameters.  Uses Frierson defaults if ``None``.
    """

    def __init__(
        self,
        transform: SpectralTransform,
        planet: PlanetaryConstants,
        levels: SigmaLevels,
        config: PhysicsSuiteConfig | None = None,
    ) -> None:
        self.transform = transform
        self.planet = planet
        self.levels = levels
        self.config = config if config is not None else PhysicsSuiteConfig()

        # Day of year for seasonal insolation.
        # Prefer passing ``day_of_year`` as a keyword argument to
        # ``__call__`` (and to the simulation runner) rather than
        # setting this attribute directly.
        self.day_of_year: jnp.ndarray | None = None

        # Prescribed SST profile from config: (n_lat,)
        sst_config = PrescribedSST(
            t_min=self.config.sst_t_min,
            t_delta=self.config.sst_t_delta,
            phi_w=self.config.sst_phi_w,
        )
        self.prescribed_sst: jnp.ndarray = compute_sst(sst_config, transform.grid.latitudes)

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
        ~70 % RH tapered to zero above sigma = 0.3.  Used by ``build_pe_stepper``
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

    def _sst_2d(self, sst: jnp.ndarray | None = None) -> jnp.ndarray:
        """Broadcast SST to 2-D ``(n_lat, n_lon)`` if needed."""
        sst = sst if sst is not None else self.prescribed_sst
        if sst.ndim == 1:
            return sst[:, None]
        return sst

    # ------------------------------------------------------------------
    # Public surface-flux methods (used by the coupled stepper)
    # ------------------------------------------------------------------

    def compute_insolation(
        self,
        *,
        day_of_year: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute TOA insolation [W/m²], shape ``(n_lat,)``.

        Uses daily-mean solar geometry when orbital parameters and
        ``day_of_year`` are both available; otherwise falls back to the
        fixed Frierson analytic profile.
        """
        sin_lat = self.transform.grid.sin_lat
        day = day_of_year if day_of_year is not None else self.day_of_year
        if day is not None and self.config.orbital is not None:
            return daily_mean_insolation(
                sin_lat,
                day,
                self.planet.solar_constant,
                self.config.orbital,
            )
        delta_s = self.config.delta_s
        return self.planet.solar_constant / 4.0 * (1.0 + delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)

    def compute_clouds(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> CloudDiagnostic | None:
        """Diagnose clouds for SPEEDY radiation.

        Returns ``None`` when clouds are disabled or the radiation
        scheme is not :class:`SpeedyRadiation`.
        """
        rad = self.config.radiation
        if not isinstance(rad, SpeedyRadiation) or rad.clouds is None:
            return None
        if state.humidity is None:
            return None
        levels = self.levels
        planet = self.planet
        t_grid = jax.vmap(self.transform.spectral_to_grid)(state.temperature)
        q_grid = jnp.maximum(
            jax.vmap(self.transform.spectral_to_grid)(state.humidity),
            0.0,
        )
        pressure = levels.sigma_full[:, None, None] * surface_pressure[None, :, :]
        q_sat = saturation_specific_humidity(t_grid, pressure, planet.epsilon_moisture)
        rh = q_grid / jnp.maximum(q_sat, 1e-10)
        geopotential = planet.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t_grid)
        return diagnose_clouds(
            rh,
            q_grid,
            t_grid,
            geopotential,
            precipitation_rate=jnp.zeros(surface_pressure.shape),
            convective_mask=jnp.zeros_like(t_grid, dtype=bool),
            gravity=planet.gravity,
            specific_heat_cp=planet.specific_heat_cp,
            config=rad.clouds,
        )

    def compute_sw_down_surface(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
        *,
        effective_albedo: float | jnp.ndarray,
        day_of_year: jnp.ndarray | None = None,
        cloud: CloudDiagnostic | None = None,
    ) -> jnp.ndarray:
        """Downward SW flux at the surface [W/m²].

        Dispatches across Frierson / Byrne / SPEEDY radiation schemes.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current atmospheric state (spectral).
        surface_pressure : jnp.ndarray
            Surface pressure [Pa], shape ``(n_lat, n_lon)``.
        effective_albedo : float or jnp.ndarray
            Surface albedo (scalar or spatially varying).
        day_of_year : jnp.ndarray or None
            Day of year for seasonal insolation.
        cloud : CloudDiagnostic or None
            Pre-computed cloud diagnostic (SPEEDY only).

        Returns
        -------
        jnp.ndarray
            Downward SW flux at the surface, shape ``(n_lat, n_lon)``.
        """
        rad = self.config.radiation
        levels = self.levels
        planet = self.planet

        if isinstance(rad, SpeedyRadiation) and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(self.transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            insolation = self.compute_insolation(day_of_year=day_of_year)
            _, sw_down_sfc = speedy_shortwave_heating(
                levels.dsigma,
                levels.sigma_full,
                q_grid,
                surface_pressure,
                planet.reference_pressure,
                insolation,
                planet.gravity,
                planet.specific_heat_cp,
                surface_albedo=effective_albedo,
                absdry=rad.absdry,
                absaer=rad.absaer,
                abswv1=rad.sw_abswv1,
                abswv2=rad.sw_abswv2,
                visible_fraction=rad.visible_fraction,
                cloud=cloud,
            )
            return sw_down_sfc

        if isinstance(rad, (FriersonRadiation, ByrneRadiation)) and rad.sw_tau_0 > 0.0:
            insolation = self.compute_insolation(day_of_year=day_of_year)
            tau_sw: jnp.ndarray | None = None
            if isinstance(rad, ByrneRadiation) and state.humidity is not None:
                q_grid = jnp.maximum(
                    jax.vmap(self.transform.spectral_to_grid)(state.humidity),
                    0.0,
                )
                tau_sw = byrne_shortwave_optical_depth(
                    levels.dsigma,
                    q_grid,
                    surface_pressure,
                    planet.reference_pressure,
                    sw_tau_0=rad.sw_tau_0,
                    byrne_sw_a=rad.sw_a,
                    byrne_sw_b=rad.sw_b,
                )
            _, sw_down_sfc = shortwave_heating(
                levels.sigma_half,
                levels.dsigma,
                self.transform.grid.sin_lat,
                surface_pressure,
                planet.solar_constant,
                planet.gravity,
                planet.specific_heat_cp,
                sw_tau_0=rad.sw_tau_0,
                sw_exponent=rad.sw_exponent,
                delta_s=self.config.delta_s,
                insolation=insolation,
                tau_sw_half=tau_sw,
                surface_albedo=effective_albedo,
            )
            return sw_down_sfc

        n_lat, n_lon = surface_pressure.shape
        return jnp.zeros((n_lat, n_lon))

    def compute_lw_down_surface(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
        *,
        sst: jnp.ndarray | None = None,
        cloud: CloudDiagnostic | None = None,
    ) -> jnp.ndarray:
        """Downward LW flux at the surface [W/m²].

        Dispatches across Frierson / Byrne / SPEEDY radiation schemes.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current atmospheric state (spectral).
        surface_pressure : jnp.ndarray
            Surface pressure [Pa], shape ``(n_lat, n_lon)``.
        sst : jnp.ndarray or None
            Surface temperature [K].  Required for SPEEDY LW.
            Defaults to ``self.prescribed_sst``.
        cloud : CloudDiagnostic or None
            Pre-computed cloud diagnostic (SPEEDY only).

        Returns
        -------
        jnp.ndarray
            Downward LW flux at the surface, shape ``(n_lat, n_lon)``.
        """
        rad = self.config.radiation
        levels = self.levels
        planet = self.planet
        t_grid = jax.vmap(self.transform.spectral_to_grid)(state.temperature)

        if isinstance(rad, SpeedyRadiation) and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(self.transform.spectral_to_grid)(state.humidity),
                0.0,
            )
            t_sfc = sst if sst is not None else self.prescribed_sst
            return speedy_lw_down_surface(
                t_grid,
                t_sfc,
                q_grid,
                levels.dsigma,
                surface_pressure,
                planet.reference_pressure,
                epslw=rad.epslw,
                surface_emissivity=rad.surface_emissivity,
                ablwin=rad.ablwin,
                ablco2=rad.ablco2,
                ablwv1=rad.ablwv1,
                ablwv2=rad.ablwv2,
                cloud=cloud,
            )

        if isinstance(rad, ByrneRadiation) and state.humidity is not None:
            q_grid = jnp.maximum(
                jax.vmap(self.transform.spectral_to_grid)(state.humidity),
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
        else:
            fri = rad if isinstance(rad, FriersonRadiation) else FriersonRadiation()
            tau_half = longwave_optical_depth(
                levels.sigma_half,
                self.transform.grid.sin_lat,
                tau_equator=fri.tau_equator,
                tau_pole=fri.tau_pole,
                linear_fraction=fri.linear_fraction,
                alpha=fri.alpha,
            )

        return lw_down_surface(t_grid, tau_half)

    # ------------------------------------------------------------------
    # Internal radiation helpers
    # ------------------------------------------------------------------

    def _compute_radiation_heating(
        self,
        t_grid: jnp.ndarray,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
        *,
        day_of_year: jnp.ndarray | None = None,
        sst: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Compute total radiative heating rate and surface fluxes.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]
            ``(heating_rate, lw_down_surface, sw_down_surface, olr)``
            where heating is [K/s] and fluxes are [W/m²].  OLR is
            ``None`` for Frierson/Byrne schemes (not computed).
        """
        cfg = self.config
        rad = cfg.radiation
        levels = self.levels
        planet = self.planet
        sin_lat = self.transform.grid.sin_lat
        day = day_of_year if day_of_year is not None else self.day_of_year
        sst_val = sst if sst is not None else self.prescribed_sst

        if isinstance(rad, SpeedyRadiation):
            return self._compute_speedy_radiation(
                t_grid,
                state,
                surface_pressure,
                day_of_year=day,
                sst=sst_val,
            )

        # Initialise surface flux accumulators (updated below)
        lw_down_sfc = jnp.zeros_like(surface_pressure)
        sw_down_sfc = jnp.zeros_like(surface_pressure)

        # --- Frierson / Byrne LW ---
        if isinstance(rad, ByrneRadiation) and state.humidity is not None:
            q_grid_for_rad = jax.vmap(self.transform.spectral_to_grid)(state.humidity)
            q_grid_for_rad = jnp.maximum(q_grid_for_rad, 0.0)
            tau_half = byrne_longwave_optical_depth(
                levels.dsigma,
                q_grid_for_rad,
                surface_pressure,
                planet.reference_pressure,
                byrne_a=rad.a,
                byrne_b=rad.b,
            )
        # For Frierson, or Byrne without humidity, use gray optical depth.
        # ByrneRadiation doesn't have Frierson LW params, so use Frierson defaults.
        elif isinstance(rad, FriersonRadiation):
            tau_half = longwave_optical_depth(
                levels.sigma_half,
                sin_lat,
                tau_equator=rad.tau_equator,
                tau_pole=rad.tau_pole,
                linear_fraction=rad.linear_fraction,
                alpha=rad.alpha,
            )
        else:
            # Byrne without humidity: fall back to Frierson defaults
            fri = FriersonRadiation()
            tau_half = longwave_optical_depth(
                levels.sigma_half,
                sin_lat,
                tau_equator=fri.tau_equator,
                tau_pole=fri.tau_pole,
                linear_fraction=fri.linear_fraction,
                alpha=fri.alpha,
            )

        q_lw, lw_down_sfc = longwave_heating(
            t_grid,
            sst_val,
            tau_half,
            levels.dsigma,
            surface_pressure,
            planet.gravity,
            planet.specific_heat_cp,
        )

        # --- Frierson / Byrne SW ---
        if rad.sw_tau_0 > 0.0:
            sw_insolation = None
            if day is not None and cfg.orbital is not None:
                sw_insolation = daily_mean_insolation(
                    sin_lat,
                    day,
                    planet.solar_constant,
                    cfg.orbital,
                )

            tau_sw: jnp.ndarray | None = None
            if isinstance(rad, ByrneRadiation) and state.humidity is not None:
                q_grid_for_rad = jax.vmap(self.transform.spectral_to_grid)(state.humidity)
                q_grid_for_rad = jnp.maximum(q_grid_for_rad, 0.0)
                tau_sw = byrne_shortwave_optical_depth(
                    levels.dsigma,
                    q_grid_for_rad,
                    surface_pressure,
                    planet.reference_pressure,
                    sw_tau_0=rad.sw_tau_0,
                    byrne_sw_a=rad.sw_a,
                    byrne_sw_b=rad.sw_b,
                )

            q_sw, sw_down_sfc = shortwave_heating(
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
                insolation=sw_insolation,
                tau_sw_half=tau_sw,
            )
            q_lw += q_sw

        return q_lw, lw_down_sfc, sw_down_sfc, None

    def _compute_speedy_radiation(
        self,
        t_grid: jnp.ndarray,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
        *,
        day_of_year: jnp.ndarray | None = None,
        sst: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute radiation heating with the SPEEDY multi-band scheme.

        Returns ``(heating_rate, lw_down_surface, sw_down_surface, olr)``.
        """
        cfg = self.config
        rad = cfg.radiation
        if not isinstance(rad, SpeedyRadiation):  # pragma: no cover
            msg = "Expected SpeedyRadiation"
            raise TypeError(msg)
        levels = self.levels
        planet = self.planet
        sin_lat = self.transform.grid.sin_lat
        sst_val = sst if sst is not None else self.prescribed_sst

        # Humidity (required for SPEEDY -- always used with moist state)
        if state.humidity is None:
            msg = "SPEEDY radiation scheme requires humidity"
            raise ValueError(msg)
        q_grid = jax.vmap(self.transform.spectral_to_grid)(state.humidity)
        q_grid = jnp.maximum(q_grid, 0.0)

        # Cloud diagnosis (RH-based, no precipitation in explicit path)
        cloud = None
        if rad.clouds is not None:
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
                config=rad.clouds,
            )

        # LW heating
        q_lw, lw_down_sfc, olr = speedy_longwave_heating(
            t_grid,
            sst_val,
            q_grid,
            levels.dsigma,
            surface_pressure,
            planet.reference_pressure,
            planet.gravity,
            planet.specific_heat_cp,
            epslw=rad.epslw,
            surface_emissivity=rad.surface_emissivity,
            ablwin=rad.ablwin,
            ablco2=rad.ablco2,
            ablwv1=rad.ablwv1,
            ablwv2=rad.ablwv2,
            cloud=cloud,
        )

        # SW heating (always active for SPEEDY)
        sw_insolation = None
        if day_of_year is not None and cfg.orbital is not None:
            sw_insolation = daily_mean_insolation(
                sin_lat,
                day_of_year,
                planet.solar_constant,
                cfg.orbital,
            )
        if sw_insolation is None:
            sw_insolation = (
                planet.solar_constant / 4.0 * (1.0 + cfg.delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)
            )

        q_sw, sw_down_sfc = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q_grid,
            surface_pressure,
            planet.reference_pressure,
            sw_insolation,
            planet.gravity,
            planet.specific_heat_cp,
            surface_albedo=planet.surface_albedo,
            absdry=rad.absdry,
            absaer=rad.absaer,
            abswv1=rad.sw_abswv1,
            abswv2=rad.sw_abswv2,
            visible_fraction=rad.visible_fraction,
            cloud=cloud,
        )

        return q_lw + q_sw, lw_down_sfc, sw_down_sfc, olr

    def _compute_surface_exchange(
        self,
        state: PrimitiveEquationState,
        t_grid: jnp.ndarray,
        surface_pressure: jnp.ndarray,
        *,
        sst: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float | jnp.ndarray]:
        """Compute surface winds, sensible heat flux, and drag coefficient.

        Returns
        -------
        tuple[wind_speed, q_sfc, c_h]
        """
        cfg = self.config
        planet = self.planet
        lowest = self.levels.n_levels - 1
        sst_val = sst if sst is not None else self.prescribed_sst

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
                self._sst_2d(sst_val) * jnp.ones_like(t_grid[lowest]),
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
            sst_val,
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
        *,
        day_of_year: jnp.ndarray | None = None,
        sst: jnp.ndarray | None = None,
    ) -> tuple[PrimitiveEquationState, PhysicsDiagnostics]:
        """Compute physics tendencies and diagnostics.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field p_s (grid space), shape ``(n_lat, n_lon)``.
        day_of_year : jnp.ndarray or None
            Day of year for seasonal insolation.  When ``None``, uses the
            value set on ``self.day_of_year`` (or fixed insolation
            if that is also ``None``).
        sst : jnp.ndarray or None
            Sea surface temperature override [K].  When ``None``, uses the
            prescribed SST profile from config (``self.prescribed_sst``).
            Pass the current ocean SST for coupled slab-ocean runs.

        Returns
        -------
        tuple[PrimitiveEquationState, PhysicsDiagnostics]
            Tendencies (spectral) and diagnostic fields (grid).
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

        # --- Radiation (returns heating + surface fluxes + OLR) ---
        q_lw, lw_down_sfc, sw_down_sfc, olr = self._compute_radiation_heating(
            t_grid,
            state,
            surface_pressure,
            day_of_year=day_of_year,
            sst=sst,
        )

        # --- Surface exchange ---
        wind_speed, q_sfc, c_h = self._compute_surface_exchange(
            state,
            t_grid,
            surface_pressure,
            sst=sst,
        )

        # --- Moist or dry pathway ---
        precip_rate: jnp.ndarray | None = None
        q_evap: jnp.ndarray | None = None
        if state.humidity is not None:
            dt_grid, dq_grid, q_evap, precip_rate = self._moist_physics(
                t_grid,
                state.humidity,
                surface_pressure,
                wind_speed,
                q_lw,
                q_sfc,
                drag_coefficient=c_h,
                sst=sst,
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

        tendencies = PrimitiveEquationState(
            vorticity=dvort_spec,
            divergence=ddiv_spec,
            temperature=dt_spec,
            log_surface_pressure=zero_lnps,
            humidity=humidity_tend,
        )

        # --- Convert tendencies to physical fluxes for diagnostics ---
        # Mass per unit area of the lowest layer: Δσ_lowest · pₛ / g
        planet = self.planet
        mass_lowest = self.dsigma_lowest * surface_pressure / planet.gravity

        # Sensible heat flux [W/m²] = cₚ · (mass/area) · q_sfc [K/s]
        sensible = planet.specific_heat_cp * mass_lowest * q_sfc

        # Evaporation and latent heat (moist only)
        evap: jnp.ndarray | None = None
        latent: jnp.ndarray | None = None
        if q_evap is not None:
            evap = mass_lowest * q_evap
            latent = planet.latent_heat_vaporization * evap

        diags = PhysicsDiagnostics(
            precipitation=precip_rate,
            evaporation=evap,
            olr=olr,
            sw_down_surface=sw_down_sfc,
            lw_down_surface=lw_down_sfc,
            sensible_heat_flux=sensible,
            latent_heat_flux=latent,
        )
        return tendencies, diags

    def _dry_physics(
        self,
        t_grid: jnp.ndarray,
        q_lw: jnp.ndarray,
        q_sfc: jnp.ndarray,
    ) -> jnp.ndarray:
        """Dry physics pathway."""
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
        sst: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Moist physics pathway with condensation and convection.

        Returns
        -------
        tuple[dt_grid, dq_grid, q_evap, precip_rate]
            Temperature tendency [K/s], humidity tendency [kg/kg/s],
            surface evaporation tendency [kg/kg/s] at lowest level,
            and column precipitation rate [kg/m²/s].
        """
        cfg = self.config
        levels = self.levels
        planet = self.planet
        lowest = levels.n_levels - 1
        c_d = drag_coefficient if drag_coefficient is not None else cfg.c_d
        sst_val = sst if sst is not None else self.prescribed_sst

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
                sst_val,
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

        # Precipitation: column-integrated moisture sink from BM + condensation
        # dq_bm + dq_cond are [kg/kg/s]; integrate: Σ (-dq) Δσ pₛ / g
        total_dq_sink = -(dq_bm + dq_cond)  # positive = moisture removed
        precip_rate = (
            jnp.sum(
                total_dq_sink * levels.dsigma[:, None, None] * surface_pressure[None, :, :],
                axis=0,
            )
            / planet.gravity
        )

        return dt_grid, dq_grid, q_evap, precip_rate

    def apply_implicit(
        self,
        state: PrimitiveEquationState,
        dt_implicit: float,
        *,
        sst: jnp.ndarray | None = None,
    ) -> PrimitiveEquationState:
        """Apply implicit physics corrections after the IMEX step.

        Treats stiff boundary-layer terms with exact exponential decay
        for unconditional stability and zero systematic bias:

        - **Rayleigh friction**: ``zeta_new = zeta * exp(-dt*k_v)``
        - **Sensible heat flux**: decay toward SST at lowest level
        - **Latent heat flux**: decay toward q_sat(SST) at lowest level

        Parameters
        ----------
        state : PrimitiveEquationState
            Post-IMEX state (spectral coefficients).
        dt_implicit : float
            Effective implicit timestep [s].
        sst : jnp.ndarray or None
            Sea surface temperature override [K].  ``None`` uses
            ``self.prescribed_sst``.

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
        sst_2d = self._sst_2d(sst)

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
                sst_2d * jnp.ones_like(t_lowest_grid),
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
        t_corrected = sst_2d + (t_lowest_grid - sst_2d) * decay_sfc
        new_temp = state.temperature.at[lowest].set(transform.grid_to_spectral(t_corrected))

        # --- Latent heat flux (exact exponential decay at lowest level) ---
        new_humidity: jnp.ndarray | None = None
        if state.humidity is not None:
            q_lowest_grid = transform.spectral_to_grid(state.humidity[lowest])
            q_sat_sfc = saturation_specific_humidity(
                sst_2d,
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
