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
from notus.physics.convection import (
    betts_miller_convection,
    dry_convective_adjustment,
    large_scale_condensation,
)
from notus.physics.radiation import longwave_heating
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
    tau_equator : float
        Equatorial longwave optical depth.
    tau_pole : float
        Polar longwave optical depth.
    linear_fraction : float
        Fraction of LW optical depth with linear pressure dependence.
    alpha : float
        Pressure exponent for the nonlinear part of LW optical depth.
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
    """

    tau_equator: float = 6.0
    tau_pole: float = 0.1
    linear_fraction: float = 0.1
    alpha: float = 4.0
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

        # Pre-compute prescribed SST profile: (n_lat,)
        sst_config = PrescribedSST(
            t_min=self.config.sst_t_min,
            t_delta=self.config.sst_t_delta,
            phi_w=self.config.sst_phi_w,
        )
        self.sst = compute_sst(sst_config, transform.grid.latitudes)

        # Pre-compute Rayleigh friction coefficient per level
        sigma_full = np.asarray(levels.sigma_full)
        sigma_frac = np.maximum(
            0.0,
            (sigma_full - self.config.sigma_b) / (1.0 - self.config.sigma_b),
        )
        self.k_v = jnp.array(self.config.k_f * sigma_frac)  # (n_levels,)

        # Pre-compute lowest-level dsigma as a Python float (JIT-safe)
        self.dsigma_lowest = float(np.asarray(levels.dsigma)[-1])

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
        levels = self.levels
        planet = self.planet
        sin_lat = self.transform.grid.sin_lat

        # --- Rayleigh friction (spectral space, diagonal) ---
        k_v = self.k_v[:, None]  # (n_levels, 1)
        dvort_spec = -k_v * state.vorticity
        ddiv_spec = -k_v * state.divergence

        # --- Temperature tendencies (grid space) ---
        t_grid = jax.vmap(self.transform.spectral_to_grid)(
            state.temperature,
        )  # (n_levels, n_lat, n_lon)

        # Longwave heating rate (no atmospheric SW — Frierson convention)
        q_lw = longwave_heating(
            t_grid,
            self.sst,
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            planet.gravity,
            planet.specific_heat_cp,
            tau_equator=cfg.tau_equator,
            tau_pole=cfg.tau_pole,
            linear_fraction=cfg.linear_fraction,
            alpha=cfg.alpha,
        )

        # Surface winds (lowest level)
        lowest = levels.n_levels - 1
        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[lowest],
            state.divergence[lowest],
            self.transform.arrays,
        )
        u_cos_grid = self.transform.spectral_to_grid(u_cos_spec)
        v_cos_grid = self.transform.spectral_to_grid(v_cos_spec)
        cos_lat = self.transform.grid.cos_lat[:, None]
        cos_lat_safe = jnp.maximum(cos_lat, 1.0e-6)
        u_grid = u_cos_grid / cos_lat_safe
        v_grid = v_cos_grid / cos_lat_safe
        wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

        # Surface sensible heat flux
        q_sfc = surface_sensible_heat_flux(
            self.sst,
            t_grid[lowest],
            wind_speed,
            surface_pressure,
            planet.gravity,
            planet.specific_heat_cp,
            planet.gas_constant,
            self.dsigma_lowest,
            drag_coefficient=cfg.c_d,
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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Moist physics pathway with condensation and convection."""
        cfg = self.config
        levels = self.levels
        planet = self.planet
        lowest = levels.n_levels - 1

        # Transform humidity to grid
        q_grid = jax.vmap(self.transform.spectral_to_grid)(
            humidity_spec,
        )  # (n_levels, n_lat, n_lon)

        # Pressure at full levels
        sigma = levels.sigma_full[:, None, None]
        pressure = sigma * surface_pressure[None, :, :]

        # --- Surface evaporation ---
        q_evap = surface_latent_heat_flux(
            self.sst,
            q_grid[lowest],
            wind_speed,
            surface_pressure,
            planet.gravity,
            planet.gas_constant,
            self.dsigma_lowest,
            planet.epsilon_moisture,
            drag_coefficient=cfg.c_d,
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
        dt_grid = dt_grid.at[lowest].add(q_sfc_sensible)

        # --- Total humidity tendency ---
        dq_grid = dq_bm + dq_cond
        dq_grid = dq_grid.at[lowest].add(q_evap)

        return dt_grid, dq_grid
