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
from notus.physics.moisture import saturation_specific_humidity
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
    implicit_surface : bool
        When True, Rayleigh friction and surface fluxes (sensible +
        latent heat) are excluded from the explicit tendencies and
        should instead be applied implicitly via :meth:`apply_implicit`.
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
    implicit_surface: bool = True

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
        implicit = cfg.implicit_surface

        # --- Rayleigh friction (spectral space, diagonal) ---
        # When implicit_surface=True, friction is applied via apply_implicit.
        if implicit:
            dvort_spec = jnp.zeros_like(state.vorticity)
            ddiv_spec = jnp.zeros_like(state.divergence)
        else:
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

        # Surface winds (lowest level) — needed for both explicit and moist fluxes
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
        # When implicit_surface=True, this is handled via apply_implicit.
        if implicit:
            q_sfc = jnp.zeros_like(t_grid[lowest])
        else:
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
        # When implicit_surface=True, BM is applied via apply_implicit.
        if cfg.implicit_surface:
            dt_grid = q_lw + dt_cond
            dq_grid = dq_cond
        else:
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

        Treats stiff terms with backward Euler for unconditional stability:

        - **Rayleigh friction**: ``ζ_new = ζ / (1 + dt·k_v)`` per level
        - **Sensible heat flux**: relaxation toward SST at lowest level
        - **Latent heat flux**: relaxation toward q_sat(SST) at lowest level
        - **Betts-Miller convection**: relaxation toward moist adiabat

        For a relaxation tendency ``(X_ref - X) / τ``, backward Euler gives::

            X_new = (X + (dt/τ)·X_ref) / (1 + dt/τ)

        which is unconditionally stable for any dt and τ.

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

        # --- Rayleigh friction (spectral, diagonal, all levels) ---
        damp = 1.0 / (1.0 + dt_implicit * self.k_v[:, None])
        new_vort = state.vorticity * damp
        new_div = state.divergence * damp

        # --- Transform T (and q) to grid for surface + BM corrections ---
        t_grid = jax.vmap(transform.spectral_to_grid)(state.temperature)

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
        t_safe = jnp.maximum(t_grid[lowest], 1.0)
        rho_sfc = ps_grid * sigma_lowest / (planet.gas_constant * t_safe)
        k_sfc = planet.gravity * rho_sfc * cfg.c_d * wind_speed / dp_safe

        # --- Sensible heat flux (backward Euler at lowest level) ---
        t_grid = t_grid.at[lowest].set(
            (t_grid[lowest] + dt_implicit * k_sfc * self.sst[:, None])
            / (1.0 + dt_implicit * k_sfc)
        )

        # --- Humidity: surface latent heat + Betts-Miller ---
        q_grid: jnp.ndarray | None = None
        if state.humidity is not None:
            q_grid = jax.vmap(transform.spectral_to_grid)(state.humidity)

            # Latent heat flux (backward Euler at lowest level)
            q_sat_sfc = saturation_specific_humidity(
                self.sst[:, None], ps_grid, planet.epsilon_moisture,
            )
            q_grid = q_grid.at[lowest].set(
                (q_grid[lowest] + dt_implicit * k_sfc * q_sat_sfc)
                / (1.0 + dt_implicit * k_sfc)
            )

            # --- Betts-Miller convection (backward Euler relaxation) ---
            pressure = levels.sigma_full[:, None, None] * ps_grid[None, :, :]
            dt_bm, dq_bm = betts_miller_convection(
                t_grid, q_grid, pressure, levels.dsigma,
                planet.epsilon_moisture, planet.latent_heat_vaporization,
                planet.specific_heat_cp, planet.gas_constant,
                tau_bm=cfg.tau_bm, rh_ref=cfg.rh_ref,
            )
            # BM returns (T_ref - T) / τ.  Backward Euler scales this by
            # 1 / (1 + dt/τ) for unconditional stability.
            bm_scale = dt_implicit / (1.0 + dt_implicit / cfg.tau_bm)
            t_grid += bm_scale * dt_bm
            q_grid += bm_scale * dq_bm

        # --- Transform back to spectral ---
        new_temp = jax.vmap(transform.grid_to_spectral)(t_grid)
        new_humidity: jnp.ndarray | None = None
        if q_grid is not None:
            new_humidity = jax.vmap(transform.grid_to_spectral)(q_grid)

        return PrimitiveEquationState(
            vorticity=new_vort,
            divergence=new_div,
            temperature=new_temp,
            log_surface_pressure=state.log_surface_pressure,
            humidity=new_humidity if new_humidity is not None else state.humidity,
        )
