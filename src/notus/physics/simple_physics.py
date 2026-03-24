"""Simple physics forcing following Frierson et al. (2006).

Composes gray longwave/shortwave radiation, dry convective adjustment,
prescribed SST, and Rayleigh boundary-layer drag into a single ``Forcing``
implementation suitable for aquaplanet experiments.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.physics.convection import dry_convective_adjustment
from notus.physics.radiation import longwave_heating, shortwave_heating
from notus.physics.surface import PrescribedSST, compute_sst
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


@dataclasses.dataclass(frozen=True, slots=True)
class SimplePhysicsConfig:
    """Configuration for simple physics (Frierson et al. 2006).

    Parameters
    ----------
    tau_equator : float
        Equatorial longwave optical depth.
    tau_pole : float
        Polar longwave optical depth.
    alpha : float
        Pressure exponent for LW optical depth profile.
    sw_tau_0 : float
        Shortwave optical depth.
    sw_exponent : float
        Shortwave pressure exponent.
    delta_s : float
        Insolation distribution parameter.
    sst_t_max : float
        Maximum (equatorial) SST [K].
    sst_delta_t : float
        Equator-to-pole SST difference [K].
    k_f : float
        Boundary-layer Rayleigh friction rate [s⁻¹].
    sigma_b : float
        Boundary-layer top (σ coordinate).
    n_adjustment_iterations : int
        Number of dry convective adjustment sweeps.
    tau_adjustment : float
        Relaxation timescale for convective adjustment tendency [s].
        Must exceed 2·dt to avoid leapfrog instability.
    """

    tau_equator: float = 7.2
    tau_pole: float = 1.8
    alpha: float = 4.0
    sw_tau_0: float = 0.22
    sw_exponent: float = 2.0
    delta_s: float = 1.4
    sst_t_max: float = 285.0
    sst_delta_t: float = 40.0
    k_f: float = 1.0 / (1.0 * 86400.0)
    sigma_b: float = 0.7
    n_adjustment_iterations: int = 3
    tau_adjustment: float = 43200.0


class SimplePhysics:
    """Frierson et al. (2006) simple physics forcing.

    Composes gray radiation (longwave + shortwave), dry convective
    adjustment, and Rayleigh boundary-layer drag for an aquaplanet
    with prescribed SST.

    Implements the ``Forcing`` protocol.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (provides gravity, c_p, κ, solar constant).
    levels : SigmaLevels
        Sigma vertical coordinate.
    config : SimplePhysicsConfig | None
        Scheme parameters.  Uses Frierson (2006) defaults if ``None``.
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
            t_max=self.config.sst_t_max,
            delta_t=self.config.sst_delta_t,
        )
        self.sst = compute_sst(sst_config, transform.grid.sin_lat)

        # Pre-compute Rayleigh friction coefficient per level
        sigma_full = np.asarray(levels.sigma_full)
        sigma_frac = np.maximum(
            0.0, (sigma_full - self.config.sigma_b) / (1.0 - self.config.sigma_b),
        )
        self.k_v = jnp.array(self.config.k_f * sigma_frac)  # (n_levels,)

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
            Surface pressure field pₛ (grid space), shape ``(n_lat, n_lon)``.

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
        # Transform temperature to grid
        t_grid = jax.vmap(self.transform.spectral_to_grid)(
            state.temperature,
        )  # (n_levels, n_lat, n_lon)

        # Longwave heating rate
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
            alpha=cfg.alpha,
        )

        # Shortwave heating rate
        q_sw = shortwave_heating(
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
        )

        # Dry convective adjustment
        t_adjusted = dry_convective_adjustment(
            t_grid,
            levels.sigma_full,
            levels.dsigma,
            planet.kappa,
            n_iterations=cfg.n_adjustment_iterations,
        )
        q_adj = (t_adjusted - t_grid) / cfg.tau_adjustment

        # Total temperature tendency
        dt_grid = q_lw + q_sw + q_adj

        # Transform to spectral
        dt_spec = jax.vmap(self.transform.grid_to_spectral)(dt_grid)

        # No surface pressure tendency
        zero_lnps = jnp.zeros_like(state.log_surface_pressure)

        return PrimitiveEquationState(
            vorticity=dvort_spec,
            divergence=ddiv_spec,
            temperature=dt_spec,
            log_surface_pressure=zero_lnps,
        )
