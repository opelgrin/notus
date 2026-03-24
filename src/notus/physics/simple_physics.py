"""Simple physics forcing following Frierson et al. (2006).

Composes gray longwave radiation, dry convective adjustment, bulk
surface sensible heat flux, prescribed SST, and Rayleigh boundary-layer
drag into a single ``Forcing`` implementation suitable for aquaplanet
experiments.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators.vector import uv_from_vordiv
from notus.physics.convection import dry_convective_adjustment
from notus.physics.radiation import longwave_heating
from notus.physics.surface import PrescribedSST, compute_sst, surface_sensible_heat_flux
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
        Surface drag coefficient for sensible heat flux.
    n_adjustment_iterations : int
        Number of dry convective adjustment sweeps.
    tau_adjustment : float
        Relaxation timescale for convective adjustment tendency [s].
        Must exceed 2*dt to avoid leapfrog instability.
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

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.sigma_b >= 1.0:
            msg = f"sigma_b must be < 1.0, got {self.sigma_b}"
            raise ValueError(msg)
        if self.tau_adjustment <= 0.0:
            msg = f"tau_adjustment must be > 0, got {self.tau_adjustment}"
            raise ValueError(msg)


class SimplePhysics:
    """Frierson et al. (2006) simple physics forcing.

    Composes gray longwave radiation, dry convective adjustment, bulk
    surface sensible heat flux, and Rayleigh boundary-layer drag for
    an aquaplanet with prescribed SST.

    Shortwave radiation is applied as a prescribed surface flux (no
    atmospheric absorption), following the Frierson convention.

    Implements the ``Forcing`` protocol.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (provides gravity, c_p, kappa, solar constant).
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
        # Transform temperature to grid
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

        # Surface sensible heat flux (lowest level only)
        # Recover wind speed at the lowest level
        lowest = levels.n_levels - 1
        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[lowest],
            state.divergence[lowest],
            self.transform.arrays,
        )
        u_cos_grid = self.transform.spectral_to_grid(u_cos_spec)
        v_cos_grid = self.transform.spectral_to_grid(v_cos_spec)
        cos_lat = self.transform.grid.cos_lat[:, None]  # (n_lat, 1)
        # Gaussian grids do not hit the poles exactly, but a small floor
        # limits numerical amplification at very high latitudes.
        cos_lat_safe = jnp.maximum(cos_lat, 1.0e-6)
        u_grid = u_cos_grid / cos_lat_safe
        v_grid = v_cos_grid / cos_lat_safe
        wind_speed = jnp.sqrt(u_grid**2 + v_grid**2)

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

        # Dry convective adjustment
        t_adjusted = dry_convective_adjustment(
            t_grid,
            levels.sigma_full,
            levels.dsigma,
            planet.kappa,
            n_iterations=cfg.n_adjustment_iterations,
        )
        q_adj = (t_adjusted - t_grid) / cfg.tau_adjustment

        # Total temperature tendency: LW + convective adjustment + surface flux
        dt_grid = q_lw + q_adj
        dt_grid = dt_grid.at[lowest].add(q_sfc)

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
