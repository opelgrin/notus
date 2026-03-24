"""Physics forcing protocols and implementations.

Defines the interface that all forcing functions must satisfy, plus the
Held-Suarez (1994) forcing for the standard benchmark.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


class Forcing(Protocol):
    """Protocol for physics forcing functions.

    A forcing function receives the current model state and returns a
    ``PrimitiveEquationState`` containing the tendencies due to physics
    parameterizations (e.g. radiation, boundary layer drag).

    The tendencies are added to the explicit dynamics tendencies at each
    time step.
    """

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        """Compute physics tendencies.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field pₛ (grid space), shape ``(n_lat, n_lon)``.

        Returns
        -------
        PrimitiveEquationState
            Tendencies due to physics forcing (spectral coefficients).
        """
        ...


class HeldSuarez:
    """Held-Suarez (1994) forcing for the standard GCM benchmark.

    Implements Newtonian relaxation toward a prescribed equilibrium
    temperature profile and Rayleigh drag in the boundary layer.

    The equilibrium temperature is::

        T_eq = max(T_min, [315 - ΔT_y sin²φ - Δθ_z log(p/p₀) cos²φ] (p/p₀)^κ)

    The Newtonian relaxation rate varies from k_a in the free atmosphere
    to k_s at the surface within the boundary layer (σ > σ_b)::

        k_T = k_a + (k_s - k_a) max(0, (σ - σ_b)/(1 - σ_b)) cos⁴φ

    Rayleigh friction damps vorticity and divergence in the boundary layer::

        k_v = k_f max(0, (σ - σ_b)/(1 - σ_b))

    References
    ----------
    Held, I. M. & Suarez, M. J. (1994). A proposal for the intercomparison
    of the dynamical cores of atmospheric general circulation models.
    BAMS 75(10), 1825-1830.
    """

    def __init__(
        self,
        transform: SpectralTransform,
        planet: PlanetaryConstants,
        levels: SigmaLevels,
        *,
        k_a: float = 1.0 / (40.0 * 86400.0),
        k_s: float = 1.0 / (4.0 * 86400.0),
        k_f: float = 1.0 / (1.0 * 86400.0),
        delta_t_y: float = 60.0,
        delta_theta_z: float = 10.0,
        sigma_b: float = 0.7,
        t_min: float = 200.0,
        p0: float = 1.0e5,
    ) -> None:
        """Initialise Held-Suarez forcing.

        Parameters
        ----------
        transform : SpectralTransform
            Pre-computed spectral transform (for grid ↔ spectral conversion).
        planet : PlanetaryConstants
            Planetary constants (provides κ = R/cₚ).
        levels : SigmaLevels
            Sigma vertical coordinate.
        k_a : float
            Free-atmosphere relaxation rate [s⁻¹].
        k_s : float
            Surface relaxation rate [s⁻¹].
        k_f : float
            Boundary-layer friction rate [s⁻¹].
        delta_t_y : float
            Equator-to-pole temperature difference [K].
        delta_theta_z : float
            Vertical temperature lapse-rate parameter [K].
        sigma_b : float
            Boundary-layer top (σ coordinate).
        t_min : float
            Minimum equilibrium temperature [K].
        p0 : float
            Reference pressure [Pa].
        """
        self.transform = transform
        self.planet = planet
        self.levels = levels
        self.k_a = k_a
        self.k_s = k_s
        self.k_f = k_f
        self.delta_t_y = delta_t_y
        self.delta_theta_z = delta_theta_z
        self.sigma_b = sigma_b
        self.t_min = t_min
        self.p0 = p0
        self.kappa = planet.kappa

        # Pre-compute boundary-layer ramp: max(0, (σ - σ_b)/(1 - σ_b))
        sigma_full = np.asarray(levels.sigma_full)
        self.sigma_frac = jnp.array(np.maximum(0.0, (sigma_full - sigma_b) / (1.0 - sigma_b)))

        # Rayleigh friction coefficient per level (level-only, no lat dependence)
        self.k_v = k_f * self.sigma_frac

        # Grid geometry (captured for use in __call__)
        self.sin_lat = transform.grid.sin_lat
        self.cos_lat = transform.grid.cos_lat

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        """Compute Held-Suarez tendencies.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field pₛ (grid space), shape ``(n_lat, n_lon)``.

        Returns
        -------
        PrimitiveEquationState
            Tendencies due to Held-Suarez forcing (spectral coefficients).
        """
        # --- Rayleigh friction (spectral space) ---
        # k_v depends only on sigma level, so multiplication is diagonal
        k_v = self.k_v[:, None]  # (n_levels, 1)
        dvort_spec = -k_v * state.vorticity
        ddiv_spec = -k_v * state.divergence

        # --- Newtonian temperature relaxation (grid space) ---
        # Transform temperature to grid
        t_grid = jax.vmap(self.transform.spectral_to_grid)(state.temperature)

        # Pressure at each level: p = σ · pₛ
        sigma = self.levels.sigma_full[:, None, None]  # (n_levels, 1, 1)
        ps = surface_pressure[None, :, :]  # (1, n_lat, n_lon)
        p = sigma * ps

        # Equilibrium temperature (H&S 1994, Eq. 1)
        sin2 = self.sin_lat[None, :, None] ** 2  # (1, n_lat, 1)
        cos2 = self.cos_lat[None, :, None] ** 2  # (1, n_lat, 1)
        p_ratio = p / self.p0
        t_eq = jnp.maximum(
            self.t_min,
            (315.0 - self.delta_t_y * sin2 - self.delta_theta_z * jnp.log(p_ratio) * cos2)
            * p_ratio**self.kappa,
        )

        # Newtonian relaxation rate (H&S 1994, Eq. 2)
        cos4 = self.cos_lat[None, :, None] ** 4  # (1, n_lat, 1)
        sigma_frac = self.sigma_frac[:, None, None]  # (n_levels, 1, 1)
        k_t = self.k_a + (self.k_s - self.k_a) * sigma_frac * cos4

        # Temperature tendency in grid space
        dt_grid = -k_t * (t_grid - t_eq)

        # Transform back to spectral
        dt_spec = jax.vmap(self.transform.grid_to_spectral)(dt_grid)

        # No surface pressure tendency from Held-Suarez forcing
        zero_lnps = jnp.zeros_like(state.log_surface_pressure)

        return PrimitiveEquationState(
            vorticity=dvort_spec,
            divergence=ddiv_spec,
            temperature=dt_spec,
            log_surface_pressure=zero_lnps,
        )
