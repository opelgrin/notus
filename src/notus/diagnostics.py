"""Diagnostics for the primitive equations.

Provides:

- **Conservation diagnostics**: global integrals of mass, total energy,
  and angular momentum (should be conserved by adiabatic dynamics).
- **Zonal-mean diagnostics**: time-averaged zonal-mean fields for
  climatological analysis (zonal wind, temperature, eddy statistics).

All horizontal integrals use exact Gaussian quadrature and midpoint-rule
sigma integration in the vertical.

References
----------
- Held & Suarez (1994), §4: conservation properties
- Durran, "Numerical Methods for Fluid Dynamics", §8.6
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from notus.constants import PlanetaryConstants
from notus.grid import GaussianGrid
from notus.operators import uv_from_vordiv
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


def spherical_integral(
    field: jnp.ndarray,
    grid: GaussianGrid,
) -> jnp.ndarray:
    """Integrate a nodal field over the sphere.

    Uses Gaussian quadrature in latitude and uniform trapezoidal rule
    in longitude:

        ∫ f dA = Σⱼ wⱼ · Σᵢ f(λᵢ, φⱼ) · Δλ

    where wⱼ are Gauss-Legendre quadrature weights (which absorb the
    cosφ Jacobian factor).

    Parameters
    ----------
    field : jnp.ndarray
        Nodal field, shape ``(..., n_lat, n_lon)``.
    grid : GaussianGrid
        Gaussian grid (provides quadrature weights).

    Returns
    -------
    jnp.ndarray
        Scalar integral (or batch of scalars if field has leading dims).
    """
    dlon = grid.dlon
    # Sum over longitude, then weighted sum over latitude
    lon_sum = jnp.sum(field, axis=-1) * dlon  # (..., n_lat)
    return jnp.sum(lon_sum * grid.lat_weights, axis=-1)


def sigma_integral(
    field: jnp.ndarray,
    levels: SigmaLevels,
) -> jnp.ndarray:
    """Integrate a field over sigma levels (midpoint rule).

    Computes ∫₀¹ f dσ ≈ Σₖ f(σₖ) Δσₖ.

    Parameters
    ----------
    field : jnp.ndarray
        Per-level field, shape ``(n_levels, ...)``.
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    jnp.ndarray
        Column integral, shape ``(...,)`` (level axis removed).
    """
    # dsigma shape (n_levels,), broadcast to field
    weights = jnp.reshape(levels.dsigma, (-1,) + (1,) * (field.ndim - 1))
    return jnp.sum(field * weights, axis=0)


@dataclass(frozen=True, slots=True)
class ConservationDiagnostics:
    """Global conservation quantities.

    Attributes
    ----------
    mass : float
        Total atmospheric mass [kg].
    total_energy : float
        Total energy [J].
    angular_momentum : float
        Total axial angular momentum [kg·m²/s].
    kinetic_energy : float
        Global kinetic energy [J].
    internal_energy : float
        Global internal energy (cₚ·T, pressure-weighted) [J].
    potential_energy : float
        Global potential energy (Φₛ, pressure-weighted) [J].
    """

    mass: float
    total_energy: float
    angular_momentum: float
    kinetic_energy: float
    internal_energy: float
    potential_energy: float


def compute_conservation_diagnostics(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    surface_geopotential: jnp.ndarray,
) -> ConservationDiagnostics:
    """Compute global conservation diagnostics from a PE state.

    Parameters
    ----------
    state : PrimitiveEquationState
        Model state in spectral space.
    transform : SpectralTransform
        Spectral transform (for grid conversion).
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    surface_geopotential : jnp.ndarray
        Surface geopotential g·zₛ in spectral space, shape ``(n_spectral,)``.

    Returns
    -------
    ConservationDiagnostics
        Global mass, energy, and angular momentum.
    """
    grid = transform.grid
    arrays = transform.arrays

    # --- Surface pressure on grid ---
    lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
    ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)  # (n_lat, n_lon)

    # --- Mass: M = (1/g) ∫ ps dA  where dA = a²·cosφ·dφ·dλ ---
    a2 = arrays.radius**2
    mass = a2 * spherical_integral(ps_grid, grid) / planet.gravity

    # --- Per-level grid-point fields (vectorized) ---
    # Reconstruct winds: vmap over levels
    def _uv_at_level(vort: jnp.ndarray, div: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return uv_from_vordiv(vort, div, arrays)

    u_cos_spec, v_cos_spec = jax.vmap(_uv_at_level)(state.vorticity, state.divergence)

    # Batch all spectral→grid transforms: u_cos, v_cos, temperature per level
    all_spec = jnp.concatenate([u_cos_spec, v_cos_spec, state.temperature], axis=0)
    all_grid = jax.vmap(transform.spectral_to_grid)(all_spec)

    n_levels = levels.n_levels
    u_cos_grid = all_grid[:n_levels]
    v_cos_grid = all_grid[n_levels : 2 * n_levels]
    t_grid = all_grid[2 * n_levels : 3 * n_levels]

    cos_lat = grid.cos_lat[None, :, None]  # (1, n_lat, 1)
    u = u_cos_grid / cos_lat
    v = v_cos_grid / cos_lat

    ke_levels = 0.5 * (u**2 + v**2)
    ie_levels = planet.specific_heat_cp * t_grid
    omega_a_cos = planet.rotation_rate * arrays.radius * cos_lat
    am_levels = (u + omega_a_cos) * arrays.radius * cos_lat

    # --- Surface geopotential on grid ---
    phi_s_grid = transform.spectral_to_grid(surface_geopotential)

    # --- Vertical + horizontal integration ---
    # Each integrand is weighted by ps/g (mass per unit area per dσ)
    # dA = a² cosφ dφ dλ (the a² is factored out)
    ps_over_g = ps_grid / planet.gravity  # (n_lat, n_lon)

    # Kinetic energy: (a²/g) ∫ ps ∫ ½|v|² dσ dA
    ke_column = sigma_integral(ke_levels, levels)  # (n_lat, n_lon)
    kinetic_energy = a2 * spherical_integral(ps_over_g * ke_column, grid)

    # Internal energy: (a²/g) ∫ ps ∫ cp*T dσ dA
    ie_column = sigma_integral(ie_levels, levels)
    internal_energy = a2 * spherical_integral(ps_over_g * ie_column, grid)

    # Potential energy: (a²/g) ∫ ps * Φs dA  (Φs doesn't depend on σ)
    potential_energy = a2 * spherical_integral(ps_over_g * phi_s_grid, grid)

    total_energy = kinetic_energy + internal_energy + potential_energy

    # Angular momentum: (a²/g) ∫ ps ∫ (u + Ωa cosφ) a cosφ dσ dA
    am_column = sigma_integral(am_levels, levels)
    angular_momentum = a2 * spherical_integral(ps_over_g * am_column, grid)

    return ConservationDiagnostics(
        mass=float(mass),
        total_energy=float(total_energy),
        angular_momentum=float(angular_momentum),
        kinetic_energy=float(kinetic_energy),
        internal_energy=float(internal_energy),
        potential_energy=float(potential_energy),
    )


# =====================================================================
# Zonal-mean diagnostics
# =====================================================================


@dataclass(frozen=True, slots=True)
class ZonalMeanState:
    """Zonal-mean fields on the latitude-sigma grid.

    All arrays have shape ``(n_levels, n_lat)``.

    Attributes
    ----------
    u : ndarray
        Zonal-mean zonal wind [m/s].
    v : ndarray
        Zonal-mean meridional wind [m/s].
    temperature : ndarray
        Zonal-mean temperature [K].
    u_prime_sq : ndarray
        Eddy zonal kinetic energy [u'²] [m²/s²].
    v_prime_sq : ndarray
        Eddy meridional kinetic energy [v'²] [m²/s²].
    uv_prime : ndarray
        Eddy momentum flux [u'v'] [m²/s²].
    vt_prime : ndarray
        Eddy heat flux [v'T'] [K·m/s].
    """

    u: jnp.ndarray
    v: jnp.ndarray
    temperature: jnp.ndarray
    u_prime_sq: jnp.ndarray
    v_prime_sq: jnp.ndarray
    uv_prime: jnp.ndarray
    vt_prime: jnp.ndarray


def compute_zonal_mean_state(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
) -> ZonalMeanState:
    """Compute instantaneous zonal-mean fields from spectral state.

    Reconstructs winds from vorticity/divergence, transforms to grid space,
    and computes zonal means and eddy statistics.

    Parameters
    ----------
    state : PrimitiveEquationState
        Model state in spectral space.
    transform : SpectralTransform
        Spectral transform (for grid conversion).

    Returns
    -------
    ZonalMeanState
        Zonal-mean fields and eddy statistics.
    """
    grid = transform.grid
    arrays = transform.arrays
    n_levels = state.n_levels
    cos_lat = grid.cos_lat

    # Reconstruct winds and temperature on grid
    def _uv_at_level(
        vort: jnp.ndarray,
        div: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return uv_from_vordiv(vort, div, arrays)

    u_cos_spec, v_cos_spec = jax.vmap(_uv_at_level)(
        state.vorticity,
        state.divergence,
    )
    all_spec = jnp.concatenate(
        [u_cos_spec, v_cos_spec, state.temperature],
        axis=0,
    )
    all_grid = jax.vmap(transform.spectral_to_grid)(all_spec)

    u_cos_grid = all_grid[:n_levels]
    v_cos_grid = all_grid[n_levels : 2 * n_levels]
    t_grid = all_grid[2 * n_levels : 3 * n_levels]

    # u*cos(lat) -> u
    cos_lat_bc = cos_lat[None, :, None]
    u_grid = u_cos_grid / cos_lat_bc
    v_grid = v_cos_grid / cos_lat_bc

    # Zonal means: average over longitude
    u_zm = jnp.mean(u_grid, axis=-1)
    v_zm = jnp.mean(v_grid, axis=-1)
    t_zm = jnp.mean(t_grid, axis=-1)

    # Eddy fields (deviation from zonal mean)
    u_prime = u_grid - u_zm[:, :, None]
    v_prime = v_grid - v_zm[:, :, None]
    t_prime = t_grid - t_zm[:, :, None]

    # Eddy statistics (zonal mean of products)
    return ZonalMeanState(
        u=u_zm,
        v=v_zm,
        temperature=t_zm,
        u_prime_sq=jnp.mean(u_prime**2, axis=-1),
        v_prime_sq=jnp.mean(v_prime**2, axis=-1),
        uv_prime=jnp.mean(u_prime * v_prime, axis=-1),
        vt_prime=jnp.mean(v_prime * t_prime, axis=-1),
    )
