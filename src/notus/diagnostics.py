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
import numpy as np

from notus.constants import PlanetaryConstants
from notus.grid import GaussianGrid
from notus.operators import uv_from_vordiv
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


def grid_winds_at_level(
    state: PrimitiveEquationState,
    level: int,
    transform: SpectralTransform,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct grid-space winds (u, v) at a single sigma level.

    Converts spectral vorticity/divergence to wind components and
    divides by cos(lat) to obtain true (u, v).

    Parameters
    ----------
    state : PrimitiveEquationState
        Model state in spectral space.
    level : int
        Sigma level index (0 = top, n_levels-1 = surface).
    transform : SpectralTransform
        Spectral transform.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(u, v)`` wind components in m/s, each shape ``(n_lat, n_lon)``.
    """
    u_cos_spec, v_cos_spec = uv_from_vordiv(
        state.vorticity[level],
        state.divergence[level],
        transform.arrays,
    )
    cos_lat = jnp.maximum(transform.grid.cos_lat[:, None], 1.0e-30)
    u = transform.spectral_to_grid(u_cos_spec) / cos_lat
    v = transform.spectral_to_grid(v_cos_spec) / cos_lat
    return u, v


def grid_surface_pressure(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
) -> jnp.ndarray:
    """Compute surface pressure on the grid from spectral log(ps).

    Parameters
    ----------
    state : PrimitiveEquationState
        Model state in spectral space.
    transform : SpectralTransform
        Spectral transform.
    planet : PlanetaryConstants
        Planetary constants (provides reference pressure).

    Returns
    -------
    jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    """
    lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
    return planet.reference_pressure * jnp.exp(lnps_grid)


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

    @property
    def eke(self) -> jnp.ndarray:
        """Eddy kinetic energy ½(u'² + v'²), shape ``(n_levels, n_lat)``."""
        return 0.5 * (self.u_prime_sq + self.v_prime_sq)


def _zm_flatten(
    zm: ZonalMeanState,
) -> tuple[tuple[jnp.ndarray, ...], None]:
    return (
        (zm.u, zm.v, zm.temperature, zm.u_prime_sq, zm.v_prime_sq, zm.uv_prime, zm.vt_prime),
        None,
    )


def _zm_unflatten(_aux: None, children: tuple[jnp.ndarray, ...]) -> ZonalMeanState:
    return ZonalMeanState(*children)


jax.tree_util.register_pytree_node(ZonalMeanState, _zm_flatten, _zm_unflatten)


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


# =====================================================================
# Meridional overturning streamfunction
# =====================================================================


def compute_streamfunction(
    zm_v: jnp.ndarray,
    cos_lat: jnp.ndarray,
    levels: SigmaLevels,
    planet: PlanetaryConstants,
    mean_ps: float = 1.0e5,
) -> jnp.ndarray:
    """Compute the meridional mass streamfunction.

    Ψ(φ, σ) = (2π a cos φ / g) · pₛ · ∫₀^σ [v] dσ'

    where [v] is the zonal-mean meridional wind.

    Parameters
    ----------
    zm_v : jnp.ndarray
        Zonal-mean meridional wind [m/s], shape ``(n_levels, n_lat)``.
    cos_lat : jnp.ndarray
        Cosine of latitudes, shape ``(n_lat,)``.
    levels : SigmaLevels
        Sigma vertical coordinate.
    planet : PlanetaryConstants
        Planetary constants.
    mean_ps : float
        Global-mean surface pressure [Pa].  Default 1×10⁵.

    Returns
    -------
    jnp.ndarray
        Streamfunction [kg/s], shape ``(n_levels, n_lat)``.
        Positive values indicate clockwise circulation (NH Hadley cell).
    """
    # Cumulative integral of v from top (σ=0) downward
    # ∫₀^σₖ v dσ' ≈ Σ_{j=0}^{k-1} v_j Δσ_j  (midpoint rule)
    v_dsigma = zm_v * levels.dsigma[:, None]
    cumsum = jnp.cumsum(v_dsigma, axis=0)

    prefactor = 2.0 * jnp.pi * planet.radius * mean_ps / planet.gravity
    return prefactor * cos_lat[None, :] * cumsum


# =====================================================================
# Kinetic energy spectrum
# =====================================================================


def compute_ke_spectrum(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    levels: SigmaLevels,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the vertically averaged kinetic energy spectrum.

    For each total wavenumber n, the kinetic energy is:

        E(n) = (1/2) Σₖ Δσₖ · (1/(n(n+1))) · a² ·
               Σₘ (|ζₙᵐ|² + |δₙᵐ|²)

    using the spectral relationship between KE and vorticity/divergence.

    Parameters
    ----------
    state : PrimitiveEquationState
        Atmospheric state in spectral space.
    transform : SpectralTransform
        Spectral transform (provides operator arrays).
    levels : SigmaLevels
        Sigma vertical coordinate (for vertical averaging weights).

    Returns
    -------
    wavenumber : np.ndarray
        Total wavenumber n, shape ``(truncation,)``.  Starts from n=1
        (n=0 has zero KE).
    spectrum : np.ndarray
        Kinetic energy per wavenumber [m²/s²], shape ``(truncation,)``.
    """
    arrays = transform.arrays
    trunc = arrays.truncation
    n_idx = np.asarray(arrays.n_index, dtype=np.int64)
    m_idx = np.asarray(arrays.m_index, dtype=np.int64)
    a2 = arrays.radius**2

    # Vertical average of |ζ|² and |δ|² per spectral coefficient
    vort = np.asarray(state.vorticity)  # (n_levels, n_spectral)
    div = np.asarray(state.divergence)

    dsigma = np.asarray(levels.dsigma)  # (n_levels,)
    vort_sq = np.sum(dsigma[:, None] * np.abs(vort) ** 2, axis=0)
    div_sq = np.sum(dsigma[:, None] * np.abs(div) ** 2, axis=0)

    # Bin by total wavenumber n
    spectrum = np.zeros(trunc + 1)
    for i in range(len(n_idx)):
        n = int(n_idx[i])
        m = int(m_idx[i])
        # Factor of 2 for m > 0 (conjugate pair)
        weight = 2.0 if m > 0 else 1.0
        spectrum[n] += weight * (vort_sq[i] + div_sq[i])

    # Convert to KE: E(n) = a² / (2·n·(n+1)) · spectrum(n)
    nn = np.arange(trunc + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(nn > 0, a2 / (2.0 * nn * (nn + 1)), 0.0)
    spectrum *= factor

    # Return n=1..T (n=0 is always zero)
    return nn[1:], spectrum[1:]
