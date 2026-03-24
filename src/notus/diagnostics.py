"""Conservation diagnostics for the primitive equations.

Computes global integrals of mass, total energy, and angular momentum
from the model state.  These quantities should be approximately conserved
by the adiabatic dynamics (no forcing or dissipation).

All integrals use exact Gaussian quadrature in the horizontal and
midpoint-rule sigma integration in the vertical, matching the accuracy
of the spectral transform and continuity equation.

Formulas (per unit area, then integrated over the sphere):

- **Mass**: M = (1/g) ∫ pₛ dA
- **Total energy**: E = (1/g) ∫ pₛ ∫₀¹ (½|v|² + cₚT + Φₛ) dσ dA
- **Angular momentum**: L = (1/g) ∫ pₛ ∫₀¹ (u + Ωa cosφ) a cosφ dσ dA

References
----------
- Held & Suarez (1994), §4: conservation properties
- Durran, "Numerical Methods for Fluid Dynamics", §8.6
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from notus.constants import PlanetaryConstants
from notus.grid import GaussianGrid
from notus.operators import uv_from_vordiv
from notus.sigma import SigmaLevels
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform


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
    weights = levels.dsigma
    # Expand dsigma to match field dimensions
    n_extra = field.ndim - 1
    for _ in range(n_extra):
        weights = weights[..., None]
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
    n_levels = levels.n_levels
    truncation = grid.truncation
    radius = planet.radius

    # --- Surface pressure on grid ---
    lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
    ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)  # (n_lat, n_lon)

    # --- Mass: M = (1/g) ∫ ps dA  where dA = a²·cosφ·dφ·dλ ---
    a2 = radius**2
    mass = a2 * spherical_integral(ps_grid, grid) / planet.gravity

    # --- Per-level grid-point fields ---
    # Winds: u*cosφ, v*cosφ in spectral → u, v on grid
    cos_lat = grid.cos_lat[:, None]  # (n_lat, 1)

    ke_levels = jnp.zeros((n_levels, grid.n_lat, grid.n_lon))
    ie_levels = jnp.zeros((n_levels, grid.n_lat, grid.n_lon))
    am_levels = jnp.zeros((n_levels, grid.n_lat, grid.n_lon))

    for k in range(n_levels):
        # Winds
        u_cos_spec, v_cos_spec = uv_from_vordiv(
            state.vorticity[k], state.divergence[k], truncation, radius
        )
        u_cos = transform.spectral_to_grid(u_cos_spec)
        v_cos = transform.spectral_to_grid(v_cos_spec)
        u = u_cos / cos_lat  # zonal wind
        v = v_cos / cos_lat  # meridional wind

        # Temperature
        t_grid = transform.spectral_to_grid(state.temperature[k])

        # Kinetic energy: ½(u² + v²)
        ke_levels = ke_levels.at[k].set(0.5 * (u**2 + v**2))

        # Internal energy: cp * T
        ie_levels = ie_levels.at[k].set(planet.specific_heat_cp * t_grid)

        # Angular momentum integrand: (u + Ω·a·cosφ) · a·cosφ
        omega_a_cos = planet.rotation_rate * radius * cos_lat
        am_levels = am_levels.at[k].set((u + omega_a_cos) * radius * cos_lat)

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
