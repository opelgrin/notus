"""Idealized topography generators for the spectral GCM.

Each function constructs an analytic height field on the Gaussian grid,
converts to surface geopotential Φ_s = g·z_s, and returns spectral
coefficients ready to pass into :func:`~notus.timestepping.imex.build_pe_stepper`.

References
----------
Jablonowski, C. et al. (2008). Idealized test cases for the dynamical
core of atmospheric general circulation models: A proposal for the
NCAR ASP 2008 summer colloquium.

Williamson, D. L. et al. (1992). A standard test set for numerical
approximations to the shallow water equations in spherical geometry.
J. Comp. Phys., 102, 211-224.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.transforms import SpectralTransform


def gaussian_mountain(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    *,
    height: float = 2000.0,
    center_lat: float = np.pi / 4,
    center_lon: float = np.pi / 2,
    half_width: float = np.pi / 9,
) -> jnp.ndarray:
    """Isolated Gaussian mountain.

    z(φ, λ) = h₀ · exp(−((φ − φ₀)² + (λ − λ₀)²) / σ²)

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (provides gravity).
    height : float
        Peak height [m].
    center_lat : float
        Latitude of the mountain centre [rad].
    center_lon : float
        Longitude of the mountain centre [rad].
    half_width : float
        Gaussian half-width σ [rad].

    Returns
    -------
    jnp.ndarray
        Surface geopotential g·z_s in spectral space, shape ``(n_spectral,)``.
    """
    grid = transform.grid
    lat = np.asarray(grid.latitudes)[:, None]  # (n_lat, 1)
    lon = np.asarray(grid.longitudes)[None, :]  # (1, n_lon)

    dlat = lat - center_lat
    dlon = lon - center_lon
    # Wrap longitude difference to [-π, π]
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi

    z = height * np.exp(-(dlat**2 + dlon**2) / half_width**2)
    phi_s = planet.gravity * z
    return transform.grid_to_spectral(jnp.array(phi_s))


def zonal_ridge(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    *,
    height: float = 2000.0,
    center_lat: float = np.pi / 4,
    half_width: float = np.pi / 9,
) -> jnp.ndarray:
    """Zonally symmetric Gaussian ridge.

    z(φ) = h₀ · exp(−(φ − φ₀)² / σ²)

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (provides gravity).
    height : float
        Peak height [m].
    center_lat : float
        Centre latitude [rad].
    half_width : float
        Gaussian half-width σ [rad].

    Returns
    -------
    jnp.ndarray
        Surface geopotential g·z_s in spectral space, shape ``(n_spectral,)``.
    """
    grid = transform.grid
    lat = np.asarray(grid.latitudes)[:, None]  # (n_lat, 1)

    z = height * np.exp(-((lat - center_lat) ** 2) / half_width**2)
    z_grid = np.broadcast_to(z, (grid.n_lat, grid.n_lon))
    phi_s = planet.gravity * z_grid
    return transform.grid_to_spectral(jnp.array(phi_s))


def sinusoidal_mountains(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    *,
    height: float = 2000.0,
    zonal_wavenumber: int = 2,
) -> jnp.ndarray:
    """Sinusoidal mountain chain for stationary Rossby wave tests.

    z(φ, λ) = h₀ · cos²(φ) · cos(k·λ)

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (provides gravity).
    height : float
        Peak height [m].
    zonal_wavenumber : int
        Zonal wavenumber k of the mountain pattern.

    Returns
    -------
    jnp.ndarray
        Surface geopotential g·z_s in spectral space, shape ``(n_spectral,)``.
    """
    grid = transform.grid
    cos_lat = np.asarray(grid.cos_lat)[:, None]  # (n_lat, 1)
    lon = np.asarray(grid.longitudes)[None, :]  # (1, n_lon)

    z = height * cos_lat**2 * np.cos(zonal_wavenumber * lon)
    phi_s = planet.gravity * z
    return transform.grid_to_spectral(jnp.array(phi_s))
