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
from notus.operators.arrays import OperatorArrays
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


def _lanczos_sigma(arrays: OperatorArrays) -> jnp.ndarray:
    """Lanczos σ-factor: sin(π·n/T) / (π·n/T), with σ(0) = 1.

    Suppresses Gibbs ringing by tapering spectral coefficients near
    the truncation limit.

    Parameters
    ----------
    arrays : OperatorArrays
        Pre-computed operator arrays (provides n_index and truncation).

    Returns
    -------
    jnp.ndarray
        Multiplicative weights, shape ``(n_spectral,)``.
    """
    x = jnp.pi * arrays.n_index / (arrays.truncation + 1)
    # n=0 gives x=0; use safe division and patch with the analytic limit sinc(0)=1
    safe_x = jnp.where(arrays.n_index > 0, x, 1.0)
    return jnp.where(arrays.n_index > 0, jnp.sin(safe_x) / safe_x, 1.0)


def smooth_orography(
    surface_geopotential: jnp.ndarray,
    arrays: OperatorArrays,
    *,
    method: str = "lanczos",
    order: int = 1,
) -> jnp.ndarray:
    """Smooth spectral orography to reduce Gibbs ringing.

    Applies a multiplicative spectral taper to surface geopotential
    coefficients.  This should be called once on the initial
    ``surface_geopotential`` before passing it to the time stepper.

    Available methods:

    - ``"lanczos"``: σ-factor = sin(π·n/T) / (π·n/T), raised to *order*.
      Order 1 is the classic Lanczos smoothing; order 2 gives stronger
      damping near truncation.
    - ``"exponential"``: exp(−κ·(n/T)²) with κ = 2·order.  Provides
      Gaussian-like tapering that preserves large scales.

    Parameters
    ----------
    surface_geopotential : jnp.ndarray
        Spectral coefficients of g·z_s, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.
    method : str
        Smoothing method: ``"lanczos"`` or ``"exponential"``.
    order : int
        Controls the strength of damping (higher = more aggressive).

    Returns
    -------
    jnp.ndarray
        Smoothed spectral coefficients, same shape as input.
    """
    if method == "lanczos":
        weights = _lanczos_sigma(arrays) ** order
    elif method == "exponential":
        k = arrays.n_index / (arrays.truncation + 1)
        weights = jnp.exp(-2.0 * order * k**2)
    else:
        raise ValueError(f"Unknown smoothing method: {method!r}")
    return surface_geopotential * weights
