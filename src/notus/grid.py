"""Gaussian grid for spectral-transform models.

A Gaussian grid uses latitudes placed at the roots of the Legendre polynomial
P_N(sin(lat)), where N is the number of latitudes.  This placement ensures
exact quadrature of spherical harmonics up to the truncation wavenumber.

Longitudes are uniformly spaced.

The grid is planet-agnostic — it depends only on the truncation and the
number of grid points, not on any physical constants.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclasses.dataclass(frozen=True, slots=True)
class GaussianGrid:
    """Gaussian grid on the sphere.

    Parameters
    ----------
    truncation : int
        Triangular truncation wavenumber (e.g. 21 for T21).
    n_lat : int
        Number of Gaussian latitudes.  Default: ``truncation * 3 // 2 + 1``
        (the "quadratic" or alias-free grid for truncation T).
        Must be even.
    n_lon : int
        Number of longitudes.  Default: ``2 * n_lat`` (the standard
        choice giving square-ish grid cells near the equator).
    """

    truncation: int
    n_lat: int
    n_lon: int

    # Computed arrays — set via __post_init__
    latitudes: jnp.ndarray = dataclasses.field(repr=False, compare=False)
    longitudes: jnp.ndarray = dataclasses.field(repr=False, compare=False)
    lat_weights: jnp.ndarray = dataclasses.field(repr=False, compare=False)
    cos_lat: jnp.ndarray = dataclasses.field(repr=False, compare=False)
    sin_lat: jnp.ndarray = dataclasses.field(repr=False, compare=False)

    def __init__(self, truncation: int, n_lat: int | None = None, n_lon: int | None = None) -> None:
        if n_lat is None:
            n_lat = _default_n_lat(truncation)
        if n_lon is None:
            n_lon = 2 * n_lat

        if n_lat % 2 != 0:
            msg = f"n_lat must be even, got {n_lat}"
            raise ValueError(msg)
        if n_lon < 1:
            msg = f"n_lon must be positive, got {n_lon}"
            raise ValueError(msg)

        # Gaussian latitudes: roots of P_{n_lat}(x) via numpy (exact),
        # then convert to JAX arrays.
        sin_lat_np, weights_np = leggauss(n_lat)

        # leggauss returns roots in ascending order of x = sin(lat),
        # i.e. south-to-north.  Reverse to north-to-south (90° → -90°).
        sin_lat_np = sin_lat_np[::-1]
        weights_np = weights_np[::-1]

        latitudes_np = np.arcsin(sin_lat_np)
        cos_lat_np = np.cos(latitudes_np)

        # Longitudes: [0, 2π) uniformly spaced.
        longitudes_np = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

        # Store as JAX arrays (float64 for accuracy in transforms).
        object.__setattr__(self, "truncation", truncation)
        object.__setattr__(self, "n_lat", n_lat)
        object.__setattr__(self, "n_lon", n_lon)
        object.__setattr__(self, "latitudes", jnp.array(latitudes_np))
        object.__setattr__(self, "longitudes", jnp.array(longitudes_np))
        object.__setattr__(self, "lat_weights", jnp.array(weights_np))
        object.__setattr__(self, "cos_lat", jnp.array(cos_lat_np))
        object.__setattr__(self, "sin_lat", jnp.array(sin_lat_np))

    @property
    def n_total_wavenumbers(self) -> int:
        """Maximum total wavenumber n (inclusive)."""
        return self.truncation

    @property
    def n_spectral_coeffs(self) -> int:
        """Number of complex spectral coefficients in the lower triangle.

        For triangular truncation T, coefficients exist for
        0 ≤ m ≤ T and m ≤ n ≤ T, giving (T+1)(T+2)/2 complex values.
        """
        t = self.truncation
        return (t + 1) * (t + 2) // 2

    @property
    def dlon(self) -> float:
        """Longitude spacing [rad]."""
        return 2.0 * math.pi / self.n_lon

    @property
    def latitudes_deg(self) -> jnp.ndarray:
        """Gaussian latitudes in degrees, shape ``(n_lat,)``."""
        return jnp.degrees(self.latitudes)

    def spectral_index(self, m: int, n: int) -> int:
        """Linear index into the lower-triangular spectral array.

        Convention: coefficients stored column-major by zonal wavenumber m,
        then by total wavenumber n from m to T.

            index(m, n) = m * (T + 1) - m * (m - 1) // 2 + (n - m)

        where T = truncation.
        """
        t = self.truncation
        return m * (t + 1) - m * (m - 1) // 2 + (n - m)


def _default_n_lat(truncation: int) -> int:
    """Compute the default number of latitudes for quadratic (alias-free) grid.

    The standard choice is ``(3T + 1) / 2`` rounded up to the next even number,
    ensuring exact quadrature of products of three spherical harmonics (the
    nonlinear advection terms).
    """
    n = math.ceil((3 * truncation + 1) / 2)
    if n % 2 != 0:
        n += 1
    return n
