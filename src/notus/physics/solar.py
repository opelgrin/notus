"""Solar geometry for seasonal insolation.

Orbital parameters, solar declination, Earth-Sun distance factor,
and daily-mean insolation.  All functions are JIT-compatible.

References
----------
Berger, A. (1978). Long-term variations of daily insolation and
Quaternary climatic changes. J. Atmos. Sci., 35(12), 2362-2367.

Hartmann, D. L. (2016). Global Physical Climatology (2nd ed.), Ch. 2.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, slots=True)
class OrbitalParameters:
    """Orbital parameters for computing seasonal insolation.

    Parameters
    ----------
    eccentricity : float
        Orbital eccentricity (0 = circular).
    obliquity : float
        Axial tilt [radians].
    longitude_perihelion : float
        Longitude of perihelion relative to vernal equinox [radians].
    days_per_year : float
        Orbital period [days].
    """

    eccentricity: float = 0.0167086
    obliquity: float = 23.44 * math.pi / 180.0
    longitude_perihelion: float = 282.9 * math.pi / 180.0
    days_per_year: float = 365.25


EARTH_ORBIT = OrbitalParameters()


def solar_declination(
    day_of_year: jnp.ndarray,
    orbital: OrbitalParameters,
) -> jnp.ndarray:
    """Compute solar declination angle.

    The declination is the latitude at which the sun is directly
    overhead at solar noon.  Varies from ``-obliquity`` to ``+obliquity``
    over the year.

    Parameters
    ----------
    day_of_year : jnp.ndarray
        Day of year (0 = vernal equinox). Scalar or array.
    orbital : OrbitalParameters
        Orbital configuration.

    Returns
    -------
    jnp.ndarray
        Solar declination [radians], same shape as ``day_of_year``.
    """
    # True solar longitude from vernal equinox (d=0 at equinox)
    lambda_s = 2.0 * jnp.pi * day_of_year / orbital.days_per_year
    return jnp.arcsin(jnp.sin(orbital.obliquity) * jnp.sin(lambda_s))


def earth_sun_distance_factor(
    day_of_year: jnp.ndarray,
    orbital: OrbitalParameters,
) -> jnp.ndarray:
    """Compute (a/r)^2 factor for Earth-Sun distance variation.

    Modulates the solar constant due to orbital eccentricity.
    Returns 1.0 for a circular orbit.

    Parameters
    ----------
    day_of_year : jnp.ndarray
        Day of year (0 = vernal equinox). Scalar or array.
    orbital : OrbitalParameters
        Orbital configuration.

    Returns
    -------
    jnp.ndarray
        Distance factor (a/r)^2, same shape as ``day_of_year``.
    """
    e = orbital.eccentricity
    if e == 0.0:
        return jnp.ones_like(day_of_year)

    # For small eccentricity, first-order approximation:
    # (a/r)^2 ≈ 1 + 2e·cos(λ - ω)
    # where λ is the true solar longitude from vernal equinox
    # and ω is the longitude of perihelion
    lambda_s = 2.0 * jnp.pi * day_of_year / orbital.days_per_year
    return 1.0 + 2.0 * e * jnp.cos(lambda_s - orbital.longitude_perihelion)


def daily_mean_insolation(
    sin_lat: jnp.ndarray,
    day_of_year: jnp.ndarray,
    solar_constant: float,
    orbital: OrbitalParameters,
) -> jnp.ndarray:
    """Compute daily-mean top-of-atmosphere insolation.

    Analytically integrates the cosine of the solar zenith angle over
    the diurnal cycle, accounting for polar day and polar night::

        S(φ, d) = S₀/π · (a/r)² · [h₀·sinφ·sinδ + cosφ·cosδ·sin(h₀)]

    where ``h₀`` is the sunrise hour angle::

        h₀ = arccos(-tan(φ)·tan(δ))

    clamped to ``[0, π]`` for polar night and polar day.

    Parameters
    ----------
    sin_lat : jnp.ndarray
        Sine of latitude, shape ``(n_lat,)``.
    day_of_year : jnp.ndarray
        Day of year (0 = vernal equinox). Scalar.
    solar_constant : float
        Total solar irradiance S₀ [W/m²].
    orbital : OrbitalParameters
        Orbital configuration.

    Returns
    -------
    jnp.ndarray
        Daily-mean insolation [W/m²], shape ``(n_lat,)``.
    """
    decl = solar_declination(day_of_year, orbital)
    dist_factor = earth_sun_distance_factor(day_of_year, orbital)

    sin_decl = jnp.sin(decl)
    cos_decl = jnp.cos(decl)

    cos_lat = jnp.sqrt(1.0 - sin_lat**2)
    # Avoid division by zero at poles
    cos_lat_safe = jnp.maximum(cos_lat, 1.0e-30)
    tan_lat = sin_lat / cos_lat_safe

    tan_decl = sin_decl / jnp.maximum(cos_decl, 1.0e-30)

    # Sunrise hour angle: h0 = arccos(-tan(lat)*tan(decl))
    # Clamped: cos_h0 < -1 → polar day (h0 = pi), cos_h0 > 1 → polar night (h0 = 0)
    cos_h0 = -tan_lat * tan_decl
    h0 = jnp.where(
        cos_h0 <= -1.0,
        jnp.pi,
        jnp.where(cos_h0 >= 1.0, 0.0, jnp.arccos(cos_h0)),
    )

    # Daily-mean insolation: S0/pi * (a/r)^2 * [h0*sin(lat)*sin(decl) + cos(lat)*cos(decl)*sin(h0)]
    insolation = (
        solar_constant
        / jnp.pi
        * dist_factor
        * (h0 * sin_lat * sin_decl + cos_lat * cos_decl * jnp.sin(h0))
    )

    return jnp.maximum(insolation, 0.0)
