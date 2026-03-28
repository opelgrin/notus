"""Radiation schemes for idealized GCM experiments.

Longwave:
- Frierson (2006) gray scheme: prescribed tau(sigma, latitude)
- Byrne (Isca) scheme: humidity-dependent tau for water vapor feedback

Shortwave:
- Beer-Lambert absorption with prescribed or humidity-dependent optical depth

All functions are JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


STEFAN_BOLTZMANN: float = 5.670374419e-8  # [W/(m² K⁴)]


# ---------------------------------------------------------------------------
# Longwave radiation
# ---------------------------------------------------------------------------


def longwave_optical_depth(
    sigma_half: jnp.ndarray,
    sin_lat: jnp.ndarray,
    *,
    tau_equator: float,
    tau_pole: float,
    linear_fraction: float,
    alpha: float,
) -> jnp.ndarray:
    """Compute longwave optical depth at half-levels.

    Frierson et al. (2006) mixed linear + power-law pressure dependence::

        tau_0(phi) = tau_e + (tau_p - tau_e) sin^2(phi)
        tau(sigma, phi) = tau_0 * [f_l * sigma + (1 - f_l) * sigma^alpha]

    where ``f_l`` is ``linear_fraction``.

    Parameters
    ----------
    sigma_half : jnp.ndarray
        Half-level sigma values (0 at top, 1 at surface), shape ``(n_levels+1,)``.
    sin_lat : jnp.ndarray
        Sine of latitude, shape ``(n_lat,)``.
    tau_equator : float
        Equatorial longwave optical depth.
    tau_pole : float
        Polar longwave optical depth.
    linear_fraction : float
        Fraction of optical depth with linear pressure dependence (0-1).
    alpha : float
        Pressure exponent for the nonlinear part.

    Returns
    -------
    jnp.ndarray
        Optical depth at half-levels, shape ``(n_levels+1, n_lat)``.
    """
    tau_0 = tau_equator + (tau_pole - tau_equator) * sin_lat**2  # (n_lat,)
    sigma = sigma_half[:, None]  # (n_levels+1, 1)
    pressure_profile = linear_fraction * sigma + (1.0 - linear_fraction) * sigma**alpha
    return tau_0[None, :] * pressure_profile  # (n_levels+1, n_lat)


def longwave_heating(
    temperature: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    tau_half: jnp.ndarray,
    dsigma: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    gravity: float,
    specific_heat_cp: float,
) -> jnp.ndarray:
    """Compute longwave radiative heating rate using two-stream model.

    Upward and downward fluxes are computed at half-level interfaces via
    ``jax.lax.scan``, then the net flux divergence yields the heating rate.

    The optical depth ``tau_half`` can be either 2-D (Frierson gray scheme,
    latitude-only) or 3-D (Byrne scheme, humidity-dependent).

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels, shape ``(n_levels, n_lat, n_lon)``.
    surface_temperature : jnp.ndarray
        Surface temperature, shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    tau_half : jnp.ndarray
        Longwave optical depth at half-levels, shape
        ``(n_levels+1, n_lat)`` or ``(n_levels+1, n_lat, n_lon)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].

    Returns
    -------
    jnp.ndarray
        Longwave heating rate [K/s], shape ``(n_levels, n_lat, n_lon)``.
    """
    n_lat, n_lon = surface_pressure.shape

    # Ensure tau_half is 3-D: (n_levels+1, n_lat, n_lon)
    if tau_half.ndim == 2:  # noqa: PLR2004
        tau_half = jnp.broadcast_to(
            tau_half[:, :, None],
            (*tau_half.shape, n_lon),
        )

    # Layer optical thickness and transmissivity: (n_levels, n_lat, n_lon)
    dtau = tau_half[1:] - tau_half[:-1]
    transmissivity = jnp.exp(-dtau)

    # Blackbody emission at each full level: (n_levels, n_lat, n_lon)
    bb = STEFAN_BOLTZMANN * temperature**4
    # Surface blackbody: (n_lat, n_lon)
    if surface_temperature.ndim == 1:
        surface_temperature = surface_temperature[:, None]
    bb_surface = jnp.broadcast_to(
        STEFAN_BOLTZMANN * surface_temperature**4,
        (n_lat, n_lon),
    )

    # Upward flux: scan from surface (bottom) to TOA (top).
    trans_rev = transmissivity[::-1]  # (n_levels, n_lat, n_lon)
    bb_rev = bb[::-1]  # (n_levels, n_lat, n_lon)

    def _upward_step(
        f_up: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trans_k, bb_k = layer  # (n_lat, n_lon), (n_lat, n_lon)
        f_up_new = f_up * trans_k + bb_k * (1.0 - trans_k)
        return f_up_new, f_up_new

    _, f_up_interfaces_rev = jax.lax.scan(
        _upward_step,
        bb_surface,
        (trans_rev, bb_rev),
    )
    f_up_inner = f_up_interfaces_rev[::-1]  # (n_levels, n_lat, n_lon)
    f_up = jnp.concatenate(
        [f_up_inner, bb_surface[None, :, :]],
        axis=0,
    )  # (n_levels+1, n_lat, n_lon)

    # --- Downward flux: scan from TOA (top) to surface (bottom) ---
    def _downward_step(
        f_down: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trans_k, bb_k = layer  # (n_lat, n_lon), (n_lat, n_lon)
        f_down_new = f_down * trans_k + bb_k * (1.0 - trans_k)
        return f_down_new, f_down_new

    f_down_toa = jnp.zeros((n_lat, n_lon))
    _, f_down_interfaces = jax.lax.scan(
        _downward_step,
        f_down_toa,
        (transmissivity, bb),
    )
    f_down = jnp.concatenate(
        [f_down_toa[None, :, :], f_down_interfaces],
        axis=0,
    )  # (n_levels+1, n_lat, n_lon)

    # --- Net flux and heating rate ---
    f_net = f_up - f_down  # (n_levels+1, n_lat, n_lon)
    df_net = f_net[:-1] - f_net[1:]  # (n_levels, n_lat, n_lon)
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]

    return -gravity / specific_heat_cp * df_net / dp


def byrne_longwave_optical_depth(
    dsigma: jnp.ndarray,
    humidity: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    reference_pressure: float,
    *,
    byrne_a: float = 0.8678,
    byrne_b: float = 1997.9,
) -> jnp.ndarray:
    """Compute humidity-dependent longwave optical depth (Byrne/Isca).

    Optical depth varies with the column water vapor content, providing
    a self-consistent water vapor feedback::

        dτ/d(p/p₀) = a + b · q

    where *a* captures well-mixed gas absorption and *b* captures water
    vapor absorption.  Integrated in sigma coordinates as::

        Δτ_k = (a + b · q_k) · (ps / p₀) · Δσ_k

    Parameters
    ----------
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    humidity : jnp.ndarray
        Specific humidity at full levels [kg/kg],
        shape ``(n_levels, n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    reference_pressure : float
        Reference pressure p₀ [Pa] (typically 1e5).
    byrne_a : float
        Well-mixed gas absorption coefficient (default 0.8678).
    byrne_b : float
        Water vapor absorption coefficient (default 1997.9).

    Returns
    -------
    jnp.ndarray
        Optical depth at half-levels, shape ``(n_levels+1, n_lat, n_lon)``.
    """
    # Optical depth increment per layer: (n_levels, n_lat, n_lon)
    ps_ratio = surface_pressure[None, :, :] / reference_pressure
    dtau = (byrne_a + byrne_b * humidity) * dsigma[:, None, None] * ps_ratio

    # Cumulative from TOA (tau=0 at top)
    tau_cumsum = jnp.cumsum(dtau, axis=0)  # (n_levels, n_lat, n_lon)

    # Prepend zero at TOA
    tau_toa = jnp.zeros_like(tau_cumsum[:1])
    return jnp.concatenate([tau_toa, tau_cumsum], axis=0)  # (n_levels+1, ...)


# ---------------------------------------------------------------------------
# Shortwave radiation
# ---------------------------------------------------------------------------


def shortwave_heating(
    sigma_half: jnp.ndarray,
    dsigma: jnp.ndarray,
    sin_lat: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    solar_constant: float,
    gravity: float,
    specific_heat_cp: float,
    *,
    sw_tau_0: float,
    sw_exponent: float,
    delta_s: float,
    insolation: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute shortwave heating rate via Beer-Lambert absorption.

    When ``insolation`` is not provided, uses the Frierson (2006)
    zenith-angle-averaged formula::

        S(φ) = S₀/4 · [1 + δ_s · (1 − 3 sin²φ) / 4]

    When ``insolation`` is provided (e.g. from ``daily_mean_insolation``),
    it is used directly, enabling seasonal forcing.

    Downward-only (no surface reflection).

    Parameters
    ----------
    sigma_half : jnp.ndarray
        Half-level σ values, shape ``(n_levels+1,)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    sin_lat : jnp.ndarray
        Sine of latitude, shape ``(n_lat,)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    solar_constant : float
        Total solar irradiance S₀ [W/m²].
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    sw_tau_0 : float
        Shortwave optical depth.
    sw_exponent : float
        Shortwave pressure exponent.
    delta_s : float
        Insolation distribution parameter.
    insolation : jnp.ndarray or None
        Pre-computed TOA insolation [W/m²], shape ``(n_lat,)``.
        When ``None``, uses the fixed Frierson profile.

    Returns
    -------
    jnp.ndarray
        Shortwave heating rate [K/s], shape ``(n_levels, n_lat, n_lon)``.
    """
    # Insolation profile: (n_lat,)
    if insolation is None:
        insolation = solar_constant / 4.0 * (1.0 + delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)

    # SW optical depth at half-levels: (n_levels+1, n_lat)
    tau_sw_half = sw_tau_0 * sigma_half[:, None] ** sw_exponent

    # Downward SW flux at half-levels: (n_levels+1, n_lat)
    f_sw = insolation[None, :] * jnp.exp(-tau_sw_half)

    # Flux absorbed in each layer: (n_levels, n_lat)
    f_absorbed = f_sw[:-1] - f_sw[1:]

    # Heating rate: Q = g * F_absorbed / (cp * dp)
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]  # (n_levels, n_lat, n_lon)

    return gravity / specific_heat_cp * f_absorbed[:, :, None] / dp
