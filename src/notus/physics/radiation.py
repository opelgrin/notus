"""Gray radiation scheme following Frierson et al. (2006).

Two-stream gray longwave radiation with latitude-dependent optical depth,
and Beer-Lambert shortwave absorption.  All functions are JIT-compatible.
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
    sigma_half: jnp.ndarray,
    dsigma: jnp.ndarray,
    sin_lat: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    gravity: float,
    specific_heat_cp: float,
    *,
    tau_equator: float,
    tau_pole: float,
    linear_fraction: float,
    alpha: float,
) -> jnp.ndarray:
    """Compute longwave radiative heating rate using two-stream gray model.

    Upward and downward fluxes are computed at half-level interfaces via
    ``jax.lax.scan``, then the net flux divergence yields the heating rate.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels, shape ``(n_levels, n_lat, n_lon)``.
    surface_temperature : jnp.ndarray
        Surface temperature, shape ``(n_lat,)``.
    sigma_half : jnp.ndarray
        Half-level σ values, shape ``(n_levels+1,)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    sin_lat : jnp.ndarray
        Sine of latitude, shape ``(n_lat,)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    tau_equator : float
        Equatorial longwave optical depth.
    tau_pole : float
        Polar longwave optical depth.
    linear_fraction : float
        Fraction of optical depth with linear pressure dependence.
    alpha : float
        Pressure exponent for the nonlinear part.

    Returns
    -------
    jnp.ndarray
        Longwave heating rate [K/s], shape ``(n_levels, n_lat, n_lon)``.
    """
    # Optical depth at half-levels: (n_levels+1, n_lat)
    tau_half = longwave_optical_depth(
        sigma_half, sin_lat,
        tau_equator=tau_equator, tau_pole=tau_pole,
        linear_fraction=linear_fraction, alpha=alpha,
    )

    # Layer optical thickness and transmissivity: (n_levels, n_lat)
    dtau = tau_half[1:] - tau_half[:-1]
    transmissivity = jnp.exp(-dtau)

    # Blackbody emission at each full level: (n_levels, n_lat, n_lon)
    bb = STEFAN_BOLTZMANN * temperature**4
    # Surface blackbody: (n_lat, n_lon) — broadcast SST over longitude
    n_lat, n_lon = surface_pressure.shape
    bb_surface = jnp.broadcast_to(
        STEFAN_BOLTZMANN * surface_temperature[:, None] ** 4, (n_lat, n_lon),
    )

    # Upward flux: scan from surface (bottom) to TOA (top).
    # Surface boundary: upward flux equals blackbody emission at surface temperature.
    # Each layer absorbs and re-emits according to its transmissivity.

    # Reverse arrays so scan goes from bottom level to top level
    trans_rev = transmissivity[::-1]  # (n_levels, n_lat)
    bb_rev = bb[::-1]  # (n_levels, n_lat, n_lon)

    def _upward_step(
        f_up: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trans_k, bb_k = layer  # (n_lat,), (n_lat, n_lon)
        f_up_new = f_up * trans_k[:, None] + bb_k * (1.0 - trans_k[:, None])
        return f_up_new, f_up_new

    _, f_up_interfaces_rev = jax.lax.scan(
        _upward_step, bb_surface, (trans_rev, bb_rev),
    )
    # f_up_interfaces_rev: (n_levels, n_lat, n_lon) — fluxes at interfaces
    # from surface-1 to TOA (reversed order)
    # Reverse back and prepend surface flux, append TOA flux
    f_up_inner = f_up_interfaces_rev[::-1]  # (n_levels, n_lat, n_lon)
    # f_up_inner[k] = flux at interface k+1/2 (above level k)
    # We need n_levels+1 interface values: surface, then n_levels interfaces
    f_up = jnp.concatenate(
        [f_up_inner, bb_surface[None, :, :]], axis=0,
    )  # (n_levels+1, n_lat, n_lon) — index 0 = TOA, index n_levels = surface

    # --- Downward flux: scan from TOA (top) to surface (bottom) ---
    # Boundary condition: F_down at TOA = 0

    def _downward_step(
        f_down: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trans_k, bb_k = layer  # (n_lat,), (n_lat, n_lon)
        f_down_new = f_down * trans_k[:, None] + bb_k * (1.0 - trans_k[:, None])
        return f_down_new, f_down_new

    f_down_toa = jnp.zeros((n_lat, n_lon))
    _, f_down_interfaces = jax.lax.scan(
        _downward_step, f_down_toa, (transmissivity, bb),
    )
    # f_down_interfaces[k] = flux at interface k+3/2 (below level k)
    # We need n_levels+1 interface values: TOA, then n_levels interfaces
    f_down = jnp.concatenate(
        [f_down_toa[None, :, :], f_down_interfaces], axis=0,
    )  # (n_levels+1, n_lat, n_lon) — index 0 = TOA, index n_levels = surface

    # --- Net flux and heating rate ---
    f_net = f_up - f_down  # (n_levels+1, n_lat, n_lon)

    # Heating rate from net flux divergence.
    # The minus sign arises because p increases downward.
    df_net = f_net[:-1] - f_net[1:]  # (n_levels, n_lat, n_lon)
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]  # (n_levels, n_lat, n_lon)

    return -gravity / specific_heat_cp * df_net / dp


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
) -> jnp.ndarray:
    """Compute shortwave heating rate via Beer-Lambert absorption.

    Zenith-angle-averaged insolation following Frierson (2006)::

        S(φ) = S₀/4 · [1 + δ_s · (1 − 3 sin²φ) / 4]

    No diurnal cycle (Phase 7). Downward-only (no surface reflection).

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

    Returns
    -------
    jnp.ndarray
        Shortwave heating rate [K/s], shape ``(n_levels, n_lat, n_lon)``.
    """
    # Insolation profile: (n_lat,)
    insolation = solar_constant / 4.0 * (
        1.0 + delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0
    )

    # SW optical depth at half-levels: (n_levels+1, n_lat)
    tau_sw_half = sw_tau_0 * sigma_half[:, None] ** sw_exponent

    # Downward SW flux at half-levels: (n_levels+1, n_lat)
    f_sw = insolation[None, :] * jnp.exp(-tau_sw_half)

    # Flux absorbed in each layer: (n_levels, n_lat)
    f_absorbed = f_sw[:-1] - f_sw[1:]

    # Heating rate: Q = g * F_absorbed / (cp * dp)
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]  # (n_levels, n_lat, n_lon)

    return gravity / specific_heat_cp * f_absorbed[:, :, None] / dp
