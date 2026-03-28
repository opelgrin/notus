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
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    tuple[jnp.ndarray, jnp.ndarray]
        ``(heating_rate, lw_down_sfc)`` — longwave heating rate [K/s]
        shape ``(n_levels, n_lat, n_lon)`` and downward LW flux at the
        surface [W/m²] shape ``(n_lat, n_lon)``.
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

    heating_rate = -gravity / specific_heat_cp * df_net / dp

    # Downward LW flux at the surface (last half-level interface)
    lw_down_sfc = f_down[-1]  # (n_lat, n_lon)

    return heating_rate, lw_down_sfc


def lw_down_surface(
    temperature: jnp.ndarray,
    tau_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute downward longwave flux at the surface from the two-stream model.

    Performs only the downward scan (half the cost of ``longwave_heating``).
    Useful for surface energy balance in the slab ocean without recomputing
    the full heating rate.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels, shape ``(n_levels, n_lat, n_lon)``.
    tau_half : jnp.ndarray
        Longwave optical depth at half-levels, shape
        ``(n_levels+1, n_lat)`` or ``(n_levels+1, n_lat, n_lon)``.

    Returns
    -------
    jnp.ndarray
        Downward LW flux at the surface [W/m²], shape ``(n_lat, n_lon)``.
    """
    n_lon = temperature.shape[-1]

    # Ensure tau_half is 3-D
    if tau_half.ndim == 2:  # noqa: PLR2004
        tau_half = jnp.broadcast_to(
            tau_half[:, :, None],
            (*tau_half.shape, n_lon),
        )

    # Layer transmissivity
    dtau = tau_half[1:] - tau_half[:-1]
    transmissivity = jnp.exp(-dtau)

    # Blackbody emission at each full level
    bb = STEFAN_BOLTZMANN * temperature**4

    # Downward scan from TOA to surface
    def _downward_step(
        f_down: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        trans_k, bb_k = layer
        f_down_new = f_down * trans_k + bb_k * (1.0 - trans_k)
        return f_down_new, None

    n_lat = temperature.shape[1]
    f_down_toa = jnp.zeros((n_lat, n_lon))
    f_down_sfc, _ = jax.lax.scan(
        _downward_step,
        f_down_toa,
        (transmissivity, bb),
    )

    return f_down_sfc


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
    tau_sw_half: jnp.ndarray | None = None,
    surface_albedo: float | jnp.ndarray = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute shortwave heating rate and surface downward flux.

    Uses Beer-Lambert absorption for the downward pass.  When
    ``surface_albedo`` is nonzero, the reflected upward flux undergoes
    a second absorption pass through the atmosphere.

    When ``insolation`` is not provided, uses the Frierson (2006)
    zenith-angle-averaged formula::

        S(φ) = S₀/4 · [1 + δ_s · (1 − 3 sin²φ) / 4]

    When ``insolation`` is provided (e.g. from ``daily_mean_insolation``),
    it is used directly, enabling seasonal forcing.

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
        Shortwave optical depth (used only when ``tau_sw_half`` is None).
    sw_exponent : float
        Shortwave pressure exponent (used only when ``tau_sw_half`` is None).
    delta_s : float
        Insolation distribution parameter.
    insolation : jnp.ndarray or None
        Pre-computed TOA insolation [W/m²], shape ``(n_lat,)``.
        When ``None``, uses the fixed Frierson profile.
    tau_sw_half : jnp.ndarray or None
        Pre-computed SW optical depth at half-levels, shape
        ``(n_levels+1, n_lat)`` or ``(n_levels+1, n_lat, n_lon)``.
        When provided, overrides ``sw_tau_0``/``sw_exponent``.
    surface_albedo : float or jnp.ndarray
        Surface albedo (0-1) for reflected-beam absorption.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(heating_rate, sw_down_sfc)`` — shortwave heating rate [K/s]
        shape ``(n_levels, n_lat, n_lon)`` and downward SW flux at the
        surface [W/m²] shape ``(n_lat, n_lon)``.
    """
    # Insolation profile: (n_lat,)
    if insolation is None:
        insolation = solar_constant / 4.0 * (1.0 + delta_s * (1.0 - 3.0 * sin_lat**2) / 4.0)

    # SW optical depth at half-levels
    if tau_sw_half is None:
        tau_sw_half = sw_tau_0 * sigma_half[:, None] ** sw_exponent

    n_lon = surface_pressure.shape[-1]

    # Ensure tau_sw_half is 3-D: (n_levels+1, n_lat, n_lon)
    if tau_sw_half.ndim == 2:  # noqa: PLR2004
        tau_sw_half_3d = jnp.broadcast_to(
            tau_sw_half[:, :, None],
            (*tau_sw_half.shape, n_lon),
        )
    else:
        tau_sw_half_3d = tau_sw_half

    # Downward SW flux at half-levels: (n_levels+1, n_lat, n_lon)
    f_sw_down = insolation[None, :, None] * jnp.exp(-tau_sw_half_3d)

    # SW reaching the surface (before reflection)
    sw_down_sfc = f_sw_down[-1]  # (n_lat, n_lon)

    # Downward absorption per layer: (n_levels, n_lat, n_lon)
    f_absorbed_down = f_sw_down[:-1] - f_sw_down[1:]

    # Reflected upward beam: surface reflects, then Beer-Lambert back up.
    # tau from surface to level k = tau_surface - tau_k (reversed)
    tau_surface = tau_sw_half_3d[-1:]  # (1, n_lat, n_lon)
    tau_up = tau_surface - tau_sw_half_3d  # (n_levels+1, n_lat, n_lon)
    f_sw_up = sw_down_sfc[None, :, :] * surface_albedo * jnp.exp(-tau_up)

    # Upward absorption per layer (absorbed going from bottom to top)
    f_absorbed_up = f_sw_up[1:] - f_sw_up[:-1]  # (n_levels, n_lat, n_lon)

    # Total absorbed = downward + upward
    f_absorbed = f_absorbed_down + f_absorbed_up

    # Heating rate: Q = g * F_absorbed / (cp * dp)
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]  # (n_levels, n_lat, n_lon)

    heating_rate = gravity / specific_heat_cp * f_absorbed / dp

    return heating_rate, sw_down_sfc


def byrne_shortwave_optical_depth(
    dsigma: jnp.ndarray,
    humidity: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    reference_pressure: float,
    *,
    sw_tau_0: float = 0.22,
    byrne_sw_a: float = 0.0,
    byrne_sw_b: float = 0.2,
) -> jnp.ndarray:
    """Compute humidity-dependent shortwave optical depth.

    Analogous to :func:`byrne_longwave_optical_depth` but for shortwave
    near-IR water vapor absorption::

        dτ_sw/d(p/p₀) = a_sw + b_sw · q

    The well-mixed component ``a_sw`` captures non-humidity-dependent
    absorption (e.g. ozone), while ``b_sw`` captures near-IR H₂O bands.
    The total column optical depth in a dry atmosphere equals
    ``a_sw * (ps/p₀)``; to match the Frierson convention, the default
    ``a_sw`` is zero and a separate ``sw_tau_0`` sets a floor.

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
    sw_tau_0 : float
        Base shortwave optical depth (Frierson-style floor).
    byrne_sw_a : float
        Well-mixed gas SW absorption coefficient (default 0.0).
    byrne_sw_b : float
        Water vapor SW absorption coefficient (default 0.2).

    Returns
    -------
    jnp.ndarray
        SW optical depth at half-levels, shape ``(n_levels+1, n_lat, n_lon)``.
    """
    n_levels = dsigma.shape[0]

    # Humidity-dependent increment per layer
    ps_ratio = surface_pressure[None, :, :] / reference_pressure
    dtau = (byrne_sw_a + byrne_sw_b * humidity) * dsigma[:, None, None] * ps_ratio

    # Cumulative from TOA
    tau_cumsum = jnp.cumsum(dtau, axis=0)  # (n_levels, n_lat, n_lon)

    # Prepend zero at TOA
    n_lat, n_lon = surface_pressure.shape
    tau_toa = jnp.zeros((1, n_lat, n_lon))
    tau_humidity = jnp.concatenate([tau_toa, tau_cumsum], axis=0)

    # Add Frierson-style base optical depth: sw_tau_0 * sigma^2
    # Use sigma at half-levels for consistency
    sigma_half = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(dsigma),
    ])  # (n_levels+1,)
    tau_base = sw_tau_0 * sigma_half[:, None, None] ** 2

    return tau_humidity + jnp.broadcast_to(tau_base, (n_levels + 1, n_lat, n_lon))
