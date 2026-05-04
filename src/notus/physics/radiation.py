"""Radiation schemes for idealized GCM experiments.

Longwave:
- Frierson (2006) gray scheme: prescribed tau(sigma, latitude)
- Byrne (Isca) scheme: humidity-dependent tau for water vapor feedback
- SPEEDY multi-band scheme: 4 LW bands + 2 SW bands

Shortwave:
- Beer-Lambert absorption with prescribed or humidity-dependent optical depth
- SPEEDY two-band scheme with cloud reflection/absorption

All functions are JIT-compatible.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from notus.physics.clouds import CloudConfig


if TYPE_CHECKING:
    from notus.physics.clouds import CloudDiagnostic


STEFAN_BOLTZMANN: float = 5.670374419e-8  # [W/(m² K⁴)]


# ---------------------------------------------------------------------------
# Radiation configuration dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FriersonRadiation:
    """Gray longwave radiation with prescribed optical depth.

    Frierson et al. (2006) scheme: tau(sigma, lat) with mixed
    linear + power-law pressure dependence.

    Parameters
    ----------
    tau_equator : float
        Equatorial longwave optical depth.
    tau_pole : float
        Polar longwave optical depth.
    linear_fraction : float
        Fraction of LW optical depth with linear pressure dependence.
    alpha : float
        Pressure exponent for the nonlinear part of LW optical depth.
    sw_tau_0 : float
        Shortwave optical depth. Zero disables atmospheric SW absorption.
    sw_exponent : float
        Shortwave pressure exponent for Beer-Lambert absorption.
    """

    tau_equator: float = 6.0
    tau_pole: float = 1.5
    linear_fraction: float = 0.1
    alpha: float = 4.0
    sw_tau_0: float = 0.0
    sw_exponent: float = 2.0


@dataclasses.dataclass(frozen=True, slots=True)
class ByrneRadiation:
    """Humidity-dependent optical depth with water vapor feedback.

    Byrne (Isca) scheme: dτ/d(p/p₀) = a + b·q, providing a physically
    motivated water vapor feedback on longwave radiation.

    Parameters
    ----------
    a : float
        Well-mixed gas LW absorption coefficient.
    b : float
        Water vapor LW absorption coefficient.
    sw_a : float
        Well-mixed gas SW absorption coefficient.
    sw_b : float
        Water vapor SW absorption coefficient.
    sw_tau_0 : float
        Base shortwave optical depth.
    sw_exponent : float
        Shortwave pressure exponent for Beer-Lambert absorption.
    """

    a: float = 0.8678
    b: float = 1997.9
    sw_a: float = 0.0
    sw_b: float = 0.2
    sw_tau_0: float = 0.0
    sw_exponent: float = 2.0


@dataclasses.dataclass(frozen=True, slots=True)
class SpeedyRadiation:
    """SPEEDY multi-band radiation (4 LW bands, 2 SW bands).

    Based on the SPEEDY model (Molteni 2003). Supports optional
    diagnostic cloud scheme for SW reflection and LW absorption.

    Parameters
    ----------
    epslw : float
        LW PBL emission fraction.
    surface_emissivity : float
        Surface LW emissivity.
    ablwin : float
        Window-band absorptivity.
    ablco2 : float
        CO₂-band absorptivity.
    ablwv1 : float
        H₂O weak-band absorptivity coefficient.
    ablwv2 : float
        H₂O strong-band absorptivity coefficient.
    absdry : float
        Dry-air absorptivity (SW band 1).
    absaer : float
        Aerosol absorptivity coefficient (SW band 1).
    sw_abswv1 : float
        Water vapor absorptivity, visible band (SW band 1).
    sw_abswv2 : float
        Water vapor absorptivity, near-IR band (SW band 2).
    visible_fraction : float
        Fraction of solar irradiance in the visible band.
    clouds : CloudConfig or None
        Cloud scheme parameters. ``None`` disables clouds.
    """

    epslw: float = 0.05
    surface_emissivity: float = 0.98
    ablwin: float = 0.3
    ablco2: float = 6.0
    ablwv1: float = 0.7
    ablwv2: float = 50.0
    absdry: float = 0.033
    absaer: float = 0.033
    sw_abswv1: float = 0.022
    sw_abswv2: float = 15.0
    visible_fraction: float = 0.95
    clouds: CloudConfig | None = None


#: Union type for radiation scheme configuration.
RadiationConfig = FriersonRadiation | ByrneRadiation | SpeedyRadiation


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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(heating_rate, lw_down_sfc, olr)`` — longwave heating rate [K/s]
        shape ``(n_levels, n_lat, n_lon)``, downward LW flux at the surface
        [W/m²] shape ``(n_lat, n_lon)``, and outgoing longwave radiation at
        TOA [W/m²] shape ``(n_lat, n_lon)``.
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
    # Upward LW flux at TOA (first half-level interface)
    olr = f_up[0]  # (n_lat, n_lon)

    return heating_rate, lw_down_sfc, olr


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


# ---------------------------------------------------------------------------
# SPEEDY-style multi-band radiation (Molteni 2003 / JCM)
# ---------------------------------------------------------------------------

N_LW_BANDS: int = 4


def speedy_lw_band_fractions(
    temperature: jnp.ndarray,
    *,
    epslw: float = 0.05,
) -> jnp.ndarray:
    """Compute temperature-dependent LW band fractions (SPEEDY ``radset``).

    Returns the fraction of blackbody emission in each of 4 spectral
    bands as a function of temperature, following Molteni (2003)::

        f₁ = 0.148 − 3.0×10⁻⁶ (T − 247)²   (H₂O weak)
        f₂ = 0.356 − 5.2×10⁻⁶ (T − 282)²   (H₂O strong)
        f₃ = 0.314 + 1.0×10⁻⁵ (T − 315)²   (CO₂)
        f₀ = 1 − f₁ − f₂ − f₃               (window)

    All fractions are scaled by ``(1 − epslw)``.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K], any shape.
    epslw : float
        Fraction of blackbody spectrum absorbed/emitted by PBL only.

    Returns
    -------
    jnp.ndarray
        Band fractions, shape ``(4, *temperature.shape)``.
    """
    t = jnp.clip(temperature, 200.0, 320.0)
    f1 = 0.148 - 3.0e-6 * (t - 247.0) ** 2
    f2 = 0.356 - 5.2e-6 * (t - 282.0) ** 2
    f3 = 0.314 + 1.0e-5 * (t - 315.0) ** 2
    f0 = 1.0 - f1 - f2 - f3
    scale = 1.0 - epslw
    return scale * jnp.stack([f0, f1, f2, f3], axis=0)


def _speedy_two_stream_band(
    bb_band: jnp.ndarray,
    transmissivity: jnp.ndarray,
    bb_surface_band: jnp.ndarray,
    surface_emissivity: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-band two-stream radiative transfer.

    Computes upward and downward fluxes at half-level interfaces for
    one spectral band, then returns the net flux divergence per layer,
    the downward flux at the surface, and the outgoing flux at TOA.

    Designed to be ``jax.vmap``-ed over the band axis.

    Parameters
    ----------
    bb_band : jnp.ndarray
        Band-weighted blackbody emission at full levels [W/m²],
        shape ``(n_levels, n_lat, n_lon)``.
    transmissivity : jnp.ndarray
        Layer transmissivity for this band, shape
        ``(n_levels, n_lat, n_lon)``.
    bb_surface_band : jnp.ndarray
        Band-weighted surface blackbody emission [W/m²],
        shape ``(n_lat, n_lon)``.
    surface_emissivity : float
        Surface LW emissivity (0-1).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(f_net_half, lw_down_sfc, olr)`` where ``f_net_half`` is the
        net upward flux at half-level interfaces ``(n_levels+1, n_lat, n_lon)``,
        ``lw_down_sfc`` is the downward flux at the surface ``(n_lat, n_lon)``,
        and ``olr`` is the outgoing LW at TOA ``(n_lat, n_lon)``.
    """
    emissivity = 1.0 - transmissivity  # (n_levels, n_lat, n_lon)

    # --- Upward flux: scan from surface to TOA ---
    f_up_sfc = surface_emissivity * bb_surface_band  # (n_lat, n_lon)

    def _upward_step(
        f_up: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trans_k, emis_k, bb_k = layer
        f_up_new = f_up * trans_k + emis_k * bb_k
        return f_up_new, f_up_new

    _, f_up_inner_rev = jax.lax.scan(
        _upward_step,
        f_up_sfc,
        (transmissivity[::-1], emissivity[::-1], bb_band[::-1]),
    )
    f_up_inner = f_up_inner_rev[::-1]  # (n_levels, n_lat, n_lon)
    f_up = jnp.concatenate([f_up_inner, f_up_sfc[None]], axis=0)

    olr = f_up[0]  # (n_lat, n_lon)

    # --- Downward flux: scan from TOA to surface ---
    n_lat, n_lon = bb_surface_band.shape
    f_down_toa = jnp.zeros((n_lat, n_lon))

    def _downward_step(
        f_down: jnp.ndarray,
        layer: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trans_k, emis_k, bb_k = layer
        f_down_new = f_down * trans_k + emis_k * bb_k
        return f_down_new, f_down_new

    _, f_down_interfaces = jax.lax.scan(
        _downward_step,
        f_down_toa,
        (transmissivity, emissivity, bb_band),
    )
    f_down = jnp.concatenate([f_down_toa[None], f_down_interfaces], axis=0)

    lw_down_sfc = f_down[-1]  # (n_lat, n_lon)

    # Net upward flux at half-level interfaces
    f_net = f_up - f_down  # (n_levels+1, n_lat, n_lon)

    return f_net, lw_down_sfc, olr


def speedy_longwave_heating(
    temperature: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    dsigma: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    reference_pressure: float,
    gravity: float,
    specific_heat_cp: float,
    *,
    epslw: float = 0.05,
    surface_emissivity: float = 0.98,
    ablwin: float = 0.3,
    ablco2: float = 6.0,
    ablwv1: float = 0.7,
    ablwv2: float = 50.0,
    cloud: CloudDiagnostic | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute longwave heating with 4-band SPEEDY scheme.

    Four spectral bands with temperature-dependent fractions and
    humidity/CO₂-dependent absorptivities::

        Band 0 (window):    α = ablwin            (dry air only)
        Band 1 (CO₂):       α = ablco2            (well-mixed)
        Band 2 (H₂O weak):  α = ablwv1 · q        (humidity-dependent)
        Band 3 (H₂O strong): α = ablwv2 · q        (humidity-dependent)

    When ``cloud`` is provided, cloud absorptivity is added:
    below cloud top, ``ablcl1 * cloudc`` to the window band;
    above cloud top, ``ablcl2 * cloudc`` to window and H₂O bands.

    Layer transmissivity: ``τ = exp(-(ps/p₀) · Δσ · α)``

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels [K], shape ``(n_levels, n_lat, n_lon)``.
    surface_temperature : jnp.ndarray
        Surface temperature [K], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    humidity : jnp.ndarray
        Specific humidity [kg/kg], shape ``(n_levels, n_lat, n_lon)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    reference_pressure : float
        Reference pressure p₀ [Pa].
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    epslw : float
        Fraction of blackbody emitted by PBL only.
    surface_emissivity : float
        Surface LW emissivity.
    ablwin : float
        Window-band absorptivity (per Δp = p₀).
    ablco2 : float
        CO₂-band absorptivity (per Δp = p₀).
    ablwv1 : float
        H₂O weak-band absorptivity coefficient.
    ablwv2 : float
        H₂O strong-band absorptivity coefficient.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(heating_rate, lw_down_sfc, olr)`` — LW heating rate [K/s]
        shape ``(n_levels, n_lat, n_lon)``, downward LW flux at the
        surface [W/m²] ``(n_lat, n_lon)``, and outgoing LW radiation
        at TOA [W/m²] ``(n_lat, n_lon)``.
    """
    _n_levels, n_lat, n_lon = temperature.shape

    # Broadcast surface temperature
    if surface_temperature.ndim == 1:
        surface_temperature = surface_temperature[:, None]
    t_s = jnp.broadcast_to(surface_temperature, (n_lat, n_lon))

    # Pressure ratio per layer: (n_levels, n_lat, n_lon)
    ps_ratio = surface_pressure[None, :, :] / reference_pressure
    dp_factor = dsigma[:, None, None] * ps_ratio

    # Per-band absorptivity: (4, n_levels, n_lat, n_lon)
    n_levels = temperature.shape[0]
    q = jnp.maximum(humidity, 0.0)
    a_win = jnp.broadcast_to(jnp.full_like(dp_factor, ablwin), dp_factor.shape)
    a_co2 = jnp.broadcast_to(jnp.full_like(dp_factor, ablco2), dp_factor.shape)
    a_wv1 = ablwv1 * q
    a_wv2 = ablwv2 * q

    # Add cloud absorptivity to LW bands
    if cloud is not None:
        level_idx = jnp.arange(n_levels)[:, None, None]
        below_top = level_idx >= cloud.cloud_top[None, :, :]
        above_top = level_idx < cloud.cloud_top[None, :, :]
        cc = cloud.cloud_cover[None, :, :]

        # Below cloud top: thick cloud in window band (ablcl1=12.0)
        a_win += jnp.where(below_top, 12.0 * cc, 0.0)
        # Above cloud top: thin cloud in window + H₂O bands (ablcl2=0.6)
        a_win += jnp.where(above_top, 0.6 * cc, 0.0)
        a_wv1 = jnp.maximum(a_wv1, jnp.where(above_top, 0.6 * cc, 0.0))
        a_wv2 = jnp.maximum(a_wv2, jnp.where(above_top, 0.6 * cc, 0.0))

    alpha_bands = jnp.stack([a_win, a_co2, a_wv1, a_wv2], axis=0)

    # Transmissivity per band per layer
    tau_bands = dp_factor[None, :, :, :] * alpha_bands
    trans_bands = jnp.exp(-tau_bands)  # (4, n_levels, n_lat, n_lon)

    # Band-weighted blackbody at full levels: (4, n_levels, n_lat, n_lon)
    fband = speedy_lw_band_fractions(temperature, epslw=epslw)
    bb_total = STEFAN_BOLTZMANN * temperature**4  # (n_levels, n_lat, n_lon)
    bb_bands = fband * bb_total[None, :, :, :]

    # Band-weighted surface blackbody: (4, n_lat, n_lon)
    fband_sfc = speedy_lw_band_fractions(t_s, epslw=epslw)
    bb_sfc_total = STEFAN_BOLTZMANN * t_s**4
    bb_sfc_bands = fband_sfc * bb_sfc_total[None, :, :]

    # vmap the two-stream over the band axis (axis 0)
    f_net_all, lw_down_all, olr_all = jax.vmap(
        lambda bb, tr, bb_s: _speedy_two_stream_band(
            bb,
            tr,
            bb_s,
            surface_emissivity,
        ),
    )(bb_bands, trans_bands, bb_sfc_bands)

    # Sum over bands
    f_net = jnp.sum(f_net_all, axis=0)  # (n_levels+1, n_lat, n_lon)
    lw_down_sfc = jnp.sum(lw_down_all, axis=0)  # (n_lat, n_lon)
    olr = jnp.sum(olr_all, axis=0)  # (n_lat, n_lon)

    # Heating rate from net flux divergence
    df_net = f_net[:-1] - f_net[1:]  # (n_levels, n_lat, n_lon)
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]
    heating_rate = -gravity / specific_heat_cp * df_net / dp

    return heating_rate, lw_down_sfc, olr


def speedy_lw_down_surface(
    temperature: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    dsigma: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    reference_pressure: float,
    *,
    epslw: float = 0.05,
    surface_emissivity: float = 0.98,
    ablwin: float = 0.3,
    ablco2: float = 6.0,
    ablwv1: float = 0.7,
    ablwv2: float = 50.0,
    cloud: CloudDiagnostic | None = None,
) -> jnp.ndarray:
    """Compute downward LW flux at surface using 4-band SPEEDY scheme.

    Downward-only scan (cheaper than full ``speedy_longwave_heating``).

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels [K], shape ``(n_levels, n_lat, n_lon)``.
    surface_temperature : jnp.ndarray
        Surface temperature [K], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    humidity : jnp.ndarray
        Specific humidity [kg/kg], shape ``(n_levels, n_lat, n_lon)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    reference_pressure : float
        Reference pressure p₀ [Pa].
    epslw, surface_emissivity, ablwin, ablco2, ablwv1, ablwv2 : float
        SPEEDY LW parameters (see ``speedy_longwave_heating``).
    cloud : CloudDiagnostic or None
        Diagnostic cloud fields for LW cloud absorption.

    Returns
    -------
    jnp.ndarray
        Downward LW flux at the surface [W/m²], shape ``(n_lat, n_lon)``.
    """
    n_levels, n_lat, n_lon = temperature.shape

    # Pressure ratio per layer
    ps_ratio = surface_pressure[None, :, :] / reference_pressure
    dp_factor = dsigma[:, None, None] * ps_ratio

    # Per-band absorptivity and transmissivity
    q = jnp.maximum(humidity, 0.0)
    a_win = jnp.broadcast_to(jnp.full_like(dp_factor, ablwin), dp_factor.shape)
    a_co2 = jnp.broadcast_to(jnp.full_like(dp_factor, ablco2), dp_factor.shape)
    a_wv1 = ablwv1 * q
    a_wv2 = ablwv2 * q

    if cloud is not None:
        level_idx = jnp.arange(n_levels)[:, None, None]
        below_top = level_idx >= cloud.cloud_top[None, :, :]
        above_top = level_idx < cloud.cloud_top[None, :, :]
        cc = cloud.cloud_cover[None, :, :]
        a_win += jnp.where(below_top, 12.0 * cc, 0.0)
        a_win += jnp.where(above_top, 0.6 * cc, 0.0)
        a_wv1 = jnp.maximum(a_wv1, jnp.where(above_top, 0.6 * cc, 0.0))
        a_wv2 = jnp.maximum(a_wv2, jnp.where(above_top, 0.6 * cc, 0.0))

    alpha_bands = jnp.stack([a_win, a_co2, a_wv1, a_wv2], axis=0)
    trans_bands = jnp.exp(-dp_factor[None] * alpha_bands)

    # Band-weighted blackbody at full levels
    fband = speedy_lw_band_fractions(temperature, epslw=epslw)
    bb_bands = fband * (STEFAN_BOLTZMANN * temperature**4)[None]

    # Downward scan per band (vmapped)
    def _downward_band(
        bb_band: jnp.ndarray,
        trans: jnp.ndarray,
    ) -> jnp.ndarray:
        emis = 1.0 - trans

        def _step(
            f_down: jnp.ndarray,
            layer: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, None]:
            tr_k, em_k, bb_k = layer
            return f_down * tr_k + em_k * bb_k, None

        f_down_sfc, _ = jax.lax.scan(
            _step,
            jnp.zeros((n_lat, n_lon)),
            (trans, emis, bb_band),
        )
        return f_down_sfc

    lw_down_bands = jax.vmap(_downward_band)(bb_bands, trans_bands)
    return jnp.sum(lw_down_bands, axis=0)


def speedy_shortwave_heating(
    dsigma: jnp.ndarray,
    sigma_full: jnp.ndarray,
    humidity: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    reference_pressure: float,
    insolation: jnp.ndarray,
    gravity: float,
    specific_heat_cp: float,
    *,
    surface_albedo: float | jnp.ndarray = 0.0,
    absdry: float = 0.033,
    absaer: float = 0.033,
    abswv1: float = 0.022,
    abswv2: float = 15.0,
    visible_fraction: float = 0.95,
    cloud: CloudDiagnostic | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute shortwave heating with 2-band SPEEDY scheme.

    Two spectral bands:

    - **Band 1** (visible, ``visible_fraction`` of total): absorbed by
      dry air, aerosols (σ²-weighted), and weak H₂O.
    - **Band 2** (near-IR, remainder): absorbed by strong H₂O only.

    When ``cloud`` is provided, cloud reflection and absorption are
    added to the visible band.

    Downward Beer-Lambert per band, surface albedo reflection, upward
    absorption, then sum across bands.

    Parameters
    ----------
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    sigma_full : jnp.ndarray
        Full-level σ values, shape ``(n_levels,)``.
    humidity : jnp.ndarray
        Specific humidity [kg/kg], shape ``(n_levels, n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    reference_pressure : float
        Reference pressure p₀ [Pa].
    insolation : jnp.ndarray
        TOA insolation [W/m²], shape ``(n_lat,)``.
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    surface_albedo : float or jnp.ndarray
        Surface albedo (0-1).
    absdry : float
        Dry-air absorptivity (band 1).
    absaer : float
        Aerosol absorptivity coefficient (band 1, σ²-weighted).
    abswv1 : float
        Water vapor absorptivity (band 1, weak).
    abswv2 : float
        Water vapor absorptivity (band 2, strong near-IR).
    visible_fraction : float
        Fraction of solar irradiance in band 1.
    cloud : CloudDiagnostic or None
        Diagnostic cloud fields.  When provided, adds cloud reflection
        at cloud top and cloud absorption in the visible band.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(heating_rate, sw_down_sfc)`` — SW heating rate [K/s]
        shape ``(n_levels, n_lat, n_lon)`` and total downward SW flux
        at the surface [W/m²] shape ``(n_lat, n_lon)``.
    """
    n_levels, n_lat, n_lon = humidity.shape
    q = jnp.maximum(humidity, 0.0)

    # Pressure factor per layer: (n_levels, n_lat, n_lon)
    ps_ratio = surface_pressure[None, :, :] / reference_pressure
    dp_factor = dsigma[:, None, None] * ps_ratio

    # Band 1 (visible): dry air + aerosol(σ²) + weak H₂O
    alpha_vis = absdry + absaer * sigma_full[:, None, None] ** 2 + abswv1 * q

    # Add cloud absorption to visible band in cloudy layers
    if cloud is not None:
        level_idx = jnp.arange(n_levels)[:, None, None]
        in_cloud = level_idx >= cloud.cloud_top[None, :, :]
        acloud = cloud.cloud_cover * jnp.minimum(
            cloud.cloud_humidity * abswv1 * 10.0,  # scaled cloud absorptivity
            0.15,  # abscl2 cap
        )
        alpha_vis += jnp.where(in_cloud, acloud[None, :, :], 0.0)

    trans_vis = jnp.exp(-dp_factor * alpha_vis)  # (n_levels, n_lat, n_lon)

    # Apply cloud reflection at cloud-top level (visible band only)
    if cloud is not None:
        cloud_refl = 1.0 - 0.43 * cloud.cloud_cover  # albcl
        at_cloud_top = level_idx == cloud.cloud_top[None, :, :]
        trans_vis *= jnp.where(at_cloud_top, cloud_refl[None, :, :], 1.0)

        # Stratiform reflection at PBL top (lowest level)
        strat_refl = 1.0 - 0.50 * cloud.stratiform_cover  # albcls
        at_pbl = level_idx == (n_levels - 1)
        trans_vis *= jnp.where(at_pbl, strat_refl[None, :, :], 1.0)

    # Band 2 (near-IR): strong H₂O only
    alpha_nir = abswv2 * q
    trans_nir = jnp.exp(-dp_factor * alpha_nir)  # (n_levels, n_lat, n_lon)

    # TOA flux per band
    s_vis = insolation[None, :, None] * visible_fraction  # (1, n_lat, 1)
    s_nir = insolation[None, :, None] * (1.0 - visible_fraction)

    # Downward Beer-Lambert per band (cumulative product of transmissivities)
    def _downward_scan(
        carry: jnp.ndarray,
        trans_k: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        f_new = carry * trans_k
        return f_new, f_new

    s_vis_2d = jnp.broadcast_to(s_vis, (1, n_lat, n_lon))[0]  # (n_lat, n_lon)
    _carry_vis, f_vis_levels = jax.lax.scan(_downward_scan, s_vis_2d, trans_vis)
    # f_vis_levels: (n_levels, n_lat, n_lon) — flux BELOW each layer
    f_vis_down = jnp.concatenate([s_vis_2d[None], f_vis_levels], axis=0)
    # (n_levels+1, n_lat, n_lon) — flux at half-levels (TOA to surface)

    s_nir_2d = jnp.broadcast_to(s_nir, (1, n_lat, n_lon))[0]
    _carry_nir, f_nir_levels = jax.lax.scan(_downward_scan, s_nir_2d, trans_nir)
    f_nir_down = jnp.concatenate([s_nir_2d[None], f_nir_levels], axis=0)

    # SW reaching surface (sum of both bands)
    sw_down_sfc = f_vis_down[-1] + f_nir_down[-1]  # (n_lat, n_lon)

    # Absorbed per layer (downward pass)
    f_abs_vis_down = f_vis_down[:-1] - f_vis_down[1:]
    f_abs_nir_down = f_nir_down[:-1] - f_nir_down[1:]

    # --- Upward reflected beam ---
    # Reflected at surface, then Beer-Lambert back up
    # Upward flux at surface = sw_down_sfc * albedo
    # tau_up from surface to level k = sum of tau from surface upward
    # We reverse the transmissivity and scan upward
    f_up_sfc_vis = f_vis_down[-1] * surface_albedo
    f_up_sfc_nir = f_nir_down[-1] * surface_albedo

    _carry_vis_up, f_vis_up_levels = jax.lax.scan(
        _downward_scan,
        f_up_sfc_vis,
        trans_vis[::-1],
    )
    f_vis_up = jnp.concatenate(
        [f_vis_up_levels[::-1], f_up_sfc_vis[None]],
        axis=0,
    )  # (n_levels+1, n_lat, n_lon)

    _carry_nir_up, f_nir_up_levels = jax.lax.scan(
        _downward_scan,
        f_up_sfc_nir,
        trans_nir[::-1],
    )
    f_nir_up = jnp.concatenate(
        [f_nir_up_levels[::-1], f_up_sfc_nir[None]],
        axis=0,
    )

    # Absorbed per layer (upward pass)
    f_abs_vis_up = f_vis_up[1:] - f_vis_up[:-1]
    f_abs_nir_up = f_nir_up[1:] - f_nir_up[:-1]

    # Total absorbed per layer
    f_absorbed = f_abs_vis_down + f_abs_nir_down + f_abs_vis_up + f_abs_nir_up

    # Heating rate
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]
    heating_rate = gravity / specific_heat_cp * f_absorbed / dp

    return heating_rate, sw_down_sfc
