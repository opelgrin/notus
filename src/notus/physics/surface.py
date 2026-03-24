"""Surface boundary conditions and fluxes for aquaplanet experiments.

Prescribed SST profiles, bulk-aerodynamic sensible and latent heat fluxes.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from notus.physics.moisture import saturation_specific_humidity


@dataclasses.dataclass(frozen=True, slots=True)
class PrescribedSST:
    """Frierson et al. (2006) aquaplanet SST profile.

    Gaussian profile with a temperature floor::

        T_s(phi) = max(T_min, T_min + T_delta * exp(-0.5 * (phi / phi_w)^2))

    Parameters
    ----------
    t_min : float
        Minimum SST (temperature floor) [K].
    t_delta : float
        Equator-to-floor temperature difference [K].
    phi_w : float
        Latitude width parameter [rad].
    """

    t_min: float = 271.0
    t_delta: float = 29.0
    phi_w: float = 26.0 * jnp.pi / 180.0


def compute_sst(
    config: PrescribedSST,
    latitudes: jnp.ndarray,
) -> jnp.ndarray:
    """Compute SST at each latitude.

    Parameters
    ----------
    config : PrescribedSST
        SST profile parameters.
    latitudes : jnp.ndarray
        Latitude in radians, shape ``(n_lat,)``.

    Returns
    -------
    jnp.ndarray
        SST in Kelvin, shape ``(n_lat,)``. Zonally uniform.
    """
    return config.t_min + config.t_delta * jnp.exp(
        -0.5 * (latitudes / config.phi_w) ** 2,
    )


def surface_sensible_heat_flux(
    t_surface: jnp.ndarray,
    t_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    gravity: float,
    specific_heat_cp: float,
    gas_constant: float,
    dsigma_lowest: float,
    *,
    drag_coefficient: float,
) -> jnp.ndarray:
    """Compute surface sensible heat flux tendency for the lowest level.

    Bulk aerodynamic formula::

        H = rho * cp * C_D * |v| * (T_s - T_a)

    converted to a temperature tendency for the lowest model level::

        dT/dt = H / (dp/g) / cp = g * C_D * |v| * (T_s - T_a) / dp

    where dp = dsigma_lowest * ps is the pressure thickness of the
    lowest layer.

    Parameters
    ----------
    t_surface : jnp.ndarray
        Surface temperature, shape ``(n_lat,)``.
    t_air : jnp.ndarray
        Temperature at the lowest model level, shape ``(n_lat, n_lon)``.
    wind_speed : jnp.ndarray
        Wind speed at the lowest model level, shape ``(n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    drag_coefficient : float
        Surface drag coefficient C_D (dimensionless).

    Returns
    -------
    jnp.ndarray
        Temperature tendency [K/s] for the lowest model level,
        shape ``(n_lat, n_lon)``.
    """
    dp = dsigma_lowest * surface_pressure  # (n_lat, n_lon)

    # Surface density from ideal gas law: rho = p_sfc / (R * T_a)
    # where p_sfc ~ sigma_lowest * ps.
    # Tendency: dT/dt = g * H / (dp * cp), with H = rho * cp * C_D * |v| * (T_s - T_a).
    sigma_lowest = 1.0 - 0.5 * dsigma_lowest
    t_air_safe = jnp.maximum(t_air, 1.0)
    dp_safe = jnp.maximum(dp, 1.0)
    rho_sfc = surface_pressure * sigma_lowest / (gas_constant * t_air_safe)
    flux = rho_sfc * specific_heat_cp * drag_coefficient * wind_speed * (t_surface[:, None] - t_air)

    return gravity * flux / (dp_safe * specific_heat_cp)


def surface_latent_heat_flux(
    t_surface: jnp.ndarray,
    q_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    gravity: float,
    gas_constant: float,
    dsigma_lowest: float,
    epsilon: float,
    *,
    drag_coefficient: float,
) -> jnp.ndarray:
    """Compute surface evaporation tendency for the lowest level.

    Bulk aerodynamic formula::

        E = rho * C_D * |v| * (q_sat(T_s, p_s) - q_a)

    converted to a specific humidity tendency for the lowest level::

        dq/dt = g * E / dp

    where dp = dsigma_lowest * ps.

    Parameters
    ----------
    t_surface : jnp.ndarray
        Surface temperature, shape ``(n_lat,)``.
    q_air : jnp.ndarray
        Specific humidity at the lowest level, shape ``(n_lat, n_lon)``.
    wind_speed : jnp.ndarray
        Wind speed at the lowest level, shape ``(n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    epsilon : float
        Ratio R_d / R_v (≈ 0.622).
    drag_coefficient : float
        Surface drag coefficient C_D (dimensionless).

    Returns
    -------
    jnp.ndarray
        Specific humidity tendency [kg/kg/s] for the lowest level,
        shape ``(n_lat, n_lon)``.
    """
    dp = dsigma_lowest * surface_pressure
    dp_safe = jnp.maximum(dp, 1.0)

    # Surface density
    sigma_lowest = 1.0 - 0.5 * dsigma_lowest
    t_sfc_bc = t_surface[:, None]
    t_air_safe = jnp.maximum(t_sfc_bc, 1.0)
    rho_sfc = surface_pressure * sigma_lowest / (gas_constant * t_air_safe)

    # Saturation specific humidity at the surface
    q_sat_sfc = saturation_specific_humidity(
        t_sfc_bc, surface_pressure, epsilon,
    )

    # Evaporation flux (positive = upward, moistening the atmosphere)
    evap = rho_sfc * drag_coefficient * wind_speed * (q_sat_sfc - q_air)

    return gravity * evap / dp_safe
