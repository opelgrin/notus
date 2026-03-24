"""Moisture thermodynamic primitives.

Saturation vapor pressure (Bolton 1980), saturation specific humidity,
and moist pseudoadiabatic lapse rate.  All functions are JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def saturation_vapor_pressure(temperature: jnp.ndarray) -> jnp.ndarray:
    """Saturation vapor pressure over liquid water (Bolton 1980).

    ::

        e_sat(T) = 611.2 exp(17.67 (T - 273.15) / (T - 29.65))

    Valid for -35 °C to +35 °C.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K].

    Returns
    -------
    jnp.ndarray
        Saturation vapor pressure [Pa].
    """
    t_celsius = temperature - 273.15
    return 611.2 * jnp.exp(17.67 * t_celsius / (t_celsius + 243.5))


def saturation_specific_humidity(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Saturation specific humidity from temperature and pressure.

    ::

        q_sat = ε e_sat / (p - (1 - ε) e_sat)

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature [K].
    pressure : jnp.ndarray
        Total pressure [Pa].
    epsilon : float
        Ratio R_d / R_v (≈ 0.622 for Earth).

    Returns
    -------
    jnp.ndarray
        Saturation specific humidity [kg/kg].
    """
    e_sat = saturation_vapor_pressure(temperature)
    return epsilon * e_sat / (pressure - (1.0 - epsilon) * e_sat)


def moist_adiabat(
    t_surface: float | jnp.ndarray,
    p_levels: jnp.ndarray,
    p_surface: float | jnp.ndarray,
    epsilon: float,
    latent_heat: float,
    specific_heat_cp: float,
    gas_constant: float,
) -> jnp.ndarray:
    """Compute a moist pseudoadiabatic temperature profile.

    Integrates the pseudoadiabatic lapse rate upward from ``t_surface``
    at ``p_surface`` to each pressure level in ``p_levels``.

    The pseudoadiabatic lapse rate is::

        dT/dp = (R_d T + L_v q_sat) / (c_p p + L_v² q_sat ε / (R_d T²))

    Integration uses small log-pressure steps for accuracy.

    Parameters
    ----------
    t_surface : float
        Surface temperature [K].
    p_levels : jnp.ndarray
        Pressure at target levels [Pa], shape ``(n_levels,)``.
        Must be ordered from top (smallest p) to bottom (largest p).
    p_surface : float
        Surface pressure [Pa].
    epsilon : float
        Ratio R_d / R_v.
    latent_heat : float
        Latent heat of vaporization [J/kg].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].

    Returns
    -------
    jnp.ndarray
        Temperature at each pressure level [K], shape ``(n_levels,)``.
    """
    n_steps = 200
    log_p_surface = jnp.log(p_surface)

    def _integrate_to_level(log_p_target: jnp.ndarray) -> jnp.ndarray:
        d_log_p = (log_p_target - log_p_surface) / n_steps

        def _step(
            carry: tuple[jnp.ndarray, jnp.ndarray],
            _: None,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            t, log_p = carry
            q_sat = saturation_specific_humidity(
                t, jnp.exp(log_p), epsilon
            )
            numerator = gas_constant * t + latent_heat * q_sat
            lv2_term = latent_heat**2 * q_sat * epsilon
            denominator = specific_heat_cp + lv2_term / (
                gas_constant * t**2
            )
            dt_dlnp = numerator / denominator
            t_new = t + dt_dlnp * d_log_p
            log_p_new = log_p + d_log_p
            return (t_new, log_p_new), None

        (t_final, _), _ = jax.lax.scan(
            _step,
            (jnp.asarray(t_surface, dtype=jnp.float64), log_p_surface),
            None,
            length=n_steps,
        )
        return t_final

    return jax.vmap(_integrate_to_level)(jnp.log(p_levels))
