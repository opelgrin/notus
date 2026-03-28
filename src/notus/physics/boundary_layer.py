"""Monin-Obukhov surface layer parameterization.

Stability-dependent transfer coefficients using the Louis (1979)
formulation.  Replaces the constant drag coefficient with bulk
Richardson number dependent corrections to the neutral exchange
coefficients for momentum, heat, and moisture.

Reference
---------
Louis, J.-F. (1979). "A parametric model of vertical eddy fluxes in
the atmosphere." Boundary-Layer Meteorology, 17, 187-202.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


# Von Karman constant
VON_KARMAN: float = 0.4


@dataclasses.dataclass(frozen=True, slots=True)
class SurfaceLayerConfig:
    """Configuration for the Louis (1979) surface layer scheme.

    Parameters
    ----------
    z0_momentum : float
        Roughness length for momentum [m].
    z0_heat : float
        Roughness length for heat and moisture [m].
        If None, defaults to ``z0_momentum / 10``.
    louis_b : float
        Louis (1979) stability parameter *b* (Table 1).
    louis_c_m : float
        Louis (1979) unstable enhancement parameter for momentum.
    louis_c_h : float
        Louis (1979) unstable enhancement parameter for heat.
    louis_d : float
        Louis (1979) stable denominator parameter *d*.
    ri_max : float
        Maximum bulk Richardson number (stability clamp).
    """

    z0_momentum: float = 1.0e-4
    z0_heat: float | None = None
    louis_b: float = 5.0
    louis_c_m: float = 7.5
    louis_c_h: float = 5.0
    louis_d: float = 5.0
    ri_max: float = 1.0

    @property
    def z0_heat_effective(self) -> float:
        """Roughness length for heat, defaulting to z0_momentum / 10."""
        if self.z0_heat is not None:
            return self.z0_heat
        return self.z0_momentum / 10.0


def neutral_drag_coefficient(
    z_ref: float | jnp.ndarray,
    z0: float | jnp.ndarray,
) -> float | jnp.ndarray:
    r"""Neutral drag coefficient from log-wind profile.

    .. math::
        C_{DN} = \left(\frac{\kappa}{\ln(z_{ref} / z_0)}\right)^2

    Parameters
    ----------
    z_ref : float or jnp.ndarray
        Reference height [m].
    z0 : float
        Roughness length [m].

    Returns
    -------
    float or jnp.ndarray
        Neutral drag coefficient (dimensionless).
    """
    ln_ratio = jnp.log(z_ref / z0)
    return (VON_KARMAN / ln_ratio) ** 2


def neutral_heat_coefficient(
    z_ref: float | jnp.ndarray,
    z0_m: float | jnp.ndarray,
    z0_h: float | jnp.ndarray,
) -> float | jnp.ndarray:
    r"""Neutral heat transfer coefficient.

    .. math::
        C_{HN} = \frac{\kappa^2}{\ln(z_{ref}/z_{0m}) \cdot \ln(z_{ref}/z_{0h})}

    Parameters
    ----------
    z_ref : float or jnp.ndarray
        Reference height [m].
    z0_m : float
        Roughness length for momentum [m].
    z0_h : float
        Roughness length for heat [m].

    Returns
    -------
    float or jnp.ndarray
        Neutral heat transfer coefficient (dimensionless).
    """
    return VON_KARMAN**2 / (jnp.log(z_ref / z0_m) * jnp.log(z_ref / z0_h))


def bulk_richardson_number(
    t_surface: jnp.ndarray,
    t_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    z_ref: float | jnp.ndarray,
    gravity: float,
) -> jnp.ndarray:
    r"""Bulk Richardson number between the surface and reference height.

    .. math::
        Ri_b = \frac{g \cdot z_{ref} \cdot (T_a - T_s)}{T_a \cdot |V|^2}

    Parameters
    ----------
    t_surface : jnp.ndarray
        Surface temperature [K].
    t_air : jnp.ndarray
        Air temperature at reference height [K].
    wind_speed : jnp.ndarray
        Wind speed at reference height [m/s].
    z_ref : float or jnp.ndarray
        Reference height [m].
    gravity : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    jnp.ndarray
        Bulk Richardson number (dimensionless).  Negative = unstable
        (surface warmer), positive = stable (surface cooler).
    """
    t_air_safe = jnp.maximum(t_air, 1.0)
    wind_safe = jnp.maximum(wind_speed, 0.01)
    return gravity * z_ref * (t_air - t_surface) / (t_air_safe * wind_safe**2)


def louis_stability_functions(
    ri_b: jnp.ndarray,
    z_ref: float | jnp.ndarray,
    z0_m: float | jnp.ndarray,
    z0_h: float | jnp.ndarray,
    c_dn: float | jnp.ndarray,
    c_hn: float | jnp.ndarray,
    *,
    b: float = 5.0,
    c_m: float = 7.5,
    c_h: float = 5.0,
    d: float = 5.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Louis (1979) stability correction functions.

    Returns multiplicative factors ``(f_m, f_h)`` such that::

        C_D = C_DN · f_m   (momentum)
        C_H = C_HN · f_h   (heat and moisture)

    **Unstable** (:math:`Ri_b < 0`):

    .. math::
        f_m = 1 - \frac{2 b \, Ri_b}{1 + 3 b \, c_m \sqrt{C_{DN}} \sqrt{z/z_{0m}} \sqrt{|Ri_b|}}

    .. math::
        f_h = 1 - \frac{3 b \, Ri_b}{1 + 3 b \, c_h \sqrt{C_{HN}} \sqrt{z/z_{0h}} \sqrt{|Ri_b|}}

    **Stable** (:math:`Ri_b \geq 0`):

    .. math::
        f_m = f_h = \frac{1}{1 + 2 b \, Ri_b / \sqrt{1 + d \, Ri_b}}

    Parameters
    ----------
    ri_b : jnp.ndarray
        Bulk Richardson number.
    z_ref : float or jnp.ndarray
        Reference height [m].
    z0_m, z0_h : float or jnp.ndarray
        Roughness lengths for momentum and heat [m].
    c_dn, c_hn : float or jnp.ndarray
        Neutral drag / heat transfer coefficients.
    b, c_m, c_h, d : float
        Louis (1979) empirical parameters.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(f_m, f_h)`` — stability correction factors for momentum and heat.
    """
    ri_abs = jnp.abs(ri_b)
    ri_sqrt = jnp.sqrt(ri_abs)

    # Unstable branch
    denom_m_unstable = 1.0 + 3.0 * b * c_m * jnp.sqrt(c_dn) * jnp.sqrt(z_ref / z0_m) * ri_sqrt
    denom_h_unstable = 1.0 + 3.0 * b * c_h * jnp.sqrt(c_hn) * jnp.sqrt(z_ref / z0_h) * ri_sqrt
    f_m_unstable = 1.0 - 2.0 * b * ri_b / denom_m_unstable
    f_h_unstable = 1.0 - 3.0 * b * ri_b / denom_h_unstable

    # Stable branch
    f_stable = 1.0 / (1.0 + 2.0 * b * ri_b / jnp.sqrt(1.0 + d * ri_b))

    # Select branch
    f_m = jnp.where(ri_b < 0.0, f_m_unstable, f_stable)
    f_h = jnp.where(ri_b < 0.0, f_h_unstable, f_stable)

    return f_m, f_h


def reference_height_from_sigma(
    dsigma_lowest: float,
    temperature: float | jnp.ndarray,
    gravity: float,
    gas_constant: float,
) -> float | jnp.ndarray:
    r"""Estimate reference height from sigma-coordinate lowest level.

    Approximates the height of the lowest full level as:

    .. math::
        z_{ref} = \frac{R \cdot T}{g} \cdot \frac{\Delta\sigma / 2}{\sigma_{lowest}}

    where :math:`\sigma_{lowest} = 1 - \Delta\sigma / 2`.

    Parameters
    ----------
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    temperature : float or jnp.ndarray
        Temperature at the lowest level [K].
    gravity : float
        Gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].

    Returns
    -------
    float or jnp.ndarray
        Estimated reference height [m].
    """
    sigma_lowest = 1.0 - 0.5 * dsigma_lowest
    return gas_constant * temperature / gravity * (0.5 * dsigma_lowest / sigma_lowest)


def compute_transfer_coefficients(
    t_surface: jnp.ndarray,
    t_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    dsigma_lowest: float,
    gravity: float,
    gas_constant: float,
    config: SurfaceLayerConfig,
    *,
    z0_momentum_override: jnp.ndarray | None = None,
    z0_heat_override: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute stability-dependent transfer coefficients.

    Returns drag coefficient for momentum and heat transfer coefficient,
    both modified by Louis (1979) stability functions applied to the
    bulk Richardson number.

    Parameters
    ----------
    t_surface : jnp.ndarray
        Surface temperature [K].
    t_air : jnp.ndarray
        Air temperature at the lowest model level [K].
    wind_speed : jnp.ndarray
        Wind speed at the lowest model level [m/s].
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    gravity : float
        Gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    config : SurfaceLayerConfig
        Surface layer configuration.
    z0_momentum_override : jnp.ndarray or None
        If provided, spatially varying momentum roughness length [m]
        that overrides ``config.z0_momentum``.
    z0_heat_override : jnp.ndarray or None
        If provided, spatially varying heat roughness length [m]
        that overrides ``config.z0_heat_effective``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(c_d_momentum, c_h_heat)`` — transfer coefficients for
        momentum and heat/moisture, shaped like ``t_air``.
    """
    z0_m = z0_momentum_override if z0_momentum_override is not None else config.z0_momentum
    z0_h = z0_heat_override if z0_heat_override is not None else config.z0_heat_effective

    # Estimate reference height from lowest-level temperature
    z_ref = reference_height_from_sigma(dsigma_lowest, t_air, gravity, gas_constant)

    # Neutral coefficients
    c_dn = neutral_drag_coefficient(z_ref, z0_m)
    c_hn = neutral_heat_coefficient(z_ref, z0_m, z0_h)

    # Bulk Richardson number
    ri_b = bulk_richardson_number(t_surface, t_air, wind_speed, z_ref, gravity)
    ri_b = jnp.clip(ri_b, -10.0, config.ri_max)

    # Stability correction
    f_m, f_h = louis_stability_functions(
        ri_b,
        z_ref,
        z0_m,
        z0_h,
        c_dn,
        c_hn,
        b=config.louis_b,
        c_m=config.louis_c_m,
        c_h=config.louis_c_h,
        d=config.louis_d,
    )

    return c_dn * f_m, c_hn * f_h


def default_z_ref(
    dsigma_lowest: float,
    reference_temperature: float = 280.0,
    gravity: float = 9.80616,
    gas_constant: float = 287.04,
) -> float:
    """Compute a typical reference height for the lowest model level.

    Useful for estimating neutral drag coefficients at initialization.

    Parameters
    ----------
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    reference_temperature : float
        Typical surface-layer temperature [K].
    gravity : float
        Gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].

    Returns
    -------
    float
        Estimated reference height [m].
    """
    sigma_lowest = 1.0 - 0.5 * dsigma_lowest
    return gas_constant * reference_temperature / gravity * (0.5 * dsigma_lowest / sigma_lowest)
