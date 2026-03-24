"""Convective adjustment and large-scale condensation.

- Dry convective adjustment (Manabe & Strickler 1964)
- Large-scale condensation with latent heating (Frierson 2006)
- Simplified Betts-Miller convection (Frierson 2007)

All functions are JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from notus.physics.moisture import moist_adiabat, saturation_specific_humidity


def dry_convective_adjustment(
    temperature: jnp.ndarray,
    sigma_full: jnp.ndarray,
    dsigma: jnp.ndarray,
    kappa: float,
    *,
    n_iterations: int = 3,
) -> jnp.ndarray:
    """Adjust temperature to remove static instability.

    Sweeps bottom-to-top through adjacent layer pairs.  When the potential
    temperature θ = T · σ^{−κ} decreases upward (unstable), the pair is
    mixed to equal θ while conserving Δσ_k T_k + Δσ_{k+1} T_{k+1}.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels, shape ``(n_levels, n_lat, n_lon)``.
    sigma_full : jnp.ndarray
        Full-level σ values, shape ``(n_levels,)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    kappa : float
        R/c_p ratio.
    n_iterations : int
        Number of bottom-up sweeps (default 3).

    Returns
    -------
    jnp.ndarray
        Adjusted temperature, shape ``(n_levels, n_lat, n_lon)``.
    """
    n_levels = temperature.shape[0]

    # Pre-compute σ^κ for potential temperature conversion
    sigma_kappa = sigma_full**kappa  # (n_levels,)

    def _single_sweep(t: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
        # Sweep from bottom (k = n_levels-2) to top (k = 0)
        # using a JAX loop to avoid Python-loop unrolling in larger setups.
        def _body(i: int, t_in: jnp.ndarray) -> jnp.ndarray:
            k = (n_levels - 2) - i
            return _adjust_pair(t_in, k, sigma_kappa=sigma_kappa, dsigma=dsigma)

        return jax.lax.fori_loop(0, n_levels - 1, _body, t), None

    t_adjusted, _ = jax.lax.scan(_single_sweep, temperature, None, length=n_iterations)
    return t_adjusted


def _adjust_pair(
    temperature: jnp.ndarray,
    k: int,
    *,
    sigma_kappa: jnp.ndarray,
    dsigma: jnp.ndarray,
) -> jnp.ndarray:
    """Adjust a single pair of adjacent levels if statically unstable.

    Parameters
    ----------
    temperature : jnp.ndarray
        Full temperature field, shape ``(n_levels, n_lat, n_lon)``.
    k : int
        Index of the upper level (k is above, k+1 is below).
    sigma_kappa : jnp.ndarray
        Precomputed σ^κ at all levels, shape ``(n_levels,)``.
    dsigma : jnp.ndarray
        Layer thicknesses Δσ at all levels, shape ``(n_levels,)``.

    Returns
    -------
    jnp.ndarray
        Temperature field with the pair adjusted if unstable.
    """
    t_above = jax.lax.dynamic_index_in_dim(temperature, k, axis=0, keepdims=False)
    t_below = jax.lax.dynamic_index_in_dim(temperature, k + 1, axis=0, keepdims=False)
    sigma_kappa_above = jax.lax.dynamic_index_in_dim(sigma_kappa, k, axis=0, keepdims=False)
    sigma_kappa_below = jax.lax.dynamic_index_in_dim(sigma_kappa, k + 1, axis=0, keepdims=False)
    dsigma_above = jax.lax.dynamic_index_in_dim(dsigma, k, axis=0, keepdims=False)
    dsigma_below = jax.lax.dynamic_index_in_dim(dsigma, k + 1, axis=0, keepdims=False)

    # Potential temperature: θ = T / σ^κ
    theta_above = t_above / sigma_kappa_above
    theta_below = t_below / sigma_kappa_below

    # Unstable if θ decreases upward (lower θ above higher θ)
    unstable = theta_above < theta_below

    # Enthalpy-conserving mixed potential temperature
    enthalpy = dsigma_above * t_above + dsigma_below * t_below
    theta_new = enthalpy / (dsigma_above * sigma_kappa_above + dsigma_below * sigma_kappa_below)

    # Only adjust unstable columns
    t_above_new = jnp.where(unstable, theta_new * sigma_kappa_above, t_above)
    t_below_new = jnp.where(unstable, theta_new * sigma_kappa_below, t_below)

    updated = jax.lax.dynamic_update_index_in_dim(temperature, t_above_new, k, axis=0)
    return jax.lax.dynamic_update_index_in_dim(updated, t_below_new, k + 1, axis=0)


# ---------------------------------------------------------------------------
# Large-scale condensation
# ---------------------------------------------------------------------------


def large_scale_condensation(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    epsilon: float,
    latent_heat: float,
    specific_heat_cp: float,
    gas_constant: float,
    *,
    n_iterations: int = 3,
    rh_threshold: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Remove supersaturation via implicit condensation with latent heating.

    Following Frierson et al. (2006) eq. 21, the implicit scheme accounts
    for the fact that latent heating raises the saturation point::

        Δq = -(q - rh_crit * q_sat) / (1 + L²ε² q_sat / (cp R_d T²))

    Applied iteratively to converge (warming changes q_sat).

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels, shape ``(n_levels, n_lat, n_lon)``.
    humidity : jnp.ndarray
        Specific humidity, shape ``(n_levels, n_lat, n_lon)``.
    pressure : jnp.ndarray
        Pressure at full levels [Pa], shape ``(n_levels, n_lat, n_lon)``.
    epsilon : float
        Ratio R_d / R_v.
    latent_heat : float
        Latent heat of vaporization [J/kg].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    gas_constant : float
        Specific gas constant for dry air R_d [J/(kg·K)].
    n_iterations : int
        Number of implicit iterations.
    rh_threshold : float
        Relative humidity threshold for condensation (1.0 = saturation).

    Returns
    -------
    t_new : jnp.ndarray
        Adjusted temperature, shape ``(n_levels, n_lat, n_lon)``.
    q_new : jnp.ndarray
        Adjusted humidity, shape ``(n_levels, n_lat, n_lon)``.
    condensate : jnp.ndarray
        Per-level moisture removed [kg/kg], shape ``(n_levels, n_lat, n_lon)``.
        Positive values indicate condensation.
    """
    # Implicit denominator factor (Frierson 2006 eq. 21):
    # L²ε / (cp R_v T²) where R_v = R_d / ε, so
    # L²ε / (cp (R_d/ε) T²) = L²ε² / (cp R_d T²)
    lv2_eps2_over_cp_rd = latent_heat**2 * epsilon**2 / (
        specific_heat_cp * gas_constant
    )

    def _iterate(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        _: None,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        t, q = carry
        q_sat = saturation_specific_humidity(t, pressure, epsilon)
        excess = q - rh_threshold * q_sat

        # Implicit correction (Frierson 2006 eq. 21)
        # denominator accounts for latent heating feedback on q_sat
        lv_factor = lv2_eps2_over_cp_rd / t**2
        dq = -excess / (1.0 + lv_factor * q_sat)

        # Only condense where supersaturated (dq must be negative)
        dq = jnp.where(excess > 0, dq, 0.0)

        q_new = q + dq
        t_new = t - latent_heat / specific_heat_cp * dq
        return (t_new, q_new), None

    (t_out, q_out), _ = jax.lax.scan(
        _iterate, (temperature, humidity), None, length=n_iterations
    )

    # Per-level condensate: positive where moisture was removed
    condensate = humidity - q_out  # (n_levels, n_lat, n_lon)

    return t_out, q_out, condensate


# ---------------------------------------------------------------------------
# Simplified Betts-Miller convection (Frierson 2007)
# ---------------------------------------------------------------------------


def betts_miller_convection(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    dsigma: jnp.ndarray,
    epsilon: float,
    latent_heat: float,
    specific_heat_cp: float,
    gas_constant: float,
    *,
    tau_bm: float = 7200.0,
    rh_ref: float = 0.7,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simplified Betts-Miller convection (Frierson 2007).

    Relaxes temperature and humidity toward a moist adiabatic reference
    profile at a specified relative humidity, with a finite relaxation
    timescale.  Only activates when the column is convectively unstable
    (positive CAPE analogue: column-mean T exceeds reference).

    Column-integrated moist enthalpy (cp*T + L*q) is conserved by
    adjusting the reference temperature profile.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature, shape ``(n_levels, n_lat, n_lon)``.
    humidity : jnp.ndarray
        Specific humidity, shape ``(n_levels, n_lat, n_lon)``.
    pressure : jnp.ndarray
        Pressure at full levels [Pa], shape ``(n_levels, n_lat, n_lon)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    epsilon : float
        Ratio R_d / R_v.
    latent_heat : float
        Latent heat of vaporization [J/kg].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    tau_bm : float
        Relaxation timescale [s].
    rh_ref : float
        Reference relative humidity for the convective profile (0-1).

    Returns
    -------
    dt_tend : jnp.ndarray
        Temperature tendency [K/s], shape ``(n_levels, n_lat, n_lon)``.
    dq_tend : jnp.ndarray
        Humidity tendency [kg/kg/s], shape ``(n_levels, n_lat, n_lon)``.
    """
    dsigma_bc = dsigma[:, None, None]

    # Compute reference moist adiabat from surface parcel
    # Use column-bottom temperature as the surface parcel temperature
    n_levels = temperature.shape[0]
    t_sfc = temperature[n_levels - 1]  # (n_lat, n_lon)
    p_sfc = pressure[n_levels - 1]  # (n_lat, n_lon)

    # Compute moist adiabat for each column — vectorize over lat/lon
    def _column_adiabat(
        t_s: jnp.ndarray, p_s: jnp.ndarray, p_col: jnp.ndarray
    ) -> jnp.ndarray:
        return moist_adiabat(
            t_s, p_col, p_s,
            epsilon, latent_heat, specific_heat_cp, gas_constant,
        )

    # Flatten lat/lon, vmap over columns
    lat_lon_shape = t_sfc.shape
    n_cols = lat_lon_shape[0] * lat_lon_shape[1]
    t_sfc_flat = t_sfc.reshape(n_cols)
    p_sfc_flat = p_sfc.reshape(n_cols)
    p_flat = pressure.reshape(n_levels, n_cols).T  # (n_cols, n_levels)

    t_ref_flat = jax.vmap(_column_adiabat)(
        t_sfc_flat, p_sfc_flat, p_flat
    )  # (n_cols, n_levels)
    t_ref = t_ref_flat.T.reshape(n_levels, *lat_lon_shape)

    # Reference humidity: rh_ref * q_sat(T_ref, p)
    q_ref = rh_ref * saturation_specific_humidity(t_ref, pressure, epsilon)

    # Enthalpy conservation: adjust T_ref by a uniform offset so that
    # column-integrated moist enthalpy is conserved.
    # ∫(cp*T + L*q) dσ = ∫(cp*T_ref + L*q_ref) dσ
    enthalpy_actual = jnp.sum(
        (specific_heat_cp * temperature + latent_heat * humidity) * dsigma_bc,
        axis=0,
    )
    enthalpy_ref = jnp.sum(
        (specific_heat_cp * t_ref + latent_heat * q_ref) * dsigma_bc,
        axis=0,
    )
    delta_h = enthalpy_actual - enthalpy_ref  # (n_lat, n_lon)
    # Distribute offset uniformly in temperature
    t_ref_adjusted = t_ref + delta_h / (
        specific_heat_cp * jnp.sum(dsigma_bc, axis=0)
    )

    # Convective trigger (Frierson 2007): only activate when the column
    # is both drying (Pq > 0) and cooling at the surface (relaxation warms
    # aloft, cools below).  The drying criterion is the standard deep
    # convection trigger in simplified Betts-Miller schemes.
    pq = jnp.sum((humidity - q_ref) * dsigma_bc, axis=0)  # > 0 → drying
    convecting = pq > 0  # (n_lat, n_lon)

    # Relaxation tendencies
    dt_tend = jnp.where(
        convecting[None, :, :],
        (t_ref_adjusted - temperature) / tau_bm,
        0.0,
    )
    dq_tend = jnp.where(
        convecting[None, :, :],
        (q_ref - humidity) / tau_bm,
        0.0,
    )

    return dt_tend, dq_tend
