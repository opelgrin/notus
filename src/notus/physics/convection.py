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
    lv2_eps2_over_cp_rd = latent_heat**2 * epsilon**2 / (specific_heat_cp * gas_constant)

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

    (t_out, q_out), _ = jax.lax.scan(_iterate, (temperature, humidity), None, length=n_iterations)

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

    Lifts a parcel from the lowest level along a moist pseudoadiabat,
    finds the level of zero buoyancy (LZB), and relaxes temperature and
    humidity toward the reference profile only between the surface and
    LZB.  Implements deep and shallow convection following the Frierson
    (2007) qref formulation (SpeedyWeather.jl convention).

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
    dsigma_bc = dsigma[:, None, None]  # (n_levels, 1, 1)
    n_levels = temperature.shape[0]

    # --- Step 1: Compute moist adiabat reference from lowest level ---
    t_sfc = temperature[n_levels - 1]  # (n_lat, n_lon)
    p_sfc = pressure[n_levels - 1]

    def _column_adiabat(t_s: jnp.ndarray, p_s: jnp.ndarray, p_col: jnp.ndarray) -> jnp.ndarray:
        return moist_adiabat(
            t_s,
            p_col,
            p_s,
            epsilon,
            latent_heat,
            specific_heat_cp,
            gas_constant,
        )

    lat_lon_shape = t_sfc.shape
    n_cols = lat_lon_shape[0] * lat_lon_shape[1]
    t_ref_flat = jax.vmap(_column_adiabat)(
        t_sfc.reshape(n_cols),
        p_sfc.reshape(n_cols),
        pressure.reshape(n_levels, n_cols).T,
    )  # (n_cols, n_levels)
    t_ref = t_ref_flat.T.reshape(n_levels, *lat_lon_shape)

    # Reference humidity
    q_ref = rh_ref * saturation_specific_humidity(t_ref, pressure, epsilon)

    # --- Step 2: Find level of zero buoyancy (LZB) per column ---
    # Virtual temperature: T_v = T * (1 + (1/ε - 1) * q)
    eps_inv_m1 = 1.0 / epsilon - 1.0
    tv_env = temperature * (1.0 + eps_inv_m1 * humidity)
    q_sat_ref = saturation_specific_humidity(t_ref, pressure, epsilon)
    tv_ref = t_ref * (1.0 + eps_inv_m1 * q_sat_ref)

    # Parcel is buoyant where T_v_ref > T_v_env (levels ordered top→bottom)
    buoyant = tv_ref > tv_env  # (n_levels, n_lat, n_lon)

    # LZB mask: True at levels between LZB and surface (inclusive).
    # A level is below LZB if it and all levels below it are buoyant,
    # or more practically: scan from the surface upward and mask all
    # buoyant levels until the first non-buoyant level.
    # We use cumulative product from the bottom: a level is in the
    # convective column if all levels from the surface up to it are buoyant.
    buoyant_from_bottom = jnp.cumprod(buoyant[::-1], axis=0)[::-1]
    below_lzb = buoyant_from_bottom.astype(jnp.bool_)

    # Require at least 2 levels of buoyancy for convection to activate
    min_buoyant_levels = 2
    n_buoyant_levels = jnp.sum(below_lzb, axis=0)  # (n_lat, n_lon)
    has_convection = n_buoyant_levels >= min_buoyant_levels

    # --- Step 3: Compute Pq and PT integrals (surface to LZB only) ---
    masked_dsigma = jnp.where(below_lzb, dsigma_bc, 0.0)

    # Pq = ∫(q - q_ref) dσ over convective column (positive → drying)
    pq = jnp.sum((humidity - q_ref) * masked_dsigma, axis=0)

    # PT = -∫(T - T_ref) dσ over convective column (positive → cooling)
    pt = -jnp.sum((temperature - t_ref) * masked_dsigma, axis=0)

    # Depth of convective column in σ
    dsigma_lzb = jnp.sum(masked_dsigma, axis=0)
    dsigma_lzb_safe = jnp.maximum(dsigma_lzb, 1e-10)

    # --- Step 4: Deep vs. shallow classification ---
    deep = has_convection & (pq > 0) & (pt > 0)
    shallow = has_convection & (pq <= 0) & (pt > 0)

    # --- Step 5: Enthalpy-conserving T offset (Frierson 2007 eq. 5-6) ---
    # Deep: enthalpy conservation requires ΔT offset over convective depth
    delta_t_deep = (pt - pq * latent_heat / specific_heat_cp) / dsigma_lzb_safe

    # Shallow (qref formulation, Frierson 2007 eq. 11-15):
    # Rescale q_ref so Pq → 0, then ΔT = PT / Δσ_LZB
    q_ref_sum = jnp.sum(q_ref * masked_dsigma, axis=0)
    q_ref_sum_safe = jnp.maximum(jnp.abs(q_ref_sum), 1e-20)
    fq = 1.0 - pq / q_ref_sum_safe  # scaling factor
    # Apply fq only for shallow convection
    fq_applied = jnp.where(shallow, fq, 1.0)[None, :, :]
    q_ref_adjusted = q_ref * fq_applied

    delta_t_shallow = pt / dsigma_lzb_safe

    # Select the appropriate ΔT
    delta_t = jnp.where(deep, delta_t_deep, 0.0)
    delta_t = jnp.where(shallow, delta_t_shallow, delta_t)

    # Adjust T_ref
    t_ref_adjusted = t_ref - delta_t[None, :, :]

    # Use adjusted q_ref (only differs for shallow)
    q_ref_final = jnp.where(shallow[None, :, :], q_ref_adjusted, q_ref)

    # --- Step 6: Compute tendencies (only below LZB) ---
    active = (deep | shallow)[None, :, :]  # (1, n_lat, n_lon)
    level_mask = below_lzb & active  # (n_levels, n_lat, n_lon)

    dt_tend = jnp.where(
        level_mask,
        (t_ref_adjusted - temperature) / tau_bm,
        0.0,
    )
    dq_tend = jnp.where(
        level_mask,
        (q_ref_final - humidity) / tau_bm,
        0.0,
    )

    return dt_tend, dq_tend
