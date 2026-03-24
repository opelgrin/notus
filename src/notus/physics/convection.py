"""Dry convective adjustment (Manabe & Strickler, 1964).

Bottom-up pairwise adjustment that restores statically unstable columns
to the dry adiabatic lapse rate while conserving column-integrated enthalpy.
All functions are JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
