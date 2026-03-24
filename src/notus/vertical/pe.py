"""PE-specific vertical operators.

Contains quantities that are specific to the primitive equation formulation
and depend on the vertical structure of the model state.
"""

from __future__ import annotations

import jax.numpy as jnp

from notus.vertical.operators import sigma_ratios
from notus.vertical.sigma import SigmaLevels


def omega_over_pressure(
    g_term: jnp.ndarray,
    v_dot_grad_lnps: jnp.ndarray,
    levels: SigmaLevels,
) -> jnp.ndarray:
    """Compute ω/p (pressure velocity / pressure) at full levels.

    Uses the Durran (§8.6.3, eq. 8.124) approximation, following Dinosaur's
    ``_t_omega_over_sigma_sp``:

        (ω/p)_k = v⃗·∇ln(pₛ)_k - (1/Δσ_k)·[α_k·F_k + α_{k-1}·F_{k-1}]

    where F_k = Σⱼ₌₀^{k-1} G_j · Δσ_j is the cumulative integral of
    ``g_term`` from the top, and α are the sigma ratios.

    The caller decides what to pass as ``g_term``:
    - For the T_ref part of adiabatic heating: g_term = v⃗·∇ln(pₛ)
    - For the T' part: g_term = v⃗·∇ln(pₛ) + D

    Parameters
    ----------
    g_term : jnp.ndarray
        Mass-flux divergence at full levels, shape ``(n_levels, ...)``.
    v_dot_grad_lnps : jnp.ndarray
        v⃗·∇ln(pₛ) at full levels, shape ``(n_levels, ...)``.
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    jnp.ndarray
        ω/p at full levels, shape ``(n_levels, ...)``.
    """
    dsigma = levels.dsigma
    alpha = sigma_ratios(levels)

    extra_dims = g_term.ndim - 1
    dsigma_bc = jnp.reshape(dsigma, (-1,) + (1,) * extra_dims)
    alpha_bc = jnp.reshape(jnp.array(alpha), (-1,) + (1,) * extra_dims)

    # Cumulative integral from top: F[k] = Σⱼ₌₀ᵏ G_j · Δσ_j
    # f[k] corresponds to F_{k+1} in the mathematical notation
    f = jnp.cumsum(g_term * dsigma_bc, axis=0)  # (n_levels, ...)

    # α[k] · F_{k+1}
    alpha_f = alpha_bc * f

    # α[k-1] · F_k: shift by padding a zero at the top
    trailing_shape = g_term.shape[1:]
    zero_pad = jnp.zeros((1, *trailing_shape))
    alpha_f_shifted = jnp.concatenate([zero_pad, alpha_f[:-1]], axis=0)

    g_part = (alpha_f + alpha_f_shifted) / dsigma_bc

    return v_dot_grad_lnps - g_part
