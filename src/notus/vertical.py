"""Vertical operators for the primitive equations.

Hydrostatic geopotential
~~~~~~~~~~~~~~~~~~~~~~~~
The geopotential at each sigma level is obtained by vertically integrating the
hydrostatic relation ∂Φ/∂ln(σ) = -R·T.  This is a linear operation in T, so
it works directly on spectral coefficients.  The integration is encoded in a
weight matrix G such that Φ = Φ_surface + G · T, where G is upper-triangular
and built from log-sigma ratios (Durran §8.6.5, Dinosaur/NeuralGCM).

Continuity equation
~~~~~~~~~~~~~~~~~~~
The sigma-coordinate continuity equation provides two diagnostics:

1. **Surface pressure tendency**: ∂ln(pₛ)/∂t = -Σₖ Dₖ* · Δσₖ
   where Dₖ* = Dₖ + v⃗ₖ·∇ln(pₛ) is the mass-flux divergence.

2. **Sigma-dot** (vertical velocity in sigma coordinates):
   σ̇_{k+1/2} = σ_{k+1/2} · C_L - Cₖ
   where Cₖ = Σⱼ₌₁ᵏ Dⱼ* · Δσⱼ is the cumulative integral from the top,
   and C_L is the total column integral.

Both σ̇ = 0 at the boundaries (σ = 0, σ = 1) by construction.

References
----------
- Durran, "Numerical Methods for Fluid Dynamics", §8.6.3 and §8.6.5
- Dinosaur: primitive_equations.py, sigma_coordinates.py
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

from notus.sigma import SigmaLevels


def sigma_ratios(levels: SigmaLevels) -> np.ndarray:
    """Compute log-sigma ratios used as weights in the hydrostatic integration.

    Following Durran §8.6.5 (and Dinosaur's ``get_sigma_ratios``):

        α[j] = ln(σ_center[j+1] / σ_center[j]) / 2    for j < L-1
        α[L-1] = -ln(σ_center[L-1])                     (bottom level)

    Parameters
    ----------
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    np.ndarray
        Alpha vector, shape ``(n_levels,)``.
    """
    centers = np.asarray(levels.sigma_full)
    alpha: np.ndarray = np.diff(np.log(centers), append=0) / 2
    alpha[-1] = -np.log(centers[-1])
    return alpha


def geopotential_weights(levels: SigmaLevels, gas_constant: float) -> np.ndarray:
    """Build the geopotential weight matrix G.

    The matrix G encodes the hydrostatic vertical integration so that:

        Φ[k] = Φ_surface + Σ_j G[k,j] · T[j]

    G is upper triangular with structure::

                 α[0]    α[0]+α[1]   α[1]+α[2]   ...
        G / R =  0       α[1]        α[1]+α[2]   ...
                 0       0           α[2]         ...
                 ⋮       ⋮            ⋮           ⋱

    Parameters
    ----------
    levels : SigmaLevels
        Sigma vertical coordinate.
    gas_constant : float
        Specific gas constant R_d [J/(kg·K)].

    Returns
    -------
    np.ndarray
        Weight matrix, shape ``(n_levels, n_levels)``.
    """
    alpha = sigma_ratios(levels)
    n = levels.n_levels
    weights = np.zeros((n, n))
    for j in range(n):
        weights[j, j] = alpha[j]
        for k in range(j + 1, n):
            weights[j, k] = alpha[k] + alpha[k - 1]
    return gas_constant * weights


def geopotential(
    temperature: jnp.ndarray,
    surface_geopotential: jnp.ndarray,
    levels: SigmaLevels,
    gas_constant: float,
) -> jnp.ndarray:
    """Compute geopotential at each sigma level from the hydrostatic equation.

    Evaluates Φ = Φ_surface + G · T where G is the Durran weight matrix.
    The operation is linear in T, so it works directly on spectral coefficients.

    Parameters
    ----------
    temperature : jnp.ndarray
        Temperature at full levels, shape ``(n_levels, n_spectral)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential Φ_s = g·z_s, shape ``(n_spectral,)``.
    levels : SigmaLevels
        Sigma vertical coordinate.
    gas_constant : float
        Specific gas constant R_d [J/(kg·K)].

    Returns
    -------
    jnp.ndarray
        Geopotential at full levels, shape ``(n_levels, n_spectral)``.
    """
    sigma_full_key = tuple(np.asarray(levels.sigma_full).tolist())
    weights = _cached_geopotential_weights(levels.n_levels, sigma_full_key, gas_constant)
    phi_diff = jnp.einsum("kj,j...->k...", weights, temperature)
    return surface_geopotential + phi_diff


@functools.lru_cache(maxsize=16)
def _cached_geopotential_weights(
    n_levels: int,
    sigma_full_tuple: tuple[float, ...],
    gas_constant: float,
) -> jnp.ndarray:
    """Cached version of geopotential weights, keyed by hashable args."""
    centers = np.array(sigma_full_tuple)
    alpha = np.diff(np.log(centers), append=0) / 2
    alpha[-1] = -np.log(centers[-1])

    weights = np.zeros((n_levels, n_levels))
    for j in range(n_levels):
        weights[j, j] = alpha[j]
        for k in range(j + 1, n_levels):
            weights[j, k] = alpha[k] + alpha[k - 1]
    return jnp.array(gas_constant * weights)


def surface_pressure_tendency(
    column_divergence: jnp.ndarray,
    levels: SigmaLevels,
) -> jnp.ndarray:
    """Compute the surface pressure tendency from the continuity equation.

    ∂ln(pₛ)/∂t = -Σₖ Dₖ* · Δσₖ

    where Dₖ* is the mass-flux divergence at each level (typically
    D + v⃗·∇ln pₛ, but this function is agnostic — the caller decides
    what to pass in).

    Parameters
    ----------
    column_divergence : jnp.ndarray
        Mass-flux divergence at full levels, shape ``(n_levels, ...)``.
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    jnp.ndarray
        Tendency ∂ln(pₛ)/∂t, shape ``(...)``.
    """
    dsigma = levels.dsigma
    # Broadcast Δσ to match column_divergence: (n_levels,) → (n_levels, 1, ...)
    extra_dims = column_divergence.ndim - 1
    dsigma_bc = jnp.reshape(dsigma, (-1,) + (1,) * extra_dims)
    return -jnp.sum(column_divergence * dsigma_bc, axis=0)


def sigma_dot(
    column_divergence: jnp.ndarray,
    levels: SigmaLevels,
) -> jnp.ndarray:
    """Compute sigma-dot (vertical velocity) at half levels.

    σ̇_{k+1/2} = σ_{k+1/2} · C_L - Cₖ

    where Cₖ = Σⱼ₌₁ᵏ Dⱼ* · Δσⱼ is the cumulative integral from the top,
    and C_L is the total column integral.

    σ̇ = 0 at both boundaries (σ = 0 and σ = 1) by construction.

    Parameters
    ----------
    column_divergence : jnp.ndarray
        Mass-flux divergence at full levels, shape ``(n_levels, ...)``.
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    jnp.ndarray
        Sigma-dot at half levels (interfaces), shape ``(n_levels + 1, ...)``.
        Index 0 = top (σ̇ = 0), index n_levels = surface (σ̇ = 0).
    """
    dsigma = levels.dsigma
    sigma_half = levels.sigma_half

    # Broadcast Δσ: (n_levels,) → (n_levels, 1, ...)
    extra_dims = column_divergence.ndim - 1
    dsigma_bc = jnp.reshape(dsigma, (-1,) + (1,) * extra_dims)

    # Weighted column: D_k * Δσ_k at each level
    weighted = column_divergence * dsigma_bc

    # Cumulative integral from top: C_k = Σⱼ₌₁ᵏ Dⱼ · Δσⱼ
    # cumsum gives [C_1, C_2, ..., C_L] with shape (n_levels, ...)
    cumulative = jnp.cumsum(weighted, axis=0)

    # Total column integral C_L (last element)
    c_total = cumulative[-1:]  # keep dim for broadcasting, shape (1, ...)

    # σ at half levels: (n_levels + 1,) → (n_levels + 1, 1, ...)
    sigma_bc = jnp.reshape(sigma_half, (-1,) + (1,) * extra_dims)

    # σ̇ at internal half levels k+1/2 for k = 0..L-1:
    # σ̇_{k+1/2} = σ_{k+1/2} · C_L - C_k
    # cumulative[k] = C_{k+1} (cumsum is 0-indexed: cumulative[0] = C_1)
    # So internal interfaces 1..L-1 use cumulative[0..L-2]
    # and interface 0 (top) has C_0 = 0 → σ̇ = 0
    # and interface L (surface) has σ = 1, C_L → σ̇ = C_L - C_L = 0

    # Build the full array including boundaries
    trailing_shape = column_divergence.shape[1:]
    zero_boundary = jnp.zeros((1, *trailing_shape))

    # Internal interfaces: σ̇ at half levels 1 through L-1
    # sigma_half[1:-1] are the internal interfaces
    # cumulative[0:L-1] = C_1 through C_{L-1}
    internal_sigma = sigma_bc[1:-1]  # (L-1, ...)
    internal_cumulative = cumulative[:-1]  # C_1 through C_{L-1}
    internal_sigma_dot = internal_sigma * c_total - internal_cumulative

    return jnp.concatenate([zero_boundary, internal_sigma_dot, zero_boundary], axis=0)


def vertical_advection(
    sigma_dot_half: jnp.ndarray,
    field: jnp.ndarray,
    levels: SigmaLevels,
) -> jnp.ndarray:
    """Compute vertical advection tendency using centered 2nd-order differences.

    Returns ``-σ̇ · ∂f/∂σ`` at full levels, following Dinosaur's centered
    vertical advection scheme.  The computation:

    1. ∂f/∂σ at internal half levels via centered differences, with
       ∂f/∂σ = 0 at top and bottom boundaries.
    2. Multiply by σ̇ at each half level.
    3. Average the product to full levels.

    Parameters
    ----------
    sigma_dot_half : jnp.ndarray
        Sigma-dot at half levels (interfaces), shape ``(n_levels + 1, ...)``.
        Includes boundary values (typically zero at top and bottom).
    field : jnp.ndarray
        Field at full levels, shape ``(n_levels, ...)``.
    levels : SigmaLevels
        Sigma vertical coordinate.

    Returns
    -------
    jnp.ndarray
        Vertical advection tendency ``-σ̇ · ∂f/∂σ`` at full levels,
        shape ``(n_levels, ...)``.
    """
    # ∂f/∂σ at internal half levels (n_levels - 1 values)
    # Uses center-to-center distances for the denominator
    sigma_full = levels.sigma_full
    center_to_center = sigma_full[1:] - sigma_full[:-1]  # (n_levels - 1,)

    # Broadcast center_to_center to match field dimensions
    extra_dims = field.ndim - 1
    c2c_bc = jnp.reshape(center_to_center, (-1,) + (1,) * extra_dims)

    df = field[1:] - field[:-1]  # (n_levels - 1, ...)
    df_dsigma_internal = df / c2c_bc  # (n_levels - 1, ...)

    # Pad with zero boundary conditions for ∂f/∂σ at top and bottom
    trailing_shape = field.shape[1:]
    zero_bc = jnp.zeros((1, *trailing_shape))
    df_dsigma = jnp.concatenate(
        [zero_bc, df_dsigma_internal, zero_bc], axis=0
    )  # (n_levels + 1, ...)

    # Product σ̇ · ∂f/∂σ at all half levels
    product = sigma_dot_half * df_dsigma  # (n_levels + 1, ...)

    # Average to full levels and negate (tendency = -σ̇ · ∂f/∂σ)
    return -0.5 * (product[1:] + product[:-1])
