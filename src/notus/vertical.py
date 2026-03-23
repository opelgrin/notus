"""Vertical operators for the primitive equations.

Hydrostatic geopotential computation using the Durran (§8.6.5) discretization,
following the Dinosaur/NeuralGCM formulation.

The geopotential at each sigma level is obtained by vertically integrating the
hydrostatic relation:

    ∂Φ/∂ln(σ) = -R·T

This is a linear operation in T, so it works directly on spectral coefficients.
The integration is encoded in a weight matrix G such that:

    Φ = Φ_surface + G · T

where G is an upper-triangular matrix built from log-sigma ratios (α).

References
----------
- Durran, "Numerical Methods for Fluid Dynamics", §8.6.5
- Dinosaur: primitive_equations.py (get_sigma_ratios, get_geopotential_weights)
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
    weights = _cached_geopotential_weights(
        levels.n_levels, sigma_full_key, gas_constant
    )
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
