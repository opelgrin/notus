"""Cached spectral index arrays and coupling matrices.

These are private implementation details used by the public operator functions
in :mod:`notus.operators.core` and :mod:`notus.operators.vector`.  They are
computed once per truncation and reused via :func:`functools.lru_cache`.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp


def _epsilon(n: int, m: int) -> float:
    """Coupling coefficient ε(n,m) = √((n²−m²)/(4n²−1))."""
    if n == 0:
        return 0.0
    return ((n * n - m * m) / (4.0 * n * n - 1.0)) ** 0.5


def _spectral_index(truncation: int, m: int, n: int) -> int:
    """Linear index into lower-triangular spectral storage."""
    return m * (truncation + 1) - m * (m - 1) // 2 + (n - m)


@functools.lru_cache(maxsize=16)
def _laplacian_eigenvalues(truncation: int, radius: float) -> jnp.ndarray:
    """Pre-compute -n(n+1)/a² for all spectral indices."""
    vals = [
        -n * (n + 1) / radius**2 for m in range(truncation + 1) for n in range(m, truncation + 1)
    ]
    return jnp.array(vals)


@functools.lru_cache(maxsize=16)
def _m_index_array(truncation: int) -> jnp.ndarray:
    """Array of zonal wavenumber m for each spectral index."""
    vals = [m for m in range(truncation + 1) for _n in range(m, truncation + 1)]
    return jnp.array(vals, dtype=jnp.float64)


@functools.lru_cache(maxsize=16)
def _n_index_array(truncation: int) -> jnp.ndarray:
    """Array of total wavenumber n for each spectral index."""
    vals = [n for m in range(truncation + 1) for n in range(m, truncation + 1)]
    return jnp.array(vals, dtype=jnp.float64)


@functools.lru_cache(maxsize=16)
def _meridional_coupling(
    truncation: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pre-compute index arrays and coefficients for the meridional derivative.

    The spectral recurrence (gathering by output index n):

        [cos(φ)·∂f/∂φ]ₙᵐ = (n+2)·ε(n+1,m)·f_{n+1}^m − (n−1)·ε(n,m)·f_{n−1}^m

    where ε(n,m) = √((n²−m²)/(4n²−1)).

    Returns
    -------
    tuple of (idx_lower, idx_upper, coeff_lower, coeff_upper)
        idx_lower[i]  : flat index of (m, n−1), self-referencing if n == m
        idx_upper[i]  : flat index of (m, n+1), self-referencing if n == T
        coeff_lower[i]: −(n−1)·ε(n,m), zero if n == m
        coeff_upper[i]: (n+2)·ε(n+1,m), zero if n == T
    """
    t = truncation

    idx_lower = []
    idx_upper = []
    c_lower = []
    c_upper = []

    for m in range(t + 1):
        for n in range(m, t + 1):
            i = _spectral_index(t, m, n)

            # Lower coupling: reads f_{n-1}^m, coefficient -(n-1)*eps(n,m)
            if n > m:
                idx_lower.append(_spectral_index(t, m, n - 1))
                c_lower.append(-(n - 1) * _epsilon(n, m))
            else:
                idx_lower.append(i)
                c_lower.append(0.0)

            # Upper coupling: reads f_{n+1}^m, coefficient (n+2)*eps(n+1,m)
            if n < t:
                idx_upper.append(_spectral_index(t, m, n + 1))
                c_upper.append((n + 2) * _epsilon(n + 1, m))
            else:
                idx_upper.append(i)
                c_upper.append(0.0)

    return (
        jnp.array(idx_lower, dtype=jnp.int32),
        jnp.array(idx_upper, dtype=jnp.int32),
        jnp.array(c_lower),
        jnp.array(c_upper),
    )


def _mu_derivative(
    coeffs: jnp.ndarray,
    truncation: int,
) -> jnp.ndarray:
    """Compute d/dμ in spectral space via integration-by-parts recurrence.

    This is NOT the same as meridional_derivative (which computes cos²φ · d/dμ).
    This operator arises when computing the spectral representation of d(G)/dμ
    via Gaussian quadrature and integration by parts:

        [dG/dμ]ₙᵐ = −(n+1)·ε(n,m)·Ĝ_{n−1}^m + n·ε(n+1,m)·Ĝ_{n+1}^m

    Used internally by spectral_divergence and spectral_curl.
    """
    idx_lower, idx_upper, c_lower, c_upper = _mu_derivative_coupling(truncation)
    return c_lower * coeffs[idx_lower] + c_upper * coeffs[idx_upper]


@functools.lru_cache(maxsize=16)
def _mu_derivative_coupling(
    truncation: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pre-compute coupling arrays for the μ-derivative recurrence.

    [dG/dμ]ₙᵐ = −(n+1)·ε(n,m)·Ĝ_{n−1} + n·ε(n+1,m)·Ĝ_{n+1}
    """
    t = truncation

    idx_lower = []
    idx_upper = []
    c_lower = []
    c_upper = []

    for m in range(t + 1):
        for n in range(m, t + 1):
            i = _spectral_index(t, m, n)

            # Lower: reads f_{n-1}, coefficient = -(n+1)*eps(n,m)
            if n > m:
                idx_lower.append(_spectral_index(t, m, n - 1))
                c_lower.append(-(n + 1) * _epsilon(n, m))
            else:
                idx_lower.append(i)
                c_lower.append(0.0)

            # Upper: reads f_{n+1}, coefficient = n*eps(n+1,m)
            if n < t:
                idx_upper.append(_spectral_index(t, m, n + 1))
                c_upper.append(n * _epsilon(n + 1, m))
            else:
                idx_upper.append(i)
                c_upper.append(0.0)

    return (
        jnp.array(idx_lower, dtype=jnp.int32),
        jnp.array(idx_upper, dtype=jnp.int32),
        jnp.array(c_lower),
        jnp.array(c_upper),
    )
