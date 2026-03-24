"""Pure compute functions for spectral operator arrays.

These functions are called once by :meth:`OperatorArrays.build
<notus.operators.arrays.OperatorArrays.build>` and should not be called
directly at runtime.  There is no caching here; caching is the caller's
responsibility (via :class:`~notus.operators.arrays.OperatorArrays`).
"""

from __future__ import annotations

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Elementary helpers (no arrays involved)
# ---------------------------------------------------------------------------


def _epsilon(n: int, m: int) -> float:
    """Coupling coefficient ε(n,m) = √((n²−m²)/(4n²−1))."""
    if n == 0:
        return 0.0
    return ((n * n - m * m) / (4.0 * n * n - 1.0)) ** 0.5


def _spectral_index(truncation: int, m: int, n: int) -> int:
    """Linear index into lower-triangular spectral storage."""
    return m * (truncation + 1) - m * (m - 1) // 2 + (n - m)


# ---------------------------------------------------------------------------
# Array builders (called once by OperatorArrays.build)
# ---------------------------------------------------------------------------


def _compute_laplacian_eigenvalues(truncation: int, radius: float) -> jnp.ndarray:
    """Pre-compute -n(n+1)/a² for all spectral indices."""
    vals = [
        -n * (n + 1) / radius**2 for m in range(truncation + 1) for n in range(m, truncation + 1)
    ]
    return jnp.array(vals)


def _compute_m_index_array(truncation: int) -> jnp.ndarray:
    """Array of zonal wavenumber m for each spectral index."""
    vals = [m for m in range(truncation + 1) for _n in range(m, truncation + 1)]
    return jnp.array(vals, dtype=jnp.float64)


def _compute_n_index_array(truncation: int) -> jnp.ndarray:
    """Array of total wavenumber n for each spectral index."""
    vals = [n for m in range(truncation + 1) for n in range(m, truncation + 1)]
    return jnp.array(vals, dtype=jnp.float64)


def _compute_meridional_coupling(
    truncation: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pre-compute index arrays and coefficients for the meridional derivative.

    The spectral recurrence (gathering by output index n):

        [cos(φ)·∂f/∂φ]ₙᵐ = (n+2)·ε(n+1,m)·f_{n+1}^m − (n−1)·ε(n,m)·f_{n−1}^m

    Returns (idx_lower, idx_upper, coeff_lower, coeff_upper).
    """
    t = truncation
    idx_lower, idx_upper, c_lower, c_upper = [], [], [], []

    for m in range(t + 1):
        for n in range(m, t + 1):
            i = _spectral_index(t, m, n)
            if n > m:
                idx_lower.append(_spectral_index(t, m, n - 1))
                c_lower.append(-(n - 1) * _epsilon(n, m))
            else:
                idx_lower.append(i)
                c_lower.append(0.0)
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


def _compute_mu_derivative_coupling(
    truncation: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pre-compute coupling arrays for the μ-derivative recurrence.

    [dG/dμ]ₙᵐ = −(n+1)·ε(n,m)·Ĝ_{n−1} + n·ε(n+1,m)·Ĝ_{n+1}
    """
    t = truncation
    idx_lower, idx_upper, c_lower, c_upper = [], [], [], []

    for m in range(t + 1):
        for n in range(m, t + 1):
            i = _spectral_index(t, m, n)
            if n > m:
                idx_lower.append(_spectral_index(t, m, n - 1))
                c_lower.append(-(n + 1) * _epsilon(n, m))
            else:
                idx_lower.append(i)
                c_lower.append(0.0)
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
