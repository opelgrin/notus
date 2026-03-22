"""Associated Legendre polynomials for spherical harmonic transforms.

We use the *fully normalized* associated Legendre polynomials (also called
"4π-normalized" or "geodesy normalization"), defined so that:

    ∫₋₁¹ [P̄ₙᵐ(x)]² dx = 2 / (2n + 1)     [for all n, m]

which makes the spherical harmonics orthonormal over the unit sphere:

    ∫∫ Yₙᵐ Yₙ'ᵐ'* dΩ = δ_{nn'} δ_{mm'}

The recurrence relation for P̄ₙᵐ(x) with x = sin(lat) = μ is:

    P̄ₙᵐ(μ) = aₙᵐ · μ · P̄ₙ₋₁ᵐ(μ)  -  bₙᵐ · P̄ₙ₋₂ᵐ(μ)

with:
    aₙᵐ = √[(2n-1)(2n+1) / ((n-m)(n+m))]
    bₙᵐ = √[(2n+1)(n+m-1)(n-m-1) / ((n-m)(n+m)(2n-3))]

Starting values:
    P̄ₘᵐ(μ) = √[(2m+1)!! / (2m)!!] · (1-μ²)^{m/2} · (-1)^m
             (computed iteratively to avoid overflow)
    P̄ₘ₊₁ᵐ(μ) = aₘ₊₁ᵐ · μ · P̄ₘᵐ(μ)

All computations use JAX for GPU/JIT compatibility.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def compute_legendre_polynomials(
    truncation: int,
    sin_lat: jnp.ndarray,
) -> jnp.ndarray:
    """Compute fully-normalized associated Legendre polynomials.

    Parameters
    ----------
    truncation : int
        Maximum total wavenumber T.
    sin_lat : jnp.ndarray
        Sine of Gaussian latitudes, shape ``(n_lat,)``.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(n_lat, n_spectral)`` where ``n_spectral = (T+1)(T+2)/2``.
        Column ordering follows the lower-triangular convention:
        for each m from 0 to T, n runs from m to T.
    """
    return _compute_legendre_impl(truncation, sin_lat)


@functools.partial(jax.jit, static_argnums=(0,))
def _compute_legendre_impl(truncation: int, sin_lat: jnp.ndarray) -> jnp.ndarray:
    n_lat = sin_lat.shape[0]
    n_spectral = (truncation + 1) * (truncation + 2) // 2
    cos_lat = jnp.sqrt(1.0 - sin_lat**2)

    # Pre-compute recurrence coefficients (on-device).
    a_coeffs, b_coeffs = _recurrence_coefficients(truncation)

    # We build the result column by column (one m at a time).
    # Each column for fixed m has length (T - m + 1) entries for n = m..T.
    # We'll fill a flat array using the same indexing as GaussianGrid.spectral_index.

    result = jnp.zeros((n_lat, n_spectral))

    def body_m(m: int, result: jnp.ndarray) -> jnp.ndarray:
        """Compute all P̄ₙᵐ for n = m..T at all latitudes."""
        idx_mm = _spectral_index(truncation, m, m)

        # Compute sectoral value P̄ₘᵐ(μ)
        p_mm = _sectoral_value(m, cos_lat)
        result = result.at[:, idx_mm].set(p_mm)

        # First tesseral value (n = m+1)
        def set_mp1(result: jnp.ndarray) -> jnp.ndarray:
            idx_mp1 = _spectral_index(truncation, m, m + 1)
            p_mp1 = a_coeffs[idx_mp1] * sin_lat * p_mm
            return result.at[:, idx_mp1].set(p_mp1)

        result = jax.lax.cond(m < truncation, set_mp1, lambda r: r, result)

        # Remaining tesseral values via three-term recurrence (n = m+2 .. T)
        def body_n(n_minus_m: int, carry: jnp.ndarray) -> jnp.ndarray:
            n = m + n_minus_m
            idx_n = _spectral_index(truncation, m, n)
            idx_n1 = _spectral_index(truncation, m, n - 1)
            idx_n2 = _spectral_index(truncation, m, n - 2)
            p_n = a_coeffs[idx_n] * sin_lat * carry[:, idx_n1] - b_coeffs[idx_n] * carry[:, idx_n2]
            return carry.at[:, idx_n].set(p_n)

        n_tesseral = truncation - m - 1
        return jax.lax.fori_loop(2, jnp.maximum(2, n_tesseral + 2), body_n, result)

    return jax.lax.fori_loop(0, truncation + 1, body_m, result)


def _sectoral_value(m: int, cos_lat: jnp.ndarray) -> jnp.ndarray:
    """Compute P̄ₘᵐ(μ) = sectoral Legendre polynomial.

    Built iteratively to avoid overflow from double factorials:
        P̄₀⁰ = 1
        P̄ₘᵐ = cos(lat) · √((2m+1)/(2m)) · P̄ₘ₋₁ᵐ⁻¹
    """

    def body(i: int, p: jnp.ndarray) -> jnp.ndarray:
        factor = jnp.sqrt((2.0 * i + 1.0) / (2.0 * i))
        return p * cos_lat * factor

    return jax.lax.fori_loop(1, m + 1, body, jnp.ones_like(cos_lat))


def _recurrence_coefficients(truncation: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-compute the a and b recurrence coefficients for all (m, n).

    Returns flat arrays indexed by spectral_index(m, n).
    """
    n_spectral = (truncation + 1) * (truncation + 2) // 2
    a = jnp.zeros(n_spectral)
    b = jnp.zeros(n_spectral)

    for m in range(truncation + 1):
        for n in range(m + 1, truncation + 1):
            idx = _spectral_index(truncation, m, n)
            nm = float(n - m)
            np_ = float(n + m)
            a_val = ((2.0 * n - 1.0) * (2.0 * n + 1.0) / (nm * np_)) ** 0.5
            if n >= m + 2:
                b_val = (
                    (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0) / (nm * np_ * (2.0 * n - 3.0))
                ) ** 0.5
            else:
                b_val = 0.0
            a = a.at[idx].set(a_val)
            b = b.at[idx].set(b_val)

    return a, b


def _spectral_index(truncation: int, m: int, n: int) -> int:
    """Linear index into lower-triangular spectral storage.

    Same convention as GaussianGrid.spectral_index.
    """
    return m * (truncation + 1) - m * (m - 1) // 2 + (n - m)
