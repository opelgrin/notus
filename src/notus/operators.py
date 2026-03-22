"""Spectral differential operators on the sphere.

All operators work on spectral coefficients and return spectral coefficients.
They exploit the fact that differentiation in spectral space is just
multiplication by wavenumber-dependent factors.

Key identities for a scalar f with spectral coefficients f̂ₙᵐ:

- **Laplacian:** ∇²f has coefficients  -n(n+1)/a² · f̂ₙᵐ
- **Inverse Laplacian:** ∇⁻²f has coefficients  -a²/[n(n+1)] · f̂ₙᵐ  (n>0)

For vector operations (gradient, divergence, curl), we use the
vorticity-divergence formulation where the wind is decomposed as:

    u = -1/(a cos φ) ∂ψ/∂φ + 1/(a cos φ) ∂χ/∂λ
    v =  1/a ∂ψ/∂λ/(cos φ) ... [actually need careful treatment]

More precisely, from streamfunction ψ and velocity potential χ:
    ζ = ∇²ψ  (vorticity)
    δ = ∇²χ  (divergence)

So:  ψ = ∇⁻²ζ,  χ = ∇⁻²δ

The gradient in spectral space uses recurrence relations on P̄ₙᵐ for
the meridional derivative.  However, it is more practical to compute
derivatives via the "cos(φ) · ∂f/∂φ" form which avoids 1/cos(φ)
singularities.

For the dynamical core, the key operations needed are:
1. Laplacian / inverse Laplacian (trivial in spectral space)
2. Grid-point products → spectral (nonlinear terms computed on grid)
3. Spectral → grid for physics and diagnostics

The meridional and zonal derivatives are best handled via:
- Zonal: ∂f/∂λ → im · f̂ₙᵐ  (then transform to grid for 1/cos(φ) factor)
- Meridional: cos(φ)·∂f/∂φ uses a recurrence on the Legendre polynomials
"""

from __future__ import annotations

import functools

import jax.numpy as jnp


def laplacian(
    coeffs: jnp.ndarray,
    truncation: int,
    radius: float,
) -> jnp.ndarray:
    """Apply the Laplacian operator in spectral space.

    ∇²f has coefficients  -n(n+1)/a² · f̂ₙᵐ

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)
    return coeffs * eigenvalues


def inverse_laplacian(
    coeffs: jnp.ndarray,
    truncation: int,
    radius: float,
) -> jnp.ndarray:
    """Apply the inverse Laplacian (∇⁻²) in spectral space.

    The n=0, m=0 mode (global mean) has zero eigenvalue and is left unchanged
    (set to zero), since ∇⁻²(constant) is undefined up to a constant.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)
    # Avoid division by zero for the n=0 mode (global mean).
    # The n=0 eigenvalue is exactly 0; we replace it with 1 to prevent NaN,
    # then zero out the n=0 result below.
    safe_eigenvalues = jnp.where(eigenvalues == 0, 1.0, eigenvalues)
    result = coeffs / safe_eigenvalues
    # Zero out the n=0 mode.
    return result.at[0].set(0.0)


def hyperdiffusion(
    coeffs: jnp.ndarray,
    truncation: int,
    radius: float,
    order: int = 2,
    damping_timescale: float = 0.5 * 86400.0,
) -> jnp.ndarray:
    """Compute the hyperdiffusion tendency in spectral space.

    Applies (-1)^{p+1} · ν_{2p} · ∇^{2p} f where p = order.
    The diffusion coefficient ν_{2p} is chosen so that the smallest
    resolved wavenumber (n = T) is damped with e-folding time = damping_timescale.

    For order=2 (biharmonic / del⁴), this is the standard choice for GCMs.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].
    order : int
        Order p of the hyperdiffusion (p=1 → ∇², p=2 → ∇⁴, etc.).
    damping_timescale : float
        E-folding damping time for the smallest scale [s]. Default: 0.5 days.

    Returns
    -------
    jnp.ndarray
        Tendency due to hyperdiffusion (spectral coefficients).
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)

    # Eigenvalue at truncation wavenumber (most negative)
    t_eigenvalue = -truncation * (truncation + 1) / radius**2

    # Diffusion coefficient: 1/τ = ν · |λ_T|^p  →  ν = 1/(τ · |λ_T|^p)
    nu = 1.0 / (damping_timescale * abs(t_eigenvalue) ** order)

    # Tendency: (-1)^{p+1} · ν · (∇²)^p · f
    # = (-1)^{p+1} · ν · eigenvalue^p · f
    # Since eigenvalue = -n(n+1)/a² < 0 for n>0:
    #   eigenvalue^p has sign (-1)^p
    #   (-1)^{p+1} · (-1)^p = (-1)^{2p+1} = -1
    # So the tendency is always -ν · |eigenvalue|^p · f  (damping).
    return -nu * jnp.abs(eigenvalues) ** order * coeffs


def zonal_derivative(
    coeffs: jnp.ndarray,
    truncation: int,
) -> jnp.ndarray:
    """Compute ∂f/∂λ in spectral space.

    ∂f/∂λ has coefficients  im · f̂ₙᵐ

    Note: this gives the derivative with respect to longitude.
    To get (1/a·cos φ) ∂f/∂λ, divide by (a·cos φ) in grid space.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    """
    m_values = _m_index_array(truncation)
    return 1j * m_values * coeffs


@functools.lru_cache(maxsize=16)
def _laplacian_eigenvalues(truncation: int, radius: float) -> jnp.ndarray:
    """Pre-compute -n(n+1)/a² for all spectral indices."""
    vals = [
        -n * (n + 1) / radius**2
        for m in range(truncation + 1)
        for n in range(m, truncation + 1)
    ]
    return jnp.array(vals)


@functools.lru_cache(maxsize=16)
def _m_index_array(truncation: int) -> jnp.ndarray:
    """Array of zonal wavenumber m for each spectral index."""
    vals = [m for m in range(truncation + 1) for _n in range(m, truncation + 1)]
    return jnp.array(vals, dtype=jnp.float64)
