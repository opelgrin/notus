"""Scalar spectral operators on the sphere.

Laplacian family, scalar derivatives, and spectral filters.  All operators
work on spectral coefficients and return spectral coefficients.

Key identities for a scalar f with spectral coefficients f̂ₙᵐ:

- **Laplacian:** ∇²f has coefficients  -n(n+1)/a² · f̂ₙᵐ
- **Inverse Laplacian:** ∇⁻²f has coefficients  -a²/[n(n+1)] · f̂ₙᵐ  (n>0)
"""

from __future__ import annotations

import jax.numpy as jnp

from notus.operators.caches import (
    _laplacian_eigenvalues,
    _m_index_array,
    _meridional_coupling,
    _n_index_array,
)


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

    Returns
    -------
    jnp.ndarray
        Spectral coefficients of ∇²f, same shape as *coeffs*.
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

    Returns
    -------
    jnp.ndarray
        Spectral coefficients of ∂f/∂λ, same shape as *coeffs*.
    """
    m_values = _m_index_array(truncation)
    return 1j * m_values * coeffs


def meridional_derivative(
    coeffs: jnp.ndarray,
    truncation: int,
) -> jnp.ndarray:
    """Compute cos(φ)·∂f/∂φ in spectral space via Legendre recurrence.

    Uses the identity:

        [cos(φ)·∂f/∂φ]ₙᵐ = (n+1)·ε(n,m)·f̂_{n-1}^m − n·ε(n+1,m)·f̂_{n+1}^m

    where ε(n,m) = √((n²−m²)/(4n²−1)).

    The result includes the cos(φ) factor, which avoids polar singularities.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.

    Returns
    -------
    jnp.ndarray
        Spectral coefficients of cos(φ)·∂f/∂φ, same shape as *coeffs*.
    """
    idx_lower, idx_upper, c_lower, c_upper = _meridional_coupling(truncation)
    return c_lower * coeffs[idx_lower] + c_upper * coeffs[idx_upper]


def exponential_filter(
    truncation: int,
    dt: float,
    rotation_rate: float = 7.292e-5,
    tau: float = 0.010938,
    order: int = 18,
) -> jnp.ndarray:
    """Build a multiplicative spectral filter array (Hou & Li 2007).

    Applied once per timestep, this damps high-wavenumber modes as:

        scaling(n) = exp(-attenuation · (n / T) ^ (2·order))

    where attenuation = 2·Ω·dt / tau (the nondimensional timestep divided
    by the filter timescale).

    Default parameters match Dinosaur / NeuralGCM: ``tau = 0.010938``,
    ``order = 18``, giving a k^36 rolloff that removes > 99 % of the
    energy at the truncation wavenumber every step while leaving n ≤ 0.8·T
    essentially untouched.

    Parameters
    ----------
    truncation : int
        Triangular truncation T.
    dt : float
        Timestep [s].
    rotation_rate : float
        Planetary angular velocity Ω [rad/s]. Default: Earth.
    tau : float
        Nondimensional filter timescale.
    order : int
        Polynomial order of the filter exponent.

    Returns
    -------
    jnp.ndarray
        Multiplicative scaling array, shape ``(n_spectral,)``.
        Multiply spectral coefficients by this every timestep.
    """
    n_vals = _n_index_array(truncation)
    # Normalize by T+1 (total_wavenumbers), matching Dinosaur / NeuralGCM.
    # This ensures the filter at n=T retains ~20% instead of ~0.03%.
    k = n_vals / (truncation + 1)
    attenuation = dt * 2.0 * rotation_rate / tau
    return jnp.exp(-attenuation * k ** (2 * order))
