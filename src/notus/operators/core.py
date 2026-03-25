"""Scalar spectral operators on the sphere.

Laplacian family, scalar derivatives, and spectral filters.  All operators
accept an :class:`~notus.operators.arrays.OperatorArrays` instance instead
of raw ``(truncation, radius)`` pairs, so pre-computed arrays are read
directly from instance attributes with no global cache lookups.
"""

from __future__ import annotations

import jax.numpy as jnp

from notus.operators.arrays import OperatorArrays


def laplacian(
    coeffs: jnp.ndarray,
    arrays: OperatorArrays,
) -> jnp.ndarray:
    """Apply the Laplacian operator in spectral space.

    ∇²f has coefficients  -n(n+1)/a² · f̂ₙᵐ

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays for the grid/planet.

    Returns
    -------
    jnp.ndarray
        Spectral coefficients of ∇²f, same shape as *coeffs*.
    """
    return coeffs * arrays.laplacian_eigenvalues


def inverse_laplacian(
    coeffs: jnp.ndarray,
    arrays: OperatorArrays,
) -> jnp.ndarray:
    """Apply the inverse Laplacian (∇⁻²) in spectral space.

    The n=0, m=0 mode (global mean) has zero eigenvalue and is set to zero,
    since ∇⁻²(constant) is undefined up to a constant.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.
    """
    return coeffs * arrays.inverse_laplacian_eigenvalues


def hyperdiffusion_scaling(
    arrays: OperatorArrays,
    order: int = 2,
    damping_timescale: float = 0.5 * 86400.0,
) -> jnp.ndarray:
    """Pre-compute the hyperdiffusion scaling array.

    Returns the multiplicative factor ``-ν · |λₙ|^p`` so that the
    hyperdiffusion tendency is simply ``scaling * coeffs``.

    Parameters
    ----------
    arrays : OperatorArrays
        Pre-computed operator arrays.
    order : int
        Order p (p=1 → ∇², p=2 → ∇⁴, etc.).
    damping_timescale : float
        E-folding damping time for the smallest scale [s].

    Returns
    -------
    jnp.ndarray
        Scaling array, shape ``(n_spectral,)``.
    """
    eigenvalues = arrays.laplacian_eigenvalues
    t_eigenvalue = -arrays.truncation * (arrays.truncation + 1) / arrays.radius**2
    nu = 1.0 / (damping_timescale * abs(t_eigenvalue) ** order)
    return -nu * jnp.abs(eigenvalues) ** order


def hyperdiffusion(
    coeffs: jnp.ndarray,
    arrays: OperatorArrays,
    order: int = 2,
    damping_timescale: float = 0.5 * 86400.0,
) -> jnp.ndarray:
    """Compute the hyperdiffusion tendency in spectral space.

    Applies (-1)^{p+1} · ν_{2p} · ∇^{2p} f where p = order.
    The diffusion coefficient ν_{2p} is chosen so that the smallest
    resolved wavenumber (n = T) is damped with e-folding time
    = damping_timescale.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.
    order : int
        Order p (p=1 → ∇², p=2 → ∇⁴, etc.).
    damping_timescale : float
        E-folding damping time for the smallest scale [s].

    Returns
    -------
    jnp.ndarray
        Tendency due to hyperdiffusion.
    """
    return hyperdiffusion_scaling(arrays, order, damping_timescale) * coeffs


def zonal_derivative(
    coeffs: jnp.ndarray,
    arrays: OperatorArrays,
) -> jnp.ndarray:
    """Compute ∂f/∂λ in spectral space.

    ∂f/∂λ has coefficients  im · f̂ₙᵐ

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    jnp.ndarray
        Spectral coefficients of ∂f/∂λ.
    """
    return 1j * arrays.m_index * coeffs


def meridional_derivative(
    coeffs: jnp.ndarray,
    arrays: OperatorArrays,
) -> jnp.ndarray:
    """Compute cos(φ)·∂f/∂φ in spectral space via Legendre recurrence.

    Uses the identity:

        [cos(φ)·∂f/∂φ]ₙᵐ = (n+1)·ε(n,m)·f̂_{n-1}^m − n·ε(n+1,m)·f̂_{n+1}^m

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    jnp.ndarray
        Spectral coefficients of cos(φ)·∂f/∂φ.
    """
    idx_lower, idx_upper, c_lower, c_upper = arrays.meridional_coupling
    return c_lower * coeffs[idx_lower] + c_upper * coeffs[idx_upper]


def exponential_filter(
    arrays: OperatorArrays,
    dt: float,
    rotation_rate: float = 7.292e-5,
    tau: float = 0.010938,
    order: int = 18,
) -> jnp.ndarray:
    """Build a multiplicative spectral filter array (Hou & Li 2007).

    Applied once per timestep, this damps high-wavenumber modes as:

        scaling(n) = exp(-attenuation · (n / T) ^ (2·order))

    Parameters
    ----------
    arrays : OperatorArrays
        Pre-computed operator arrays (provides n_index and truncation).
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
    """
    k = arrays.n_index / (arrays.truncation + 1)
    attenuation = dt * 2.0 * rotation_rate / tau
    return jnp.exp(-attenuation * k ** (2 * order))
