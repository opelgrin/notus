"""Vector spectral operators on the sphere.

Wind reconstruction from vorticity/divergence, and spectral curl/divergence
of vector flux fields.  All operators accept an
:class:`~notus.operators.arrays.OperatorArrays` instance.
"""

from __future__ import annotations

import jax.numpy as jnp

from notus.operators.arrays import OperatorArrays
from notus.operators.core import inverse_laplacian, meridional_derivative, zonal_derivative


def _mu_derivative(
    coeffs: jnp.ndarray,
    mu_derivative_coupling: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Compute d/dμ in spectral space via integration-by-parts recurrence.

    [dG/dμ]ₙᵐ = −(n+1)·ε(n,m)·Ĝ_{n−1} + n·ε(n+1,m)·Ĝ_{n+1}

    Parameters
    ----------
    coeffs : jnp.ndarray
        Spectral coefficients.
    mu_derivative_coupling : tuple
        Pre-computed coupling arrays from :attr:`OperatorArrays.mu_derivative_coupling`.
    """
    idx_lower, idx_upper, c_lower, c_upper = mu_derivative_coupling
    return c_lower * coeffs[idx_lower] + c_upper * coeffs[idx_upper]


def uv_from_vordiv(
    vorticity: jnp.ndarray,
    divergence: jnp.ndarray,
    arrays: OperatorArrays,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct cosine-weighted winds from spectral vorticity and divergence.

    Computes U = u·cos(φ) and V = v·cos(φ) in spectral space via:

        ψ = ∇⁻²ζ,  χ = ∇⁻²δ
        U = (1/a)·[−cos(φ)·∂ψ/∂φ + ∂χ/∂λ]
        V = (1/a)·[ ∂ψ/∂λ + cos(φ)·∂χ/∂φ]

    Parameters
    ----------
    vorticity : jnp.ndarray
        Spectral vorticity ζ, shape ``(n_spectral,)``.
    divergence : jnp.ndarray
        Spectral divergence δ, shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (U_spectral, V_spectral), each shape ``(n_spectral,)``.
    """
    psi = inverse_laplacian(vorticity, arrays)
    chi = inverse_laplacian(divergence, arrays)

    dpsi_dlam = zonal_derivative(psi, arrays)
    dchi_dlam = zonal_derivative(chi, arrays)
    cosphi_dpsi_dphi = meridional_derivative(psi, arrays)
    cosphi_dchi_dphi = meridional_derivative(chi, arrays)

    inv_a = 1.0 / arrays.radius
    u_spec = inv_a * (-cosphi_dpsi_dphi + dchi_dlam)
    v_spec = inv_a * (dpsi_dlam + cosphi_dchi_dphi)
    return u_spec, v_spec


def spectral_divergence(
    a_hat: jnp.ndarray,
    b_hat: jnp.ndarray,
    arrays: OperatorArrays,
) -> jnp.ndarray:
    """Compute spectral divergence from flux components divided by cos²(φ).

    The spectral divergence is:

        [∇·F]ₙᵐ = (1/a)·[im·Â + D_μ(B̂)]

    Parameters
    ----------
    a_hat : jnp.ndarray
        Spectral coefficients of the zonal flux / cos²(φ), shape ``(n_spectral,)``.
    b_hat : jnp.ndarray
        Spectral coefficients of the meridional flux / cos²(φ), shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    jnp.ndarray
        Spectral divergence coefficients, shape ``(n_spectral,)``.
    """
    inv_a = 1.0 / arrays.radius
    return inv_a * (
        zonal_derivative(a_hat, arrays) + _mu_derivative(b_hat, arrays.mu_derivative_coupling)
    )


def spectral_curl(
    a_hat: jnp.ndarray,
    b_hat: jnp.ndarray,
    arrays: OperatorArrays,
) -> jnp.ndarray:
    """Compute spectral curl (vertical component) from flux components divided by cos²(φ).

    The spectral curl is:

        [curl_z(F)]ₙᵐ = (1/a)·[−im·B̂ + D_μ(Â)]

    Parameters
    ----------
    a_hat : jnp.ndarray
        Spectral coefficients of the zonal flux / cos²(φ), shape ``(n_spectral,)``.
    b_hat : jnp.ndarray
        Spectral coefficients of the meridional flux / cos²(φ), shape ``(n_spectral,)``.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    jnp.ndarray
        Spectral curl coefficients, shape ``(n_spectral,)``.
    """
    inv_a = 1.0 / arrays.radius
    return inv_a * (
        -zonal_derivative(b_hat, arrays) + _mu_derivative(a_hat, arrays.mu_derivative_coupling)
    )
