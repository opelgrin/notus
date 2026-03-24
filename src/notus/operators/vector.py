"""Vector spectral operators on the sphere.

Wind reconstruction from vorticity/divergence, and spectral curl/divergence
of vector flux fields.

Working with cosine-weighted winds U = u·cos(φ), V = v·cos(φ) avoids
1/cos(φ) singularities at the poles.
"""

from __future__ import annotations

import jax.numpy as jnp

from notus.operators.caches import _mu_derivative
from notus.operators.core import inverse_laplacian, meridional_derivative, zonal_derivative


def uv_from_vordiv(
    vorticity: jnp.ndarray,
    divergence: jnp.ndarray,
    truncation: int,
    radius: float,
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
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (U_spectral, V_spectral), each shape ``(n_spectral,)``.
    """
    psi = inverse_laplacian(vorticity, truncation, radius)
    chi = inverse_laplacian(divergence, truncation, radius)

    dpsi_dlam = zonal_derivative(psi, truncation)
    dchi_dlam = zonal_derivative(chi, truncation)
    cosphi_dpsi_dphi = meridional_derivative(psi, truncation)
    cosphi_dchi_dphi = meridional_derivative(chi, truncation)

    inv_a = 1.0 / radius
    u_spec = inv_a * (-cosphi_dpsi_dphi + dchi_dlam)
    v_spec = inv_a * (dpsi_dlam + cosphi_dchi_dphi)
    return u_spec, v_spec


def spectral_divergence(
    a_hat: jnp.ndarray,
    b_hat: jnp.ndarray,
    truncation: int,
    radius: float,
) -> jnp.ndarray:
    """Compute spectral divergence from flux components divided by cos²(φ).

    The caller provides spectral transforms of grid-point fields
    ``A = F_λ / cos²(φ)`` and ``B = F_φ / cos²(φ)`` where F_λ and F_φ are
    the cosine-weighted (u·cos φ, v·cos φ) flux components already
    multiplied by the quantity being advected.

    The spectral divergence is then:

        [∇·F]ₙᵐ = (1/a)·[im·Â + D_μ(B̂)]

    where D_μ is the μ-derivative recurrence:
        D_μ(f̂)ₙ = −(n+1)·ε(n,m)·f̂_{n−1} + n·ε(n+1,m)·f̂_{n+1}

    Parameters
    ----------
    a_hat : jnp.ndarray
        Spectral coefficients of the zonal flux divided by cos²(φ),
        shape ``(n_spectral,)``.
    b_hat : jnp.ndarray
        Spectral coefficients of the meridional flux divided by cos²(φ),
        shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    jnp.ndarray
        Spectral divergence coefficients, shape ``(n_spectral,)``.
    """
    inv_a = 1.0 / radius
    return inv_a * (zonal_derivative(a_hat, truncation) + _mu_derivative(b_hat, truncation))


def spectral_curl(
    a_hat: jnp.ndarray,
    b_hat: jnp.ndarray,
    truncation: int,
    radius: float,
) -> jnp.ndarray:
    """Compute spectral curl (vertical component) from flux components divided by cos²(φ).

    Same input convention as :func:`spectral_divergence`: the caller
    provides spectral transforms of ``A = F_λ / cos²(φ)`` and
    ``B = F_φ / cos²(φ)``.

    The spectral curl is:

        [curl_z(F)]ₙᵐ = (1/a)·[−im·B̂ + D_μ(Â)]

    Parameters
    ----------
    a_hat : jnp.ndarray
        Spectral coefficients of the zonal flux divided by cos²(φ),
        shape ``(n_spectral,)``.
    b_hat : jnp.ndarray
        Spectral coefficients of the meridional flux divided by cos²(φ),
        shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    jnp.ndarray
        Spectral curl coefficients, shape ``(n_spectral,)``.
    """
    inv_a = 1.0 / radius
    return inv_a * (-zonal_derivative(b_hat, truncation) + _mu_derivative(a_hat, truncation))
