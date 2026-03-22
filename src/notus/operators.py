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
    """
    idx_lower, idx_upper, c_lower, c_upper = _meridional_coupling(truncation)
    return c_lower * coeffs[idx_lower] + c_upper * coeffs[idx_upper]


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
    """Compute divergence in spectral space from scaled flux components.

    Given spectral coefficients of A = F_λ/cos(φ) and B = F_φ/cos(φ),
    computes the spectral divergence of the vector field (F_λ, F_φ):

        div(F) = (1/a)·[im·Â + cos(φ)·∂B̂/∂φ]

    This form is used because the nonlinear products in the dynamical core
    naturally produce the 1/cos(φ)-scaled quantities after dividing by cos²(φ)
    on the grid.

    Parameters
    ----------
    a_hat : jnp.ndarray
        Spectral coefficients of zonal flux / cos(φ), shape ``(n_spectral,)``.
    b_hat : jnp.ndarray
        Spectral coefficients of meridional flux / cos(φ), shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].
    """
    inv_a = 1.0 / radius
    return inv_a * (
        zonal_derivative(a_hat, truncation) + meridional_derivative(b_hat, truncation)
    )


def spectral_curl(
    a_hat: jnp.ndarray,
    b_hat: jnp.ndarray,
    truncation: int,
    radius: float,
) -> jnp.ndarray:
    """Compute vertical component of curl in spectral space from scaled flux components.

    Given spectral coefficients of A = F_λ/cos(φ) and B = F_φ/cos(φ),
    computes the vertical component of the curl:

        curl_z(F) = (1/a)·[−im·B̂ + cos(φ)·∂Â/∂φ]

    Parameters
    ----------
    a_hat : jnp.ndarray
        Spectral coefficients of zonal flux / cos(φ), shape ``(n_spectral,)``.
    b_hat : jnp.ndarray
        Spectral coefficients of meridional flux / cos(φ), shape ``(n_spectral,)``.
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].
    """
    inv_a = 1.0 / radius
    return inv_a * (
        -zonal_derivative(b_hat, truncation) + meridional_derivative(a_hat, truncation)
    )


# ---------------------------------------------------------------------------
# Private helpers (cached)
# ---------------------------------------------------------------------------


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


@functools.lru_cache(maxsize=16)
def _n_index_array(truncation: int) -> jnp.ndarray:
    """Array of total wavenumber n for each spectral index."""
    vals = [n for m in range(truncation + 1) for n in range(m, truncation + 1)]
    return jnp.array(vals, dtype=jnp.float64)


def _spectral_index(truncation: int, m: int, n: int) -> int:
    """Linear index into lower-triangular spectral storage."""
    return m * (truncation + 1) - m * (m - 1) // 2 + (n - m)


@functools.lru_cache(maxsize=16)
def _meridional_coupling(
    truncation: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pre-compute index arrays and coefficients for the meridional derivative.

    The recurrence:
        [cos(φ)·∂f/∂φ]ₙᵐ = (n+1)·ε(n,m)·f_{n-1}^m − n·ε(n+1,m)·f_{n+1}^m

    where ε(n,m) = √((n²−m²)/(4n²−1)).

    Returns
    -------
    tuple of (idx_lower, idx_upper, coeff_lower, coeff_upper)
        idx_lower[i]  : flat index of (m, n-1), self-referencing if n == m
        idx_upper[i]  : flat index of (m, n+1), self-referencing if n == T
        coeff_lower[i]: (n+1)·ε(n,m), zero if n == m
        coeff_upper[i]: −n·ε(n+1,m), zero if n == T
    """
    t = truncation

    idx_lower = []
    idx_upper = []
    c_lower = []
    c_upper = []

    for m in range(t + 1):
        for n in range(m, t + 1):
            i = _spectral_index(t, m, n)

            # Lower coupling: f_{n-1}^m with coefficient (n+1)*eps(n,m)
            if n > m:
                eps_n = _epsilon(n, m)
                idx_lower.append(_spectral_index(t, m, n - 1))
                c_lower.append((n + 1) * eps_n)
            else:
                # n == m: no n-1 neighbor, contribute zero
                idx_lower.append(i)
                c_lower.append(0.0)

            # Upper coupling: f_{n+1}^m with coefficient -n*eps(n+1,m)
            if n < t:
                eps_np1 = _epsilon(n + 1, m)
                idx_upper.append(_spectral_index(t, m, n + 1))
                c_upper.append(-n * eps_np1)
            else:
                # n == T: no n+1 neighbor, contribute zero
                idx_upper.append(i)
                c_upper.append(0.0)

    return (
        jnp.array(idx_lower, dtype=jnp.int32),
        jnp.array(idx_upper, dtype=jnp.int32),
        jnp.array(c_lower),
        jnp.array(c_upper),
    )


def _epsilon(n: int, m: int) -> float:
    """Coupling coefficient ε(n,m) = √((n²−m²)/(4n²−1))."""
    if n == 0:
        return 0.0
    return ((n * n - m * m) / (4.0 * n * n - 1.0)) ** 0.5
