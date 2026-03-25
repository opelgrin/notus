"""Semi-implicit treatment of gravity waves in the primitive equations.

For the 3D primitive equations, the implicit terms couple divergence (δ),
temperature (T), and ln(pₛ) through the vertical structure:

    L_δ  = -∇²(G·T + R·T_ref·ln pₛ)     (geopotential + pressure gradient)
    L_T  = -H·δ                            (temperature response)
    L_lnps = -Δσᵀ·δ                        (continuity)

Substituting into the leapfrog scheme gives an L×L system per spectral mode:

    (I - s²·λₙ·M) · δ_new = δ* - s·λₙ·(G·T* + R·T_ref·lnps*)

where M = G·H + R·T_ref⊗Δσ is the vertical coupling matrix, G is the
geopotential weight matrix, H is the temperature implicit weights (encoding
the D-dependent part of adiabatic heating κ·T_ref·(ω/p)), and
λₙ = -n(n+1)/a² is the Laplacian eigenvalue.  After solving for δ_new,
T and ln(pₛ) are recovered by back-substitution.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from notus.operators.arrays import OperatorArrays
from notus.state import PrimitiveEquationState
from notus.vertical.operators import geopotential_weights, sigma_ratios
from notus.vertical.sigma import SigmaLevels


def temperature_implicit_weights(
    levels: SigmaLevels,
    kappa: float,
    reference_temperature: np.ndarray,
) -> np.ndarray:
    """Build the temperature implicit weights matrix H.

    H encodes how divergence affects temperature through two mechanisms:

    1. The D-dependent part of adiabatic heating ``κ·T_ref·(ω/p)_D``
    2. The D-dependent part of vertical advection ``-σ̇_D · ∂T_ref/∂σ``

    The implicit temperature tendency is ``L_T = -H · δ``.

    Following Dinosaur / Durran §8.6.5, the matrix entry H[r,s] is::

        H[r,s] / Δσ[s] = κ·T_ref[r]·(P(r-s)·α[r] + P(r-1-s)·α[r-1]) / Δσ[r]
                        - K[r,s] - K[r-1,s]

    where P is the step function (1 for non-negative argument, 0 otherwise),
    α are the sigma ratios, and K captures vertical advection of T_ref::

        K[r,s] = (T_ref[r+1] - T_ref[r]) / (Δσ[r+1] + Δσ[r])
                 · (P(r-s) - Σ Δσ[:r+1])

    with K[r,s] = 0 for r = L-1 (bottom level) or r < 0.

    The K terms vanish when T_ref is vertically uniform.

    Parameters
    ----------
    levels : SigmaLevels
        Sigma vertical coordinate.
    kappa : float
        Ratio R/cₚ.
    reference_temperature : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.

    Returns
    -------
    np.ndarray
        H matrix, shape ``(n_levels, n_levels)``.
    """
    alpha = sigma_ratios(levels)
    dsigma = np.asarray(levels.dsigma)
    t_ref = np.asarray(reference_temperature)
    n = levels.n_levels

    # Lower triangular indicator: p[r,s] = 1 if r >= s (step function)
    p = np.tril(np.ones((n, n)))

    # Term 1: adiabatic heating contribution (h0)
    # h0[r,s] = κ·T_ref[r]·(P(r-s)·α[r] + P(r-1-s)·α[r-1]) / Δσ[r]
    alpha_col = alpha[:, np.newaxis]  # (L, 1) broadcasts as α[r]
    p_alpha = p * alpha_col
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = kappa * t_ref[:, np.newaxis] * (p_alpha + p_alpha_shifted) / dsigma[:, np.newaxis]

    # Term 2: vertical advection of T_ref by D-dependent σ̇ (K terms)
    # K[r,s] = (T_ref[r+1]-T_ref[r])/(Δσ[r+1]+Δσ[r]) · (P(r-s) - cumsum(Δσ[:r+1]))
    # K[L-1, :] = 0 (bottom level, no level below)
    temp_diff = np.diff(t_ref)  # (L-1,)
    thickness_sum = dsigma[:-1] + dsigma[1:]  # (L-1,)
    k0 = np.concatenate([temp_diff / thickness_sum, [0.0]])[:, np.newaxis]  # (L, 1)
    thickness_cumulative = np.cumsum(dsigma)[:, np.newaxis]  # (L, 1)
    k1 = p - thickness_cumulative  # (L, L)
    k = k0 * k1
    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0

    return (h0 - k - k_shifted) * dsigma  # multiply by Δσ[s] (column-wise)


def pe_coupling_matrix(
    levels: SigmaLevels,
    gas_constant: float,
    kappa: float,
    reference_temperature: np.ndarray,
) -> np.ndarray:
    """Build the vertical coupling matrix M = G·H + R·T_ref⊗Δσ.

    M appears in the Helmholtz problem for the semi-implicit divergence
    solve:

        (I - s²·λₙ·M) · δ_new = rhs

    where λₙ = -n(n+1)/a² and s is the implicit step size.

    Parameters
    ----------
    levels : SigmaLevels
        Sigma vertical coordinate.
    gas_constant : float
        Specific gas constant R_d [J/(kg·K)].
    kappa : float
        Ratio R/cₚ.
    reference_temperature : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.

    Returns
    -------
    np.ndarray
        Coupling matrix, shape ``(n_levels, n_levels)``.
    """
    geo_w = geopotential_weights(levels, gas_constant)
    temp_w = temperature_implicit_weights(levels, kappa, reference_temperature)
    dsigma = np.asarray(levels.dsigma)
    t_ref = np.asarray(reference_temperature)
    return geo_w @ temp_w + gas_constant * np.outer(t_ref, dsigma)


@dataclasses.dataclass(frozen=True, slots=True)
class PESemiImplicitConfig:
    """Pre-computed configuration for PE semi-implicit treatment.

    All matrices are computed once at setup and reused every time step.

    Attributes
    ----------
    coupling_matrix : jnp.ndarray
        M = G·H + R·T_ref⊗Δσ, shape ``(n_levels, n_levels)``.
    coupling_eigvals : jnp.ndarray
        Eigenvalues of M, shape ``(n_levels,)``.
    coupling_p : jnp.ndarray
        Eigenvector matrix P of M, shape ``(n_levels, n_levels)``.
        M = P @ diag(eigvals) @ P_inv.
    coupling_p_inv : jnp.ndarray
        Inverse eigenvector matrix P⁻¹, shape ``(n_levels, n_levels)``.
    geopotential_weights : jnp.ndarray
        G matrix from hydrostatic integration, shape ``(n_levels, n_levels)``.
    temp_implicit_weights : jnp.ndarray
        H matrix for temperature-divergence coupling, ``(n_levels, n_levels)``.
    dsigma : jnp.ndarray
        Layer thicknesses Δσ, shape ``(n_levels,)``.
    reference_temperature : jnp.ndarray
        T_ref profile, shape ``(n_levels,)``.
    gas_constant : float
        Specific gas constant R_d [J/(kg·K)].
    alpha : float
        Implicit weighting (0.5 = centred).
    """

    coupling_matrix: jnp.ndarray
    coupling_eigvals: jnp.ndarray
    coupling_p: jnp.ndarray
    coupling_p_inv: jnp.ndarray
    geopotential_weights: jnp.ndarray
    temp_implicit_weights: jnp.ndarray
    dsigma: jnp.ndarray
    reference_temperature: jnp.ndarray
    gas_constant: float
    alpha: float = 0.5


def build_pe_semi_implicit_config(
    levels: SigmaLevels,
    gas_constant: float,
    kappa: float,
    reference_temperature: np.ndarray,
    alpha: float = 0.5,
) -> PESemiImplicitConfig:
    """Build a :class:`PESemiImplicitConfig` from physical parameters.

    Parameters
    ----------
    levels : SigmaLevels
        Sigma vertical coordinate.
    gas_constant : float
        Specific gas constant R_d [J/(kg·K)].
    kappa : float
        Ratio R/cₚ.
    reference_temperature : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.
    alpha : float
        Implicit weighting (0.5 = centred).

    Returns
    -------
    PESemiImplicitConfig
    """
    t_ref = np.asarray(reference_temperature)
    geo_w = geopotential_weights(levels, gas_constant)
    temp_w = temperature_implicit_weights(levels, kappa, t_ref)
    coupling = pe_coupling_matrix(levels, gas_constant, kappa, t_ref)

    # Eigen-decompose M once so the implicit solve becomes O(L²) per mode
    # instead of O(L³) from jnp.linalg.solve.
    # Note: M is not symmetric, but its eigenvalues are empirically real for
    # physical parameter ranges.  If complex eigenvalues ever appear the
    # diagonal solve in pe_implicit_inverse still works (complex arithmetic),
    # but the result should be checked for spurious imaginary parts.
    eigvals_m, p_mat = np.linalg.eig(coupling)
    p_inv = np.linalg.inv(p_mat)

    return PESemiImplicitConfig(
        coupling_matrix=jnp.array(coupling),
        coupling_eigvals=jnp.array(eigvals_m),
        coupling_p=jnp.array(p_mat),
        coupling_p_inv=jnp.array(p_inv),
        geopotential_weights=jnp.array(geo_w),
        temp_implicit_weights=jnp.array(temp_w),
        dsigma=levels.dsigma,
        reference_temperature=jnp.array(t_ref),
        gas_constant=gas_constant,
        alpha=alpha,
    )


def pe_implicit_terms(
    state: PrimitiveEquationState,
    config: PESemiImplicitConfig,
    arrays: OperatorArrays,
) -> PrimitiveEquationState:
    """Evaluate the linear implicit tendency L(x) for the primitive equations.

    Returns a ``PrimitiveEquationState`` whose fields are the implicit
    tendencies:

        L_ζ     = 0                                (vorticity fully explicit)
        L_δ     = -∇²(G·T + R·T_ref·ln pₛ)
        L_T     = -H·δ
        L_lnps  = -Δσᵀ·δ

    Parameters
    ----------
    state : PrimitiveEquationState
        Current state (spectral).
    config : PESemiImplicitConfig
        Pre-computed semi-implicit matrices.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    PrimitiveEquationState
        Implicit tendencies (same shapes as input state).
    """
    geo_w = config.geopotential_weights
    temp_w = config.temp_implicit_weights
    dsigma = config.dsigma
    t_ref = config.reference_temperature
    r_gas = config.gas_constant

    eigenvalues = arrays.laplacian_eigenvalues  # (n_spec,)

    # L_δ = -∇²(G @ T' + R·T_ref·lnps) where T' = T - T_ref
    # T_ref is spatially constant → only mode (0,0) in spectral space
    t_prime = state.temperature.at[:, 0].add(-t_ref)
    phi = geo_w @ t_prime + r_gas * t_ref[:, None] * state.log_surface_pressure[None, :]
    l_div = -eigenvalues[None, :] * phi

    # Temperature implicit tendency: -H @ divergence
    l_temp = -(temp_w @ state.divergence)

    # L_lnps = -Δσ @ δ: (L,) @ (L, n_spec) → (n_spec,)
    l_lnps = -(dsigma @ state.divergence)

    # Humidity is fully explicit — zero implicit tendency
    humidity_tend: jnp.ndarray | None = None
    if state.humidity is not None:
        humidity_tend = jnp.zeros_like(state.humidity)

    return PrimitiveEquationState(
        vorticity=jnp.zeros_like(state.vorticity),
        divergence=l_div,
        temperature=l_temp,
        log_surface_pressure=l_lnps,
        humidity=humidity_tend,
    )


def pe_implicit_inverse(
    state: PrimitiveEquationState,
    step_size: float,
    config: PESemiImplicitConfig,
    arrays: OperatorArrays,
) -> PrimitiveEquationState:
    """Apply (I - step_size · L)⁻¹ to the PE state.

    Solves the coupled (δ, T, ln pₛ) system via Schur complement reduction
    to a per-spectral-mode L×L linear system:

    1. Form the right-hand side:
       rhs = δ* - s·λₙ·(G·T* + R·T_ref·lnps*)

    2. Solve the Helmholtz problem per spectral mode:
       (I - s²·λₙ·M) · δ_new = rhs

    3. Back-substitute:
       T_new = T* - s·H·δ_new
       lnps_new = lnps* - s·Δσᵀ·δ_new

    Vorticity passes through unchanged (fully explicit).

    Parameters
    ----------
    state : PrimitiveEquationState
        Intermediate state (δ*, T*, lnps*) in spectral space.
    step_size : float
        Implicit step size, typically ``2·dt·α`` [s].
    config : PESemiImplicitConfig
        Pre-computed semi-implicit matrices.
    arrays : OperatorArrays
        Pre-computed operator arrays.

    Returns
    -------
    PrimitiveEquationState
        Solved state (δ_new, T_new, lnps_new) in spectral space.
    """
    geo_w = config.geopotential_weights  # (L, L)
    temp_w = config.temp_implicit_weights  # (L, L)
    dsigma = config.dsigma  # (L,)
    t_ref = config.reference_temperature  # (L,)
    r_gas = config.gas_constant
    s = step_size

    # Pre-computed eigendecomposition: M = P @ diag(μ) @ P_inv
    p_mat = config.coupling_p  # (L, L)
    p_inv = config.coupling_p_inv  # (L, L)
    mu = config.coupling_eigvals  # (L,)

    eigenvalues = arrays.laplacian_eigenvalues  # (n_spec,)

    # --- Step 1: geopotential intermediate ---
    # Φ* = G @ T'* + R·T_ref·lnps*   where T'* = T* - T_ref
    # T_ref is spatially constant → only mode (0,0) in spectral space
    t_prime_star = state.temperature.at[:, 0].add(-t_ref)
    phi_star = geo_w @ t_prime_star + r_gas * t_ref[:, None] * state.log_surface_pressure[None, :]

    # --- Step 2: right-hand side for δ solve ---
    # rhs = δ* - s·eigenvalues·Φ*
    # eigenvalues = -n(n+1)/a², so -s·eigenvalues·Φ* = s·|λ|·Φ* > 0
    rhs = state.divergence - s * eigenvalues[None, :] * phi_star  # (L, n_spec)

    # --- Step 3: solve via eigendecomposition (O(L²) per mode) ---
    # (I - s²·λₙ·M)·δ = rhs
    # In eigenspace: (1 - s²·λₙ·μⱼ)·(P⁻¹·δ)ⱼ = (P⁻¹·rhs)ⱼ
    rhs_eigen = p_inv @ rhs  # (L, n_spec)
    # Diagonal solve: d[j,n] = 1 / (1 - s²·λₙ·μⱼ)
    diag_inv = 1.0 / (1.0 - s**2 * eigenvalues[None, :] * mu[:, None])  # (L, n_spec)
    delta_new = p_mat @ (diag_inv * rhs_eigen)  # (L, n_spec)

    # --- Step 4: back-substitute for T and lnps ---
    t_new = state.temperature - s * (temp_w @ delta_new)
    lnps_new = state.log_surface_pressure - s * (dsigma @ delta_new)

    return PrimitiveEquationState(
        vorticity=state.vorticity,
        divergence=delta_new,
        temperature=t_new,
        log_surface_pressure=lnps_new,
        humidity=state.humidity,
    )
