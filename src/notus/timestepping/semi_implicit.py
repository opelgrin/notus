"""Semi-implicit treatment of gravity waves.

In the shallow water equations, gravity waves propagate at speed c = sqrt(g*H0).
Without implicit treatment, the timestep is limited by the gravity wave CFL.

The IMEX (Implicit-Explicit) leapfrog scheme (following Dinosaur / NeuralGCM)
cleanly separates the tendency into explicit nonlinear terms and implicit
linear gravity-wave terms:

    ∂x/∂t = F(x) + L(x)

where F is the nonlinear explicit tendency and L is the linear implicit
tendency:
    L_δ = -∇²Φ
    L_Φ = -Φ₀·δ

The leapfrog step becomes:
    intermediate = x_{n-1} + 2dt·(F(x_n) + (1-α)·L(x_{n-1}))
    x_{n+1} = (I - 2dt·α·L)⁻¹ · intermediate

with α = 0.5 (standard centred implicit weighting).

Primitive equations
~~~~~~~~~~~~~~~~~~~
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

from notus.operators import _laplacian_eigenvalues
from notus.sigma import SigmaLevels
from notus.state import PrimitiveEquationState
from notus.vertical import geopotential_weights, sigma_ratios


@dataclasses.dataclass(frozen=True, slots=True)
class SemiImplicitConfig:
    """Configuration for semi-implicit gravity wave treatment.

    Parameters
    ----------
    mean_geopotential : float
        Reference geopotential Phi_0 = g*H0 [m^2/s^2]. This is the mean
        depth times gravity around which gravity waves are linearized.
    alpha : float
        Implicit weighting parameter. 0.5 = centred (standard).
    """

    mean_geopotential: float
    alpha: float = 0.5


def implicit_terms(
    divergence: jnp.ndarray,
    geopotential: jnp.ndarray,
    config: SemiImplicitConfig,
    truncation: int,
    radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the linear implicit tendency L(x).

    Returns (L_δ, L_Φ) = (-∇²Φ, -Φ₀·δ).
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)
    l_div = -eigenvalues * geopotential
    l_phi = -config.mean_geopotential * divergence
    return l_div, l_phi


def implicit_inverse(
    divergence: jnp.ndarray,
    geopotential: jnp.ndarray,
    step_size: float,
    config: SemiImplicitConfig,
    truncation: int,
    radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply (I - step_size · L)⁻¹ to (divergence, geopotential).

    The 2×2 block system for each spectral mode is:

        [1               step_size·∇² ] [δ_out]   [δ_in ]
        [step_size·Φ₀    1            ] [Φ_out] = [Φ_in ]

    The Schur complement gives a diagonal solve in spectral space.

    Parameters
    ----------
    divergence : jnp.ndarray
        Intermediate spectral divergence, shape ``(n_spectral,)``.
    geopotential : jnp.ndarray
        Intermediate spectral geopotential, shape ``(n_spectral,)``.
    step_size : float
        Implicit step size, typically ``2 · dt · α`` [s] where α is the
        implicit weighting from :class:`SemiImplicitConfig`.
    config : SemiImplicitConfig
        Semi-implicit configuration (provides Φ₀).
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Solved (δ_out, Φ_out), each shape ``(n_spectral,)``.
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)  # -n(n+1)/a²
    phi0 = config.mean_geopotential

    # Schur complement: 1 - step_size² · Φ₀ · eigenvalues
    # (eigenvalues are negative, so -step_size²·Φ₀·eigenvalues > 0)
    inv_schur = 1.0 / (1.0 - step_size**2 * phi0 * eigenvalues)

    div_out = inv_schur * (divergence - step_size * eigenvalues * geopotential)
    phi_out = inv_schur * (-step_size * phi0 * divergence + geopotential)
    return div_out, phi_out


# =====================================================================
# Primitive Equations — 3D Helmholtz solver
# =====================================================================


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
    return PESemiImplicitConfig(
        coupling_matrix=jnp.array(coupling),
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
    truncation: int,
    radius: float,
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
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].

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

    eigenvalues = _laplacian_eigenvalues(truncation, radius)  # (n_spec,)

    # L_δ = -∇²(G @ T' + R·T_ref·lnps) where T' = T - T_ref
    # T_ref is spatially constant → only mode (0,0) in spectral space
    t_prime = state.temperature.at[:, 0].add(-t_ref)
    phi = geo_w @ t_prime + r_gas * t_ref[:, None] * state.log_surface_pressure[None, :]
    l_div = -eigenvalues[None, :] * phi

    # Temperature implicit tendency: -H @ divergence
    l_temp = -(temp_w @ state.divergence)

    # L_lnps = -Δσ @ δ: (L,) @ (L, n_spec) → (n_spec,)
    l_lnps = -(dsigma @ state.divergence)

    return PrimitiveEquationState(
        vorticity=jnp.zeros_like(state.vorticity),
        divergence=l_div,
        temperature=l_temp,
        log_surface_pressure=l_lnps,
    )


def pe_implicit_inverse(
    state: PrimitiveEquationState,
    step_size: float,
    config: PESemiImplicitConfig,
    truncation: int,
    radius: float,
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
    truncation : int
        Triangular truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    PrimitiveEquationState
        Solved state (δ_new, T_new, lnps_new) in spectral space.
    """
    geo_w = config.geopotential_weights  # (L, L)
    temp_w = config.temp_implicit_weights  # (L, L)
    coupling = config.coupling_matrix  # (L, L)
    dsigma = config.dsigma  # (L,)
    t_ref = config.reference_temperature  # (L,)
    r_gas = config.gas_constant
    s = step_size

    eigenvalues = _laplacian_eigenvalues(truncation, radius)  # (n_spec,)
    n_levels = state.n_levels

    # --- Step 1: geopotential intermediate ---
    # Φ* = G @ T'* + R·T_ref·lnps*   where T'* = T* - T_ref
    # T_ref is spatially constant → only mode (0,0) in spectral space
    t_prime_star = state.temperature.at[:, 0].add(-t_ref)
    phi_star = geo_w @ t_prime_star + r_gas * t_ref[:, None] * state.log_surface_pressure[None, :]

    # --- Step 2: right-hand side for δ solve ---
    # rhs = δ* - s·eigenvalues·Φ*
    # eigenvalues = -n(n+1)/a², so -s·eigenvalues·Φ* = s·|λ|·Φ* > 0
    rhs = state.divergence - s * eigenvalues[None, :] * phi_star  # (L, n_spec)

    # --- Step 3: build and solve the L×L system per spectral mode ---
    # A_nm = I_L - s²·eigenvalue_nm·M   shape (n_spec, L, L)
    eye = jnp.eye(n_levels)
    system = eye[None, :, :] - s**2 * eigenvalues[:, None, None] * coupling[None, :, :]

    # Solve system @ δ_new = rhs   (batch over spectral coefficients)
    # JAX >=0.5 requires explicit 2-D rhs for batched solve
    delta_new = jnp.linalg.solve(system, rhs.T[..., None]).squeeze(-1).T  # (L, n_spec)

    # --- Step 4: back-substitute for T and lnps ---
    t_new = state.temperature - s * (temp_w @ delta_new)
    lnps_new = state.log_surface_pressure - s * (dsigma @ delta_new)

    return PrimitiveEquationState(
        vorticity=state.vorticity,
        divergence=delta_new,
        temperature=t_new,
        log_surface_pressure=lnps_new,
    )
