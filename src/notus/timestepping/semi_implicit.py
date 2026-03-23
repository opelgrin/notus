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
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from notus.operators import _laplacian_eigenvalues


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
