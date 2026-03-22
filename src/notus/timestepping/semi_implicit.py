"""Semi-implicit treatment of gravity waves.

In the shallow water equations, gravity waves propagate at speed c = sqrt(g*H0).
Without implicit treatment, the timestep is limited by the gravity wave CFL.
The semi-implicit scheme (Robert 1969, Hoskins & Simmons 1975) treats the
linear coupling between divergence and geopotential implicitly, resulting
in a Helmholtz equation that is diagonal in spectral space.

The correction is applied AFTER the explicit leapfrog prediction, modifying
only the divergence and geopotential fields.
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
    """

    mean_geopotential: float


def semi_implicit_correction(
    delta_star: jnp.ndarray,
    phi_star: jnp.ndarray,
    delta_prev: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float,
    config: SemiImplicitConfig,
    truncation: int,
    radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply semi-implicit correction to explicit leapfrog predictions.

    Given the explicit leapfrog predictions (delta_star, phi_star) and the
    previous state (delta_prev, phi_prev), solves the implicit gravity wave
    coupling via a Helmholtz equation diagonal in spectral space.

    The formulation (Robert, Yakimiv & Daley 1972):

        delta_new = delta_prev + D
        phi_new   = phi_star - dt * Phi_0 * D

    where D solves:

        [1 + dt^2 * Phi_0 * n(n+1)/a^2] * D = (delta_star - delta_prev)
                                                - dt * eigenvalue * (phi_star - phi_prev)

    Parameters
    ----------
    delta_star : jnp.ndarray
        Explicit leapfrog divergence prediction, shape (n_spectral,).
    phi_star : jnp.ndarray
        Explicit leapfrog geopotential prediction, shape (n_spectral,).
    delta_prev : jnp.ndarray
        Divergence at time t-dt, shape (n_spectral,).
    phi_prev : jnp.ndarray
        Geopotential at time t-dt, shape (n_spectral,).
    dt : float
        Timestep [s]. This is the actual leapfrog half-step.
    config : SemiImplicitConfig
        Reference geopotential Phi_0.
    truncation : int
        Spectral truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (delta_new, phi_new) — corrected spectral fields.
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)  # -n(n+1)/a^2
    phi0 = config.mean_geopotential

    # Helmholtz factor: 1 + dt^2 * Phi_0 * n(n+1)/a^2
    helmholtz = 1.0 + dt * dt * phi0 * (-eigenvalues)

    # RHS of the Helmholtz equation
    rhs = (delta_star - delta_prev) - dt * eigenvalues * (phi_star - phi_prev)

    # Solve for the divergence increment D = delta_new - delta_prev
    d_increment = rhs / helmholtz

    # Corrected states
    delta_new = delta_prev + d_increment
    phi_new = phi_star - dt * phi0 * d_increment

    return delta_new, phi_new
