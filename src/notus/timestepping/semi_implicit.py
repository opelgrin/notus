"""Semi-implicit treatment of gravity waves.

In the shallow water equations, gravity waves propagate at speed c = √(g·H₀).
Without implicit treatment, the timestep is limited by the gravity wave CFL.
The semi-implicit scheme (Robert 1969, Hoskins & Simmons 1975) treats the
linear coupling between divergence and geopotential implicitly, resulting
in a Helmholtz equation that is diagonal in spectral space.

The implicit correction modifies only the divergence and geopotential
tendencies; the vorticity tendency is unchanged.
"""

from __future__ import annotations

import dataclasses

from notus.operators import _laplacian_eigenvalues
from notus.state import ShallowWaterState


@dataclasses.dataclass(frozen=True, slots=True)
class SemiImplicitConfig:
    """Configuration for semi-implicit gravity wave treatment.

    Parameters
    ----------
    mean_geopotential : float
        Reference geopotential Φ₀ = g·H₀ [m²/s²]. This is the mean
        depth times gravity around which gravity waves are linearized.
    alpha : float
        Implicit weight. 0.5 = Crank-Nicolson (second-order accurate),
        1.0 = fully implicit (first-order, more damping).
    """

    mean_geopotential: float
    alpha: float = 0.5


def semi_implicit_correction(
    explicit_tendency: ShallowWaterState,
    state_prev: ShallowWaterState,
    state_curr: ShallowWaterState,
    dt: float,
    config: SemiImplicitConfig,
    truncation: int,
    radius: float,
) -> ShallowWaterState:
    """Apply semi-implicit correction to explicit tendencies.

    The fully explicit tendencies already include the linear gravity wave
    terms evaluated at time t.  The semi-implicit scheme replaces these
    with a time-centered average, resulting in a Helmholtz equation:

        [1 − α²·(2dt)²·Φ₀·∇²] δ* = RHS

    Since ∇² is diagonal in spectral space with eigenvalue −n(n+1)/a²,
    the Helmholtz solve is a simple division per spectral coefficient.

    Parameters
    ----------
    explicit_tendency : ShallowWaterState
        Fully explicit tendencies (vorticity, divergence, geopotential).
    state_prev : ShallowWaterState
        State at time t−dt (spectral).
    state_curr : ShallowWaterState
        State at time t (spectral).
    dt : float
        Timestep [s]. The leapfrog effective step is 2·dt.
    config : SemiImplicitConfig
        Semi-implicit parameters (Φ₀, α).
    truncation : int
        Spectral truncation.
    radius : float
        Planet radius [m].

    Returns
    -------
    ShallowWaterState
        Corrected tendencies. Vorticity is unchanged; divergence and
        geopotential are modified.
    """
    eigenvalues = _laplacian_eigenvalues(truncation, radius)  # −n(n+1)/a²
    phi0 = config.mean_geopotential
    a = config.alpha
    two_dt = 2.0 * dt

    # Helmholtz factor: 1 + α²·(2dt)²·Φ₀·n(n+1)/a²
    # eigenvalues are negative, so −eigenvalues = n(n+1)/a² > 0
    helmholtz = 1.0 + a * a * two_dt * two_dt * phi0 * (-eigenvalues)

    # The explicit tendencies include the linear terms evaluated at time t:
    #   R_delta includes −∇²(Φ^t)
    #   R_phi   includes −Φ₀·δ^t   (for the linearized continuity)
    #
    # The semi-implicit scheme replaces the time-t evaluation with a
    # time-centered average. The correction to the leapfrog update:
    #
    #   δ^{n+1} = δ^{n-1} + 2dt·R_δ
    #             + 2dt·α·∇²·(Φ^{n-1} - Φ^{n})        [SI correction to div]
    #
    # Then Φ^{n+1} is corrected using the modified δ^{n+1}:
    #   Φ^{n+1} = Φ^{n-1} + 2dt·R_Φ
    #             - 2dt·α·Φ₀·(δ^{n+1} - δ^{n-1})       [SI correction to phi]
    #
    # After elimination, the corrected divergence tendency is:
    #   R_δ_corrected = [R_δ + α·∇²·(Φ^{n-1} - Φ^n)
    #                    + α²·(2dt)·Φ₀·∇²·R_Φ/(-∇²)] ... (via Helmholtz)
    #
    # More cleanly: compute the raw leapfrog updates, then solve the coupled
    # implicit system.

    # Raw leapfrog updates (explicit)
    delta_star = state_prev.divergence + two_dt * explicit_tendency.divergence
    phi_star = state_prev.geopotential + two_dt * explicit_tendency.geopotential

    # Semi-implicit correction terms
    # The implicit system couples δ and Φ:
    #   δ^{n+1} = δ* − α·2dt·∇²·(Φ^{n+1} − Φ^{n-1})
    #   Φ^{n+1} = Φ* − α·2dt·Φ₀·(δ^{n+1} − δ^{n-1})
    #
    # where δ* and Φ* are the explicit leapfrog results.
    # Substituting the second into the first and solving:

    # RHS for the Helmholtz equation
    rhs = delta_star + a * two_dt * eigenvalues * (phi_star - state_prev.geopotential)

    # Solve: helmholtz · δ^{n+1} = rhs
    delta_new = rhs / helmholtz

    # Back-substitute for Φ^{n+1}
    phi_new = phi_star - a * two_dt * phi0 * (delta_new - state_prev.divergence)

    # Convert back to tendencies: tendency = (x^{n+1} - x^{n-1}) / (2·dt)
    div_tend_corrected = (delta_new - state_prev.divergence) / two_dt
    phi_tend_corrected = (phi_new - state_prev.geopotential) / two_dt

    return ShallowWaterState(
        vorticity=explicit_tendency.vorticity,
        divergence=div_tend_corrected,
        geopotential=phi_tend_corrected,
    )
