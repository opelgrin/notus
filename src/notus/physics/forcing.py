"""Physics forcing protocols and implementations.

Defines the interface that all forcing functions must satisfy, plus the
Held-Suarez forcing stub for the standard benchmark.
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

from notus.state import PrimitiveEquationState
from notus.vertical.sigma import SigmaLevels


class Forcing(Protocol):
    """Protocol for physics forcing functions.

    A forcing function receives the current model state and returns a
    ``PrimitiveEquationState`` containing the tendencies due to physics
    parameterizations (e.g. radiation, boundary layer drag).

    The tendencies are added to the explicit dynamics tendencies at each
    time step.
    """

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        """Compute physics tendencies.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field pₛ (grid space), shape ``(n_lat, n_lon)``.

        Returns
        -------
        PrimitiveEquationState
            Tendencies due to physics forcing (spectral coefficients).
        """
        ...


class HeldSuarez:
    """Held-Suarez (1994) forcing for the standard GCM benchmark.

    Implements Newtonian relaxation toward a prescribed equilibrium
    temperature profile and Rayleigh drag in the boundary layer.

    References
    ----------
    - Held, I. M. & Suarez, M. J. (1994). A proposal for the intercomparison
      of the dynamical cores of atmospheric general circulation models.
      BAMS 75(10), 1825-1830.
    """

    def __init__(
        self,
        levels: SigmaLevels,
        *,
        k_a: float = 1.0 / (40.0 * 86400.0),
        k_s: float = 1.0 / (4.0 * 86400.0),
        k_f: float = 1.0 / (1.0 * 86400.0),
        delta_t_y: float = 60.0,
        delta_theta_z: float = 10.0,
        sigma_b: float = 0.7,
        t_min: float = 200.0,
        p0: float = 1.0e5,
    ) -> None:
        """Initialise Held-Suarez forcing.

        Parameters
        ----------
        levels : SigmaLevels
            Sigma vertical coordinate.
        k_a : float
            Free-atmosphere relaxation rate [s⁻¹].
        k_s : float
            Surface relaxation rate [s⁻¹].
        k_f : float
            Boundary-layer friction rate [s⁻¹].
        delta_t_y : float
            Equator-to-pole temperature difference [K].
        delta_theta_z : float
            Vertical temperature lapse-rate parameter [K].
        sigma_b : float
            Boundary-layer top (σ coordinate).
        t_min : float
            Minimum equilibrium temperature [K].
        p0 : float
            Reference pressure [Pa].
        """
        self.levels = levels
        self.k_a = k_a
        self.k_s = k_s
        self.k_f = k_f
        self.delta_t_y = delta_t_y
        self.delta_theta_z = delta_theta_z
        self.sigma_b = sigma_b
        self.t_min = t_min
        self.p0 = p0

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> PrimitiveEquationState:
        """Compute Held-Suarez tendencies.

        .. note::
            Not yet implemented.  Returns zero tendencies.
        """
        raise NotImplementedError("Held-Suarez forcing not yet implemented")
