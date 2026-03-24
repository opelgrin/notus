"""IMEX (Implicit-Explicit) leapfrog time integration.

The semi-implicit leapfrog scheme splits the tendency into:

    dx/dt = F(x) + L(x)

where F is the nonlinear explicit tendency and L is the linear implicit
tendency (gravity wave coupling).

Initialization (backward-forward Euler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intermediate = x_0 + dt · F(x_0)
    x_1 = (I - dt · L)⁻¹ · intermediate

Main step (IMEX leapfrog with Robert-Asselin filter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intermediate = x_{n-1} + 2·dt · (F(x_n) + (1-α) · L(x_{n-1}))
    x_{n+1} = (I - 2·dt·α · L)⁻¹ · intermediate
    x_n^filtered = (1 - 2r) · x_n + r · (x_{n-1} + x_{n+1})

The generic functions (``euler_init``, ``imex_leapfrog_step``) work with
any JAX pytree state via callables.  The ``build_pe_stepper`` factory
wires these up with the PE-specific explicit tendencies and semi-implicit
solver.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.dynamics.primitive_equations import primitive_equation_tendencies
from notus.physics.forcing import Forcing
from notus.state import PrimitiveEquationState
from notus.timestepping.semi_implicit_pe import (
    build_pe_semi_implicit_config,
    pe_implicit_inverse,
    pe_implicit_terms,
)
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


S = TypeVar("S")


# =====================================================================
# Generic IMEX functions (work with any JAX pytree state)
# =====================================================================


def euler_init(
    state: S,
    explicit_fn: Callable[[S], S],
    inverse_fn: Callable[[S, float], S],
    dt: float,
) -> tuple[S, S]:
    """Backward-forward Euler initialization for IMEX leapfrog.

    Computes one first-order step to bootstrap the two-level scheme:

        intermediate = x_0 + dt · F(x_0)
        x_1 = (I - dt · L)⁻¹ · intermediate

    Parameters
    ----------
    state : S
        Initial state x_0 (any JAX pytree).
    explicit_fn : Callable[[S], S]
        Computes explicit tendencies F(x).
    inverse_fn : Callable[[S, float], S]
        Applies ``(I - step_size · L)⁻¹``.
    dt : float
        Timestep [s].

    Returns
    -------
    tuple[S, S]
        ``(previous, current)`` = ``(x_0, x_1)`` ready for leapfrog.
    """
    tendency = explicit_fn(state)
    intermediate = jax.tree.map(lambda x, f: x + dt * f, state, tendency)
    current = inverse_fn(intermediate, dt)
    return state, current


def imex_leapfrog_step(
    previous: S,
    current: S,
    explicit_fn: Callable[[S], S],
    implicit_fn: Callable[[S], S],
    inverse_fn: Callable[[S, float], S],
    dt: float,
    alpha: float = 0.5,
    robert_coeff: float = 0.05,
) -> tuple[S, S]:
    """One IMEX leapfrog step with Robert-Asselin filter.

    Evaluates explicit tendencies at the current time level and implicit
    terms at the previous time level, then solves the implicit system
    and applies the Robert-Asselin filter.

    Parameters
    ----------
    previous : S
        State at t - dt (x_{n-1}).
    current : S
        State at t (x_n).
    explicit_fn : Callable[[S], S]
        Computes explicit tendencies F(x).
    implicit_fn : Callable[[S], S]
        Computes implicit tendencies L(x).
    inverse_fn : Callable[[S, float], S]
        Applies ``(I - step_size · L)⁻¹``.
    dt : float
        Timestep [s].
    alpha : float
        Implicit weighting (0.5 = centred, standard).
    robert_coeff : float
        Robert-Asselin filter coefficient r (typically 0.05).

    Returns
    -------
    tuple[S, S]
        ``(filtered_current, future)`` ready for the next step.
    """
    r = robert_coeff

    # Explicit tendency at current time
    explicit_current = explicit_fn(current)

    # Implicit tendency at previous time
    implicit_previous = implicit_fn(previous)

    # Form intermediate: x_{n-1} + 2dt * (F(x_n) + (1-alpha)*L(x_{n-1}))
    intermediate = jax.tree.map(
        lambda xp, fe, li: xp + 2.0 * dt * (fe + (1.0 - alpha) * li),
        previous,
        explicit_current,
        implicit_previous,
    )

    # Implicit solve
    eta = 2.0 * dt * alpha
    future = inverse_fn(intermediate, eta)

    # Robert-Asselin filter on current
    filtered_current = jax.tree.map(
        lambda p, c, f: (1.0 - 2.0 * r) * c + r * (p + f),
        previous,
        current,
        future,
    )

    return filtered_current, future


# =====================================================================
# PE convenience factory
# =====================================================================


def build_pe_stepper(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    reference_temperature: np.ndarray,
    surface_geopotential: jnp.ndarray,
    dt: float,
    spectral_filter: jnp.ndarray | None = None,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
    robert_coeff: float = 0.05,
    alpha: float = 0.5,
    forcing: Forcing | None = None,
) -> tuple[
    Callable[[PrimitiveEquationState], tuple[PrimitiveEquationState, PrimitiveEquationState]],
    Callable[
        [PrimitiveEquationState, PrimitiveEquationState],
        tuple[PrimitiveEquationState, PrimitiveEquationState],
    ],
]:
    """Build init and step functions for PE IMEX time integration.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    reference_temperature : np.ndarray
        Reference temperature profile T_ref, shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential g·z_s in spectral space.
    dt : float
        Timestep [s].
    spectral_filter : jnp.ndarray or None
        Multiplicative spectral filter array from :func:`exponential_filter`.
        Applied to ``future`` after each step.  None disables filtering.
    diffusion_order : int
        Order of hyperdiffusion (4 = del-8).
    diffusion_timescale : float
        E-folding damping time [s].
    robert_coeff : float
        Robert-Asselin filter coefficient (0.05 standard).
    alpha : float
        Implicit weighting (0.5 = centred).
    forcing : Forcing or None
        Physics forcing callable (e.g. Held-Suarez).  When provided,
        its tendencies are added to the explicit dynamics tendencies
        at each time step.  None disables physics forcing.

    Returns
    -------
    tuple[init_fn, step_fn]
        ``init_fn(state) -> (previous, current)``
        ``step_fn(previous, current) -> (filtered_current, future)``
    """
    arrays = transform.arrays
    t_ref = np.asarray(reference_temperature)

    # Build the explicit tendency function (JIT-compiled internally)
    explicit_fn = primitive_equation_tendencies(
        transform,
        planet,
        levels,
        t_ref,
        surface_geopotential,
        diffusion_order=diffusion_order,
        diffusion_timescale=diffusion_timescale,
    )

    # Compose dynamics + physics forcing if provided
    if forcing is not None:
        dynamics_fn = explicit_fn

        def explicit_fn(state: PrimitiveEquationState) -> PrimitiveEquationState:
            dyn_tend = dynamics_fn(state)
            lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
            ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)
            phys_tend = forcing(state, ps_grid)
            return jax.tree.map(jnp.add, dyn_tend, phys_tend)

    # Build the semi-implicit config (precomputes G, H, M matrices)
    si_config = build_pe_semi_implicit_config(
        levels,
        planet.gas_constant,
        planet.kappa,
        t_ref,
        alpha=alpha,
    )

    # Wrap implicit functions with partially applied config
    def implicit_fn(state: PrimitiveEquationState) -> PrimitiveEquationState:
        return pe_implicit_terms(state, si_config, arrays)

    def inverse_fn(
        state: PrimitiveEquationState,
        step_size: float,
    ) -> PrimitiveEquationState:
        return pe_implicit_inverse(state, step_size, si_config, arrays)

    # Spectral filter helper
    def _apply_filter(state: PrimitiveEquationState) -> PrimitiveEquationState:
        if spectral_filter is None:
            return state
        f = spectral_filter
        return state.replace(
            vorticity=state.vorticity * f[None, :],
            divergence=state.divergence * f[None, :],
            temperature=state.temperature * f[None, :],
            log_surface_pressure=state.log_surface_pressure * f,
        )

    # Build init_fn
    @jax.jit
    def init_fn(
        state: PrimitiveEquationState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState]:
        previous, current = euler_init(state, explicit_fn, inverse_fn, dt)
        current = _apply_filter(current)
        return previous, current

    # Build step_fn
    @jax.jit
    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState]:
        filtered_current, future = imex_leapfrog_step(
            previous,
            current,
            explicit_fn,
            implicit_fn,
            inverse_fn,
            dt,
            alpha=alpha,
            robert_coeff=robert_coeff,
        )
        future = _apply_filter(future)
        return filtered_current, future

    return init_fn, step_fn
