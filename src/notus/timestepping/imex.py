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
    raw_alpha: float = 0.53,
) -> tuple[S, S]:
    """One IMEX leapfrog step with Robert-Asselin-Williams (RAW) filter.

    Evaluates explicit tendencies at the current time level and implicit
    terms at the previous time level, then solves the implicit system
    and applies the RAW filter (Williams 2009, MWR).

    The RAW filter improves on the standard Robert-Asselin filter by
    compensating the amplitude damping of the physical mode.  The RA
    increment ``d = r·(x_{n-1} - 2·x_n + x_{n+1})`` is split between
    the current and future states::

        x_n^f     = x_n     + (1 - raw_alpha) · d
        x_{n+1}^f = x_{n+1} + raw_alpha · d

    Setting ``raw_alpha = 0`` recovers the standard Robert-Asselin filter.
    ``raw_alpha = 0.53`` is the Williams (2009) recommended value.

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
    raw_alpha : float
        Williams (2009) RAW filter parameter α (0.53 recommended).
        0 = standard Robert-Asselin, 0.53 = RAW default.

    Returns
    -------
    tuple[S, S]
        ``(filtered_current, filtered_future)`` ready for the next step.
    """
    r = robert_coeff
    a = raw_alpha

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

    # Robert-Asselin-Williams filter (Williams 2009)
    # d = r·(x_{n-1} - 2·x_n + x_{n+1})
    # x_n^f     = x_n     + (1 - α)·d
    # x_{n+1}^f = x_{n+1} + α·d
    filtered_current = jax.tree.map(
        lambda p, c, f: c + (1.0 - a) * r * (p - 2.0 * c + f),
        previous,
        current,
        future,
    )
    filtered_future = jax.tree.map(
        lambda p, c, f: f + a * r * (p - 2.0 * c + f),
        previous,
        current,
        future,
    )

    return filtered_current, filtered_future


# =====================================================================
# PE convenience factory
# =====================================================================


def _clip_humidity(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
) -> PrimitiveEquationState:
    """Clip negative humidity in grid space and re-project to spectral.

    Spectral Gibbs ringing creates unphysical negative humidity values at
    sharp moisture gradients.  Without correction these accumulate as a
    systematic moisture sink (18 %+ of grid points after 1200 days at T21
    L20).  Clipping in grid space and transforming back corrects the
    spectral representation itself, keeping negatives bounded at < 0.5 %.
    """
    if state.humidity is None:
        return state
    q_grid = jax.vmap(transform.spectral_to_grid)(state.humidity)
    q_grid = jnp.maximum(q_grid, 0.0)
    q_spec = jax.vmap(transform.grid_to_spectral)(q_grid)
    return state.replace(humidity=q_spec)


def _compute_virtual_reference(
    t_ref: np.ndarray,
    reference_humidity: np.ndarray | None,
    planet: PlanetaryConstants,
) -> tuple[float, np.ndarray | None]:
    """Compute virtual temperature reference from humidity profile.

    Returns ``(epsilon_v, T_v_ref)`` where ``T_v_ref`` is None when
    no reference humidity is provided (dry dynamics).
    """
    if reference_humidity is None:
        return 0.0, None
    epsilon_v = 1.0 / planet.epsilon_moisture - 1.0
    q_ref = np.asarray(reference_humidity)
    tv_ref = t_ref * (1.0 + epsilon_v * q_ref)
    return epsilon_v, tv_ref


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
    raw_alpha: float = 0.0,
    alpha: float = 0.5,
    forcing: Forcing | None = None,
    reference_humidity: np.ndarray | None = None,
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
    raw_alpha : float
        Williams (2009) RAW filter parameter (0.53 recommended).
        0 = standard Robert-Asselin, 0.53 = RAW default.
    alpha : float
        Implicit weighting (0.5 = centred).
    forcing : Forcing or None
        Physics forcing callable (e.g. Held-Suarez).  When provided,
        its tendencies are added to the explicit dynamics tendencies
        at each time step.  None disables physics forcing.
    reference_humidity : np.ndarray or None
        Reference specific humidity profile, shape ``(n_levels,)``.
        When provided, the semi-implicit solver and explicit pressure
        gradient linearize around the virtual reference temperature
        T_v_ref = T_ref · (1 + ε_v · q_ref), absorbing the leading-order
        moisture contribution into the implicit solve.

    Returns
    -------
    tuple[init_fn, step_fn]
        ``init_fn(state) -> (previous, current)``
        ``step_fn(previous, current) -> (filtered_current, future)``
    """
    arrays = transform.arrays
    t_ref = np.asarray(reference_temperature)

    # Virtual temperature reference for the semi-implicit solver
    epsilon_v, tv_ref = _compute_virtual_reference(t_ref, reference_humidity, planet)

    # Build the explicit tendency function (JIT-compiled internally)
    explicit_fn = primitive_equation_tendencies(
        transform,
        planet,
        levels,
        t_ref,
        surface_geopotential,
        diffusion_order=diffusion_order,
        diffusion_timescale=diffusion_timescale,
        reference_virtual_temperature=tv_ref,
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
        reference_humidity=reference_humidity,
        epsilon_v=epsilon_v,
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
        updates: dict[str, jnp.ndarray] = {
            "vorticity": state.vorticity * f[None, :],
            "divergence": state.divergence * f[None, :],
            "temperature": state.temperature * f[None, :],
            "log_surface_pressure": state.log_surface_pressure * f,
        }
        if state.humidity is not None:
            updates["humidity"] = state.humidity * f[None, :]
        return state.replace(**updates)

    # Build init_fn
    @jax.jit
    def init_fn(
        state: PrimitiveEquationState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState]:
        previous, current = euler_init(state, explicit_fn, inverse_fn, dt)
        current = _apply_filter(current)
        current = _clip_humidity(current, transform)
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
            raw_alpha=raw_alpha,
        )
        future = _apply_filter(future)
        future = _clip_humidity(future, transform)
        return filtered_current, future

    return init_fn, step_fn
