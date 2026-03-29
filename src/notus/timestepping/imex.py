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
from notus.physics.forcing import Forcing, ImplicitForcing, MoistForcing, PhysicsDiagnostics
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


def _compose_dynamics_physics(
    dynamics_fn: Callable[[PrimitiveEquationState], PrimitiveEquationState],
    forcing: Forcing | None,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
) -> Callable[[PrimitiveEquationState], tuple[PrimitiveEquationState, PhysicsDiagnostics]]:
    """Compose dynamics and physics into a single explicit tendency.

    Returns a callable that produces ``(combined_tendency, diagnostics)``.
    """
    if forcing is None:

        def dynamics_only(
            state: PrimitiveEquationState,
        ) -> tuple[PrimitiveEquationState, PhysicsDiagnostics]:
            return dynamics_fn(state), PhysicsDiagnostics()

        return dynamics_only

    def combined(
        state: PrimitiveEquationState,
    ) -> tuple[PrimitiveEquationState, PhysicsDiagnostics]:
        dyn_tend = dynamics_fn(state)
        lnps_grid = transform.spectral_to_grid(state.log_surface_pressure)
        ps_grid = planet.reference_pressure * jnp.exp(lnps_grid)
        phys_tend, diags = forcing(state, ps_grid)
        return jax.tree.map(jnp.add, dyn_tend, phys_tend), diags

    return combined


def _detect_forcing_capabilities(
    forcing: Forcing | None,
    t_ref: np.ndarray,
) -> tuple[
    Callable[[PrimitiveEquationState, float], PrimitiveEquationState] | None,
    np.ndarray | None,
]:
    """Detect implicit physics and reference humidity from forcing.

    Uses :class:`~notus.physics.forcing.ImplicitForcing` and
    :class:`~notus.physics.forcing.MoistForcing` protocols to check
    for optional capabilities on the forcing object.
    """
    implicit_physics: Callable[[PrimitiveEquationState, float], PrimitiveEquationState] | None = (
        None
    )
    if isinstance(forcing, ImplicitForcing):
        implicit_physics = forcing.apply_implicit

    reference_humidity: np.ndarray | None = None
    if isinstance(forcing, MoistForcing):
        reference_humidity = forcing.compute_reference_humidity(t_ref)

    return implicit_physics, reference_humidity


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
    alpha: float = 0.5,
    forcing: Forcing | None = None,
) -> tuple[
    Callable[
        [PrimitiveEquationState],
        tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics],
    ],
    Callable[
        [PrimitiveEquationState, PrimitiveEquationState],
        tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics],
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
        Physics forcing callable (e.g. Held-Suarez, PhysicsSuite).
        When provided, its tendencies are added to the explicit dynamics
        at each time step.  None disables physics forcing.

        If the forcing object exposes optional capabilities, they are
        wired automatically:

        - ``apply_implicit(state, dt) -> state``: implicit treatment of
          stiff physics (surface fluxes, friction, convection).  Called
          with ``dt`` for Euler init and ``2·dt`` for leapfrog steps.
        - ``compute_reference_humidity(T_ref) -> q_ref``: reference
          humidity profile for virtual temperature linearization in the
          semi-implicit solver.

    Returns
    -------
    tuple[init_fn, step_fn]
        ``init_fn(state) -> (previous, current, diagnostics)``
        ``step_fn(previous, current) -> (filtered_current, future, diagnostics)``
    """
    arrays = transform.arrays
    t_ref = np.asarray(reference_temperature)

    # Auto-detect capabilities from the forcing object
    implicit_physics, reference_humidity = _detect_forcing_capabilities(forcing, t_ref)

    # Virtual temperature reference for the semi-implicit solver
    epsilon_v, tv_ref = _compute_virtual_reference(t_ref, reference_humidity, planet)

    # Build the explicit tendency function (JIT-compiled internally)
    dynamics_fn = primitive_equation_tendencies(
        transform,
        planet,
        levels,
        t_ref,
        surface_geopotential,
        diffusion_order=diffusion_order,
        diffusion_timescale=diffusion_timescale,
        reference_virtual_temperature=tv_ref,
    )

    # Compose dynamics + physics forcing → returns (tendency, diagnostics)
    combined_fn = _compose_dynamics_physics(
        dynamics_fn,
        forcing,
        transform,
        planet,
    )

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

    # Post-step corrections: filter, humidity clipping, implicit physics
    def _post_step(
        state: PrimitiveEquationState,
        dt_implicit: float,
    ) -> PrimitiveEquationState:
        state = _apply_filter(state)
        state = _clip_humidity(state, transform)
        if implicit_physics is not None:
            state = implicit_physics(state, dt_implicit)
        return state

    # Build init_fn (inlines euler_init to capture diagnostics)
    @jax.jit
    def init_fn(
        state: PrimitiveEquationState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics]:
        tendency, diags = combined_fn(state)
        intermediate = jax.tree.map(lambda x, f: x + dt * f, state, tendency)
        current = inverse_fn(intermediate, dt)
        current = _post_step(current, dt)
        return state, current, diags

    # Build step_fn (inlines imex_leapfrog_step to capture diagnostics)
    r = robert_coeff

    @jax.jit
    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics]:
        explicit_current, diags = combined_fn(current)
        implicit_previous = implicit_fn(previous)
        intermediate = jax.tree.map(
            lambda xp, fe, li: xp + 2.0 * dt * (fe + (1.0 - alpha) * li),
            previous,
            explicit_current,
            implicit_previous,
        )
        eta = 2.0 * dt * alpha
        future = inverse_fn(intermediate, eta)
        filtered_current = jax.tree.map(
            lambda p, c, f: (1.0 - 2.0 * r) * c + r * (p + f),
            previous,
            current,
            future,
        )
        future = _post_step(future, 2.0 * dt)
        return filtered_current, future, diags

    return init_fn, step_fn
