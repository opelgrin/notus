"""Simulation runner for Notus GCM integrations.

Encapsulates the standard time integration loop with support for:

- Atmosphere-only (Held-Suarez, simple physics) and coupled (slab ocean)
  configurations
- Seasonal forcing with automatic day-of-year tracking
- Per-day diagnostic callbacks
- Restart save/load via :mod:`notus.io`

Examples
--------
Atmosphere-only Held-Suarez::

    from notus import (
        EARTH, GaussianGrid, SpectralTransform, HeldSuarez,
        held_suarez_initial_state, uniform_sigma_levels,
        build_pe_stepper, exponential_filter,
    )
    from notus.runner import run_simulation

    grid = GaussianGrid(21)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(20)
    state, ref_temps, surf_phi = held_suarez_initial_state(transform, EARTH, levels)
    forcing = HeldSuarez(transform, EARTH, levels)
    filt = exponential_filter(transform.arrays, 1200.0)
    init_fn, step_fn = build_pe_stepper(
        transform, EARTH, levels, ref_temps, surf_phi,
        dt=1200.0, spectral_filter=filt, forcing=forcing,
    )

    result = run_simulation(
        init_fn=init_fn, step_fn=step_fn,
        initial_state=state, dt=1200.0, n_days=1200,
    )

Coupled slab-ocean with seasonal cycle::

    from notus.runner import run_simulation

    result = run_simulation(
        init_fn=init_fn, step_fn=step_fn,
        initial_state=state, dt=1200.0, n_days=3600,
        surface=surface, forcing=forcing,
        days_per_year=365.25,
    )
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from typing import overload

import jax
import jax.numpy as jnp

from notus.physics.physics_suite import PhysicsSuite
from notus.physics.surface import SurfaceState
from notus.state import PrimitiveEquationState


logger = logging.getLogger(__name__)

# Type aliases for the two modes
_AtmCarry = tuple[PrimitiveEquationState, PrimitiveEquationState]
_CoupledCarry = tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState]

_AtmInitFn = Callable[[PrimitiveEquationState], _AtmCarry]
_AtmStepFn = Callable[
    [PrimitiveEquationState, PrimitiveEquationState],
    tuple[PrimitiveEquationState, PrimitiveEquationState],
]
_AtmCallback = Callable[[int, PrimitiveEquationState], object]

_CoupledInitFn = Callable[[PrimitiveEquationState, SurfaceState], _CoupledCarry]
_CoupledStepFn = Callable[
    [PrimitiveEquationState, PrimitiveEquationState, SurfaceState],
    tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState],
]
_CoupledCallback = Callable[[int, PrimitiveEquationState, SurfaceState], object]


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    """Result of a completed simulation.

    Attributes
    ----------
    state : PrimitiveEquationState
        Final atmospheric state (the ``current`` time level).
    previous : PrimitiveEquationState
        Previous time level (needed to continue the integration).
    surface : SurfaceState or None
        Final surface state (coupled runs only).
    n_days : int
        Number of days simulated.
    wall_time : float
        Total wall-clock time [s].
    diagnostics : list
        Collected outputs from the ``on_day`` callback.
    """

    state: PrimitiveEquationState
    previous: PrimitiveEquationState
    surface: SurfaceState | None
    n_days: int
    wall_time: float
    diagnostics: list[object]


def _build_atm_one_day(
    step_fn: _AtmStepFn,
    forcing: PhysicsSuite | None,
    seasonal: bool,
    steps_per_day: int,
) -> Callable[[_AtmCarry, jnp.ndarray], tuple[_AtmCarry, None]]:
    """Build JIT-compiled one-day function for atmosphere-only runs."""

    def one_day(carry: _AtmCarry, day_of_year: jnp.ndarray) -> tuple[_AtmCarry, None]:
        prev, curr = carry
        if forcing is not None and seasonal:
            forcing.day_of_year = day_of_year

        def step(carry: _AtmCarry, _: None) -> tuple[_AtmCarry, None]:
            p, c = carry
            p, c = step_fn(p, c)
            return (p, c), None

        (prev, curr), _ = jax.lax.scan(step, (prev, curr), None, length=steps_per_day)
        return (prev, curr), None

    return jax.jit(one_day)


def _build_coupled_one_day(
    step_fn: _CoupledStepFn,
    forcing: PhysicsSuite | None,
    seasonal: bool,
    steps_per_day: int,
) -> Callable[[_CoupledCarry, jnp.ndarray], tuple[_CoupledCarry, None]]:
    """Build JIT-compiled one-day function for coupled runs."""

    def one_day(carry: _CoupledCarry, day_of_year: jnp.ndarray) -> tuple[_CoupledCarry, None]:
        prev, curr, sfc = carry
        if forcing is not None:
            if seasonal:
                forcing.day_of_year = day_of_year
            forcing.prescribed_sst = sfc.ocean.surface_temperature

        def step(carry: _CoupledCarry, _: None) -> tuple[_CoupledCarry, None]:
            p, c, s = carry
            p, c, s = step_fn(p, c, s)
            return (p, c, s), None

        (prev, curr, sfc), _ = jax.lax.scan(step, (prev, curr, sfc), None, length=steps_per_day)
        return (prev, curr, sfc), None

    return jax.jit(one_day)


# --- Public API: two overloads for atmosphere-only vs coupled ---


@overload
def run_simulation(
    init_fn: _AtmInitFn,
    step_fn: _AtmStepFn,
    initial_state: PrimitiveEquationState,
    dt: float,
    n_days: int,
    *,
    forcing: PhysicsSuite | None = ...,
    days_per_year: float = ...,
    start_day: int = ...,
    on_day: _AtmCallback | None = ...,
    verbose: bool = ...,
    log_interval: int = ...,
) -> SimulationResult: ...


@overload
def run_simulation(
    init_fn: _CoupledInitFn,
    step_fn: _CoupledStepFn,
    initial_state: PrimitiveEquationState,
    dt: float,
    n_days: int,
    *,
    surface: SurfaceState,
    forcing: PhysicsSuite | None = ...,
    days_per_year: float = ...,
    start_day: int = ...,
    on_day: _CoupledCallback | None = ...,
    verbose: bool = ...,
    log_interval: int = ...,
) -> SimulationResult: ...


def run_simulation(
    init_fn: _AtmInitFn | _CoupledInitFn,
    step_fn: _AtmStepFn | _CoupledStepFn,
    initial_state: PrimitiveEquationState,
    dt: float,
    n_days: int,
    *,
    surface: SurfaceState | None = None,
    forcing: PhysicsSuite | None = None,
    days_per_year: float = 0.0,
    start_day: int = 0,
    on_day: _AtmCallback | _CoupledCallback | None = None,
    verbose: bool = True,
    log_interval: int = 50,
) -> SimulationResult:
    """Run a GCM integration for a given number of days.

    Handles both atmosphere-only and coupled surface configurations,
    with optional seasonal forcing and per-day diagnostics.

    Parameters
    ----------
    init_fn : callable
        Initialization function from ``build_pe_stepper`` or
        ``build_coupled_pe_stepper``.
    step_fn : callable
        Step function from ``build_pe_stepper`` or
        ``build_coupled_pe_stepper``.
    initial_state : PrimitiveEquationState
        Initial atmospheric state in spectral space.
    dt : float
        Timestep [s].
    n_days : int
        Number of simulation days.
    surface : SurfaceState or None
        Initial surface state.  Must be provided for coupled runs
        (i.e. when ``step_fn`` expects a surface argument).
    forcing : PhysicsSuite or None
        Physics forcing object.  Required when ``days_per_year > 0``
        (seasonal forcing) or when running coupled (to sync SST for
        radiation).
    days_per_year : float
        Length of year in days.  When > 0, enables seasonal insolation
        by setting ``forcing.day_of_year`` each day.  Set to 0 to
        disable seasonal cycle.
    start_day : int
        Starting day number (for continuing from a restart).
    on_day : callable or None
        Callback invoked after each simulated day.  Signature depends
        on mode:

        - **Atmosphere-only**: ``on_day(day, state) -> result``
        - **Coupled**: ``on_day(day, state, surface) -> result``

        Return values are collected in ``SimulationResult.diagnostics``.
        Return ``None`` to skip collecting for that day.
    verbose : bool
        Print progress messages.
    log_interval : int
        Print status every ``log_interval`` days.

    Returns
    -------
    SimulationResult
        Final state, surface state, wall time, and collected diagnostics.
    """
    coupled = surface is not None
    seasonal = days_per_year > 0.0
    steps_per_day = int(86400 / dt)

    if seasonal and forcing is None:
        msg = "forcing must be provided when days_per_year > 0"
        raise ValueError(msg)

    if verbose:
        logger.info(
            "Starting %s simulation: %d days, dt=%.0fs (%d steps/day)",
            "coupled" if coupled else "atmosphere-only",
            n_days,
            dt,
            steps_per_day,
        )

    # The overload signatures guarantee type safety at call sites.
    # mypy cannot narrow union Callable types through the coupled branch,
    # so we suppress arg-type for the dispatch to typed internal functions.
    if coupled and surface is not None:
        return _run_coupled(
            init_fn,  # type: ignore[arg-type]
            step_fn,  # type: ignore[arg-type]
            initial_state,
            surface,
            forcing,
            seasonal,
            days_per_year,
            steps_per_day,
            start_day,
            n_days,
            on_day,  # type: ignore[arg-type]
            verbose,
            log_interval,
        )
    return _run_atm_only(
        init_fn,  # type: ignore[arg-type]
        step_fn,  # type: ignore[arg-type]
        initial_state,
        forcing,
        seasonal,
        days_per_year,
        steps_per_day,
        start_day,
        n_days,
        on_day,  # type: ignore[arg-type]
        verbose,
        log_interval,
    )


def _run_atm_only(
    init_fn: _AtmInitFn,
    step_fn: _AtmStepFn,
    initial_state: PrimitiveEquationState,
    forcing: PhysicsSuite | None,
    seasonal: bool,
    days_per_year: float,
    steps_per_day: int,
    start_day: int,
    n_days: int,
    on_day: _AtmCallback | None,
    verbose: bool,
    log_interval: int,
) -> SimulationResult:
    """Run atmosphere-only integration."""
    one_day_jit = _build_atm_one_day(step_fn, forcing, seasonal, steps_per_day)
    diagnostics: list[object] = []

    t0 = time.perf_counter()
    prev, curr = init_fn(initial_state)
    day_val = jnp.float64(start_day % days_per_year if seasonal else 0.0)
    (prev, curr), _ = one_day_jit((prev, curr), day_val)
    if verbose:
        logger.info("Day 1 (incl. JIT compile): %.1fs", time.perf_counter() - t0)

    _invoke_atm_callback(on_day, start_day + 1, curr, diagnostics)

    t_start = time.perf_counter()
    for day_idx in range(2, n_days + 1):
        day = start_day + day_idx
        day_val = jnp.float64(day % days_per_year if seasonal else 0.0)
        (prev, curr), _ = one_day_jit((prev, curr), day_val)
        _invoke_atm_callback(on_day, day, curr, diagnostics)
        _log_progress(verbose, day_idx, day, n_days, log_interval, t_start)

    return SimulationResult(
        state=curr,
        previous=prev,
        surface=None,
        n_days=n_days,
        wall_time=time.perf_counter() - t0,
        diagnostics=diagnostics,
    )


def _run_coupled(
    init_fn: _CoupledInitFn,
    step_fn: _CoupledStepFn,
    initial_state: PrimitiveEquationState,
    surface: SurfaceState,
    forcing: PhysicsSuite | None,
    seasonal: bool,
    days_per_year: float,
    steps_per_day: int,
    start_day: int,
    n_days: int,
    on_day: _CoupledCallback | None,
    verbose: bool,
    log_interval: int,
) -> SimulationResult:
    """Run coupled atmosphere-surface integration."""
    one_day_jit = _build_coupled_one_day(step_fn, forcing, seasonal, steps_per_day)
    diagnostics: list[object] = []

    t0 = time.perf_counter()
    prev, curr, surface = init_fn(initial_state, surface)
    day_val = jnp.float64(start_day % days_per_year if seasonal else 0.0)
    (prev, curr, surface), _ = one_day_jit((prev, curr, surface), day_val)
    if verbose:
        logger.info("Day 1 (incl. JIT compile): %.1fs", time.perf_counter() - t0)

    _invoke_coupled_callback(on_day, start_day + 1, curr, surface, diagnostics)

    t_start = time.perf_counter()
    for day_idx in range(2, n_days + 1):
        day = start_day + day_idx
        day_val = jnp.float64(day % days_per_year if seasonal else 0.0)
        (prev, curr, surface), _ = one_day_jit((prev, curr, surface), day_val)
        _invoke_coupled_callback(on_day, day, curr, surface, diagnostics)
        _log_progress(verbose, day_idx, day, n_days, log_interval, t_start)

    return SimulationResult(
        state=curr,
        previous=prev,
        surface=surface,
        n_days=n_days,
        wall_time=time.perf_counter() - t0,
        diagnostics=diagnostics,
    )


def _invoke_atm_callback(
    on_day: _AtmCallback | None,
    day: int,
    state: PrimitiveEquationState,
    diagnostics: list[object],
) -> None:
    """Invoke atmosphere-only callback and collect non-None results."""
    if on_day is None:
        return
    result = on_day(day, state)
    if result is not None:
        diagnostics.append(result)


def _invoke_coupled_callback(
    on_day: _CoupledCallback | None,
    day: int,
    state: PrimitiveEquationState,
    surface: SurfaceState,
    diagnostics: list[object],
) -> None:
    """Invoke coupled callback and collect non-None results."""
    if on_day is None:
        return
    result = on_day(day, state, surface)
    if result is not None:
        diagnostics.append(result)


def _log_progress(
    verbose: bool,
    day_idx: int,
    day: int,
    n_days: int,
    log_interval: int,
    t_start: float,
) -> None:
    """Log progress at configured intervals."""
    if not verbose:
        return
    if day_idx <= 3 or day_idx % log_interval == 0 or day_idx == n_days:  # noqa: PLR2004
        elapsed = time.perf_counter() - t_start
        rate = (day_idx - 1) / elapsed if elapsed > 0 else 0
        logger.info("  Day %5d: %.1f sim-days/s", day, rate)
