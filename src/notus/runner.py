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
import numpy as np

from notus.constants import PlanetaryConstants
from notus.operators import exponential_filter
from notus.physics.forcing import PhysicsDiagnostics
from notus.physics.physics_suite import PhysicsSuite
from notus.physics.surface import (
    BucketLandConfig,
    OceanState,
    SlabOceanConfig,
    SurfaceState,
    init_land_state,
)
from notus.physics.surface_types import SurfaceProperties
from notus.state import PrimitiveEquationState
from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.timestepping.spinup import spinup_prescribed_sst
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


logger = logging.getLogger(__name__)


class NaNError(RuntimeError):
    """Raised when NaN is detected in atmospheric state during integration."""


def _check_nan(state: PrimitiveEquationState, context: str) -> None:
    """Check for NaN in prognostic fields and raise if found.

    Parameters
    ----------
    state : PrimitiveEquationState
        Atmospheric state to check.
    context : str
        Human-readable description of when the check was performed
        (e.g. "after init" or "after day 42").
    """
    has_nan = jnp.any(jnp.isnan(state.temperature)) | jnp.any(jnp.isnan(state.log_surface_pressure))
    if state.humidity is not None:
        has_nan |= jnp.any(jnp.isnan(state.humidity))
    if has_nan:
        raise NaNError(
            f"NaN detected in atmospheric state {context}. "
            "This usually indicates numerical instability — try reducing dt."
        )


# Type aliases for the two modes
_AtmCarry = tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics]
_CoupledCarry = tuple[
    PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics
]

_AtmInitFn = Callable[
    [PrimitiveEquationState],
    tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics],
]
_AtmStepFn = Callable[
    [PrimitiveEquationState, PrimitiveEquationState],
    tuple[PrimitiveEquationState, PrimitiveEquationState, PhysicsDiagnostics],
]
_AtmCallback = Callable[[int, PrimitiveEquationState, PhysicsDiagnostics], object]

_CoupledInitFn = Callable[
    [PrimitiveEquationState, SurfaceState, jnp.ndarray],
    tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics],
]
_CoupledStepFn = Callable[
    [PrimitiveEquationState, PrimitiveEquationState, SurfaceState, jnp.ndarray],
    tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics],
]
_CoupledCallback = Callable[[int, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics], object]


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
    steps_per_day: int,
) -> Callable[[_AtmCarry, jnp.ndarray], tuple[_AtmCarry, None]]:
    """Build JIT-compiled one-day function for atmosphere-only runs."""

    def one_day(carry: _AtmCarry, day_of_year: jnp.ndarray) -> tuple[_AtmCarry, None]:
        prev, curr, prev_diags = carry

        def step(carry: _AtmCarry, _: None) -> tuple[_AtmCarry, None]:
            p, c, pd = carry
            p, c, pd = step_fn(p, c)
            return (p, c, pd), None

        carry_out, _ = jax.lax.scan(
            step,
            (prev, curr, prev_diags),
            None,
            length=steps_per_day,
        )
        return carry_out, None

    return jax.jit(one_day)


def _build_coupled_one_day(
    step_fn: _CoupledStepFn,
    steps_per_day: int,
) -> Callable[[_CoupledCarry, jnp.ndarray], tuple[_CoupledCarry, None]]:
    """Build JIT-compiled one-day function for coupled runs."""

    def one_day(carry: _CoupledCarry, day_of_year: jnp.ndarray) -> tuple[_CoupledCarry, None]:
        prev, curr, sfc, prev_diags = carry

        def step(carry: _CoupledCarry, _: None) -> tuple[_CoupledCarry, None]:
            p, c, s, pd = carry
            p, c, s, pd = step_fn(p, c, s, day_of_year)
            return (p, c, s, pd), None

        carry_out, _ = jax.lax.scan(
            step,
            (prev, curr, sfc, prev_diags),
            None,
            length=steps_per_day,
        )
        return carry_out, None

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
        by passing dynamic ``day_of_year`` values through the coupled
        stepper.  Set to 0 to disable seasonal cycle.
    start_day : int
        Starting day number (for continuing from a restart).
    on_day : callable or None
        Callback invoked after each simulated day.  Signature depends
        on mode:

        - **Atmosphere-only**: ``on_day(day, state, diags)``
        - **Coupled**: ``on_day(day, state, surface, diags)``

        where ``diags`` is a :class:`~notus.physics.forcing.PhysicsDiagnostics`
        with radiation fluxes, precipitation, etc.

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
    one_day_jit = _build_atm_one_day(step_fn, steps_per_day)
    diagnostics: list[object] = []

    t0 = time.perf_counter()
    prev, curr, diags = init_fn(initial_state)
    _check_nan(curr, "after init")
    day_val = jnp.float64(start_day % days_per_year if seasonal else 0.0)
    if forcing is not None and seasonal:
        forcing.day_of_year = day_val
    (prev, curr, diags), _ = one_day_jit((prev, curr, diags), day_val)
    _check_nan(curr, f"after day {start_day + 1}")
    if verbose:
        logger.info("Day 1 (incl. JIT compile): %.1fs", time.perf_counter() - t0)

    _invoke_atm_callback(on_day, start_day + 1, curr, diags, diagnostics)

    t_start = time.perf_counter()
    for day_idx in range(2, n_days + 1):
        day = start_day + day_idx
        day_val = jnp.float64(day % days_per_year if seasonal else 0.0)
        if forcing is not None and seasonal:
            forcing.day_of_year = day_val
        (prev, curr, diags), _ = one_day_jit((prev, curr, diags), day_val)
        _check_nan(curr, f"after day {day}")
        _invoke_atm_callback(on_day, day, curr, diags, diagnostics)
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
    one_day_jit = _build_coupled_one_day(step_fn, steps_per_day)
    diagnostics: list[object] = []

    t0 = time.perf_counter()
    day_val = jnp.float64(start_day % days_per_year if seasonal else 0.0)
    prev, curr, surface, diags = init_fn(initial_state, surface, day_val)
    _check_nan(curr, "after init")
    if forcing is not None:
        forcing.prescribed_sst = surface.ocean.surface_temperature
    (prev, curr, surface, diags), _ = one_day_jit((prev, curr, surface, diags), day_val)
    _check_nan(curr, f"after day {start_day + 1}")
    if verbose:
        logger.info("Day 1 (incl. JIT compile): %.1fs", time.perf_counter() - t0)

    _invoke_coupled_callback(on_day, start_day + 1, curr, surface, diags, diagnostics)

    t_start = time.perf_counter()
    for day_idx in range(2, n_days + 1):
        day = start_day + day_idx
        day_val = jnp.float64(day % days_per_year if seasonal else 0.0)
        if forcing is not None:
            forcing.prescribed_sst = surface.ocean.surface_temperature
        (prev, curr, surface, diags), _ = one_day_jit((prev, curr, surface, diags), day_val)
        _check_nan(curr, f"after day {day}")
        _invoke_coupled_callback(on_day, day, curr, surface, diags, diagnostics)
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
    diags: PhysicsDiagnostics,
    diagnostics: list[object],
) -> None:
    """Invoke atmosphere-only callback and collect non-None results."""
    if on_day is None:
        return
    result = on_day(day, state, diags)
    if result is not None:
        diagnostics.append(result)


def _invoke_coupled_callback(
    on_day: _CoupledCallback | None,
    day: int,
    state: PrimitiveEquationState,
    surface: SurfaceState,
    diags: PhysicsDiagnostics,
    diagnostics: list[object],
) -> None:
    """Invoke coupled callback and collect non-None results."""
    if on_day is None:
        return
    result = on_day(day, state, surface, diags)
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


# ---------------------------------------------------------------------------
# Coupled convenience runner
# ---------------------------------------------------------------------------


def run_coupled_simulation(
    initial_state: PrimitiveEquationState,
    forcing: PhysicsSuite,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    reference_temperature: np.ndarray,
    surface_geopotential: jnp.ndarray,
    dt: float,
    n_days: int,
    *,
    ocean_config: SlabOceanConfig | None = None,
    surface_properties: SurfaceProperties | None = None,
    land_config: BucketLandConfig | None = None,
    spinup_days: int = 100,
    averaging_days: int = 100,
    spectral_filter: jnp.ndarray | None = None,
    surface_albedo: float | None = None,
    days_per_year: float = 365.25,
    on_day: _CoupledCallback | None = None,
    verbose: bool = True,
    log_interval: int = 50,
) -> SimulationResult:
    """Run a coupled atmosphere-ocean(-land) simulation with automatic Q-flux spinup.

    This is a convenience wrapper that handles the full coupled simulation
    workflow:

    1. Prescribed-SST spinup to diagnose the ocean Q-flux
    2. Surface state initialization (ocean + optional land)
    3. Coupled time stepping with slab ocean (and optional bucket land)

    Skipping the prescribed-SST spinup is a common source of instability
    when running coupled simulations.  This function ensures the spinup
    always runs before the coupled integration begins.

    Parameters
    ----------
    initial_state : PrimitiveEquationState
        Initial atmospheric state (can be a cold isothermal start).
    forcing : PhysicsSuite
        Physics forcing (radiation, surface fluxes, etc.).
    transform : SpectralTransform
        Spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    reference_temperature : np.ndarray
        Reference temperature profile for the semi-implicit solver,
        shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential in spectral space.
    dt : float
        Timestep [s].
    n_days : int
        Number of coupled simulation days.
    ocean_config : SlabOceanConfig or None
        Slab ocean parameters.  Defaults to ``SlabOceanConfig()``.
    surface_properties : SurfaceProperties or None
        Spatially varying surface albedo, roughness, and land fraction.
        Required when ``land_config`` is provided.
    land_config : BucketLandConfig or None
        Bucket land model parameters.  When provided together with
        ``surface_properties``, enables the land surface model.
    spinup_days : int
        Days to discard during the prescribed-SST spinup phase.
    averaging_days : int
        Days to average for Q-flux diagnosis.
    spectral_filter : jnp.ndarray or None
        Spectral filter array.  Built automatically when ``None``.
    surface_albedo : float or None
        Surface albedo for the prescribed-SST spinup.  When ``None``,
        uses ``planet.surface_albedo``.  Set this to match the albedo
        used in the coupled phase for self-consistent Q-flux diagnosis.
    days_per_year : float
        Length of year in days for seasonal forcing.
    on_day : callable or None
        Callback invoked after each coupled simulation day:
        ``on_day(day, state, surface, diags)``.
    verbose : bool
        Print progress messages.
    log_interval : int
        Print status every ``log_interval`` days.

    Returns
    -------
    SimulationResult
        Final state, surface state, wall time, and collected diagnostics.
    """
    if ocean_config is None:
        ocean_config = SlabOceanConfig()

    if spectral_filter is None:
        spectral_filter = exponential_filter(transform.arrays, dt)

    # --- Phase 1: Prescribed-SST spinup for Q-flux diagnosis ---
    if verbose:
        total_spinup = spinup_days + averaging_days
        logger.info(
            "Prescribed-SST spinup: %d days (%d spinup + %d averaging)",
            total_spinup,
            spinup_days,
            averaging_days,
        )

    spinup_result = spinup_prescribed_sst(
        initial_state,
        forcing,
        transform,
        planet,
        levels,
        reference_temperature,
        surface_geopotential,
        dt,
        spinup_days=spinup_days,
        averaging_days=averaging_days,
        spectral_filter=spectral_filter,
        surface_albedo=surface_albedo,
        verbose=verbose,
    )

    # --- Phase 2: Initialize surface state ---
    sst = forcing.prescribed_sst
    ocean = OceanState(surface_temperature=sst)

    land = None
    if land_config is not None and surface_properties is not None:
        land = init_land_state(surface_properties.land_fraction, sst, land_config)

    surface = SurfaceState(ocean=ocean, land=land)

    # --- Phase 3: Build coupled stepper and run ---
    init_fn, step_fn = build_coupled_pe_stepper(
        transform=transform,
        planet=planet,
        levels=levels,
        reference_temperature=reference_temperature,
        surface_geopotential=surface_geopotential,
        dt=dt,
        forcing=forcing,
        ocean_config=ocean_config,
        q_flux=spinup_result.q_flux,
        surface_properties=surface_properties,
        land_config=land_config,
        spectral_filter=spectral_filter,
    )

    if verbose:
        logger.info("Starting coupled integration: %d days", n_days)

    return run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=spinup_result.state,
        dt=dt,
        n_days=n_days,
        surface=surface,
        forcing=forcing,
        days_per_year=days_per_year,
        on_day=on_day,
        verbose=verbose,
        log_interval=log_interval,
    )
