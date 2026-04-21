from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from notus.physics.forcing import PhysicsDiagnostics
from notus.physics.surface import OceanState, SurfaceState
from notus.runner import _build_coupled_one_day, run_simulation
from notus.state import PrimitiveEquationState


jax.config.update("jax_enable_x64", True)


def _state(temp: float) -> PrimitiveEquationState:
    return PrimitiveEquationState(
        vorticity=jnp.zeros((1, 1)),
        divergence=jnp.zeros((1, 1)),
        temperature=jnp.full((1, 1), temp),
        log_surface_pressure=jnp.zeros((1,)),
    )


def _surface() -> SurfaceState:
    return SurfaceState(ocean=OceanState(surface_temperature=jnp.array([280.0])))


def test_build_coupled_one_day_uses_dynamic_day_of_year() -> None:
    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
        surface: SurfaceState,
        day_of_year: jnp.ndarray,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics]:
        updated = current.replace(temperature=current.temperature + day_of_year)
        return previous, updated, surface, PhysicsDiagnostics()

    one_day = _build_coupled_one_day(step_fn, steps_per_day=1)
    carry = (_state(0.0), _state(0.0), _surface(), PhysicsDiagnostics())

    (_, day0_state, _, _), _ = one_day(carry, jnp.float64(0.0))
    (_, day90_state, _, _), _ = one_day(carry, jnp.float64(90.0))

    np.testing.assert_allclose(np.asarray(day0_state.temperature), np.array([[0.0]]))
    np.testing.assert_allclose(np.asarray(day90_state.temperature), np.array([[90.0]]))


class _NoMutationForcing:
    def __init__(self) -> None:
        self.prescribed_sst = jnp.array([280.0])

    @property
    def day_of_year(self) -> jnp.ndarray | None:
        return None

    @day_of_year.setter
    def day_of_year(self, _value: jnp.ndarray) -> None:
        raise AssertionError("run_simulation must not mutate forcing.day_of_year in coupled mode")


def test_run_simulation_coupled_does_not_mutate_forcing_day_of_year() -> None:
    forcing = _NoMutationForcing()
    initial_state = _state(0.0)
    surface = _surface()
    samples: list[tuple[int, float]] = []

    def init_fn(
        state: PrimitiveEquationState,
        sfc: SurfaceState,
        day_of_year: jnp.ndarray,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics]:
        np.testing.assert_allclose(float(day_of_year), 0.0)
        return state, state, sfc, PhysicsDiagnostics()

    def step_fn(
        previous: PrimitiveEquationState,
        current: PrimitiveEquationState,
        sfc: SurfaceState,
        day_of_year: jnp.ndarray,
    ) -> tuple[PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics]:
        # Encode day_of_year directly in temperature for easy verification.
        stamped = current.replace(temperature=jnp.full_like(current.temperature, day_of_year))
        return previous, stamped, sfc, PhysicsDiagnostics()

    def on_day(
        day: int,
        state: PrimitiveEquationState,
        _surface: SurfaceState,
        _diags: PhysicsDiagnostics,
    ) -> None:
        samples.append((day, float(state.temperature[0, 0])))

    run_simulation(
        init_fn=init_fn,
        step_fn=step_fn,
        initial_state=initial_state,
        dt=86400.0,
        n_days=3,
        surface=surface,
        forcing=forcing,  # type: ignore[arg-type]
        days_per_year=365.0,
        on_day=on_day,
        verbose=False,
    )

    assert samples == [(1, 0.0), (2, 2.0), (3, 3.0)]
