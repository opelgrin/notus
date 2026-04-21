from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.physics.forcing import PhysicsDiagnostics
from notus.physics.surface import OceanState, SeaIceConfig, SurfaceState
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


def test_run_coupled_simulation_wires_sea_ice(monkeypatch: pytest.MonkeyPatch) -> None:
    import notus.runner as runner_mod

    initial_state = _state(0.0)
    forcing = SimpleNamespace(prescribed_sst=jnp.array([270.0, 275.0]))
    captured: dict[str, object] = {}

    def fake_spinup_prescribed_sst(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(state=initial_state, q_flux=jnp.zeros_like(forcing.prescribed_sst))

    def fake_build_coupled_pe_stepper(*args: object, **kwargs: object) -> tuple[object, object]:
        captured["ice_config"] = kwargs.get("ice_config")

        def init_fn(
            state: PrimitiveEquationState,
            surface: SurfaceState,
            _day_of_year: jnp.ndarray,
        ) -> tuple[
            PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics
        ]:
            return state, state, surface, PhysicsDiagnostics()

        def step_fn(
            previous: PrimitiveEquationState,
            current: PrimitiveEquationState,
            surface: SurfaceState,
            _day_of_year: jnp.ndarray,
        ) -> tuple[
            PrimitiveEquationState, PrimitiveEquationState, SurfaceState, PhysicsDiagnostics
        ]:
            return previous, current, surface, PhysicsDiagnostics()

        return init_fn, step_fn

    def fake_run_simulation(*args: object, **kwargs: object) -> runner_mod.SimulationResult:
        surface = kwargs["surface"]
        captured["surface"] = surface
        return runner_mod.SimulationResult(
            state=initial_state,
            previous=initial_state,
            surface=surface,
            n_days=1,
            wall_time=0.0,
            diagnostics=[],
        )

    monkeypatch.setattr(runner_mod, "spinup_prescribed_sst", fake_spinup_prescribed_sst)
    monkeypatch.setattr(runner_mod, "build_coupled_pe_stepper", fake_build_coupled_pe_stepper)
    monkeypatch.setattr(runner_mod, "run_simulation", fake_run_simulation)

    ice_cfg = SeaIceConfig()
    _ = runner_mod.run_coupled_simulation(
        initial_state=initial_state,
        forcing=forcing,  # type: ignore[arg-type]
        transform=SimpleNamespace(arrays=None),  # type: ignore[arg-type]
        planet=SimpleNamespace(surface_albedo=0.06),  # type: ignore[arg-type]
        levels=SimpleNamespace(),  # type: ignore[arg-type]
        reference_temperature=np.array([250.0]),
        surface_geopotential=jnp.zeros((1,)),
        dt=86400.0,
        n_days=1,
        spectral_filter=jnp.array([1.0]),
        ice_config=ice_cfg,
        verbose=False,
    )

    assert captured["ice_config"] is ice_cfg
    surface = captured["surface"]
    assert isinstance(surface, SurfaceState)
    assert surface.ice is not None
    np.testing.assert_allclose(np.asarray(surface.ice.ice_thickness), np.array([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(surface.ice.ice_fraction), np.array([1.0, 0.0]))
