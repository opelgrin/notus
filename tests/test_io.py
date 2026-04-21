from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from notus.io import load_restart, save_restart
from notus.state import PrimitiveEquationState


def _state(with_humidity: bool = True) -> PrimitiveEquationState:
    humidity = jnp.full((2, 3), 0.005) if with_humidity else None
    return PrimitiveEquationState(
        vorticity=jnp.zeros((2, 3)),
        divergence=jnp.ones((2, 3)),
        temperature=jnp.full((2, 3), 270.0),
        log_surface_pressure=jnp.zeros((3,)),
        humidity=humidity,
    )


def test_roundtrip_atmosphere_only(tmp_path: Path) -> None:
    path = tmp_path / "restart_atm.npz"
    state = _state()

    save_restart(str(path), state)
    loaded_state, ocean_sst = load_restart(str(path))

    np.testing.assert_allclose(np.asarray(loaded_state.vorticity), np.asarray(state.vorticity))
    np.testing.assert_allclose(np.asarray(loaded_state.divergence), np.asarray(state.divergence))
    np.testing.assert_allclose(np.asarray(loaded_state.temperature), np.asarray(state.temperature))
    np.testing.assert_allclose(
        np.asarray(loaded_state.log_surface_pressure),
        np.asarray(state.log_surface_pressure),
    )
    assert loaded_state.humidity is not None
    assert state.humidity is not None
    np.testing.assert_allclose(np.asarray(loaded_state.humidity), np.asarray(state.humidity))
    assert ocean_sst is None


def test_roundtrip_with_ocean_sst(tmp_path: Path) -> None:
    path = tmp_path / "restart_ocean.npz"
    state = _state(with_humidity=False)
    ocean_sst = jnp.array([271.35, 275.0, 280.0])

    save_restart(str(path), state, ocean_sst=ocean_sst)
    loaded_state, loaded_sst = load_restart(str(path))

    assert loaded_state.humidity is None
    assert loaded_sst is not None
    np.testing.assert_allclose(np.asarray(loaded_sst), np.asarray(ocean_sst))


def test_roundtrip_with_sea_ice(tmp_path: Path) -> None:
    path = tmp_path / "restart_ice.npz"
    state = _state()
    ocean_sst = jnp.array([271.35, 274.0, 279.0])
    ice_thickness = jnp.array([1.0, 0.3, 0.0])
    ice_fraction = jnp.array([1.0, 0.3, 0.0])

    save_restart(
        str(path),
        state,
        ocean_sst=ocean_sst,
        ice_thickness=ice_thickness,
        ice_fraction=ice_fraction,
    )
    _state2, sst2, h2, f2 = load_restart(str(path), include_sea_ice=True)

    assert sst2 is not None
    assert h2 is not None
    assert f2 is not None
    np.testing.assert_allclose(np.asarray(sst2), np.asarray(ocean_sst))
    np.testing.assert_allclose(np.asarray(h2), np.asarray(ice_thickness))
    np.testing.assert_allclose(np.asarray(f2), np.asarray(ice_fraction))


def test_include_sea_ice_backward_compatible_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "restart_no_ice.npz"
    state = _state()
    save_restart(str(path), state, ocean_sst=jnp.array([280.0, 281.0, 282.0]))

    _state2, _sst2, h2, f2 = load_restart(str(path), include_sea_ice=True)
    assert h2 is None
    assert f2 is None


def test_save_restart_rejects_partial_ice_inputs(tmp_path: Path) -> None:
    path = tmp_path / "restart_bad.npz"
    state = _state()

    with pytest.raises(ValueError, match="must be provided together"):
        save_restart(str(path), state, ice_thickness=jnp.array([1.0, 0.0, 0.0]))
