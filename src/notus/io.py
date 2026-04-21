"""Restart file I/O for saving and loading model state.

Supports saving and loading atmospheric state (spectral coefficients)
and optionally ocean/sea-ice surface state for warm-starting coupled
integrations.

Examples
--------
Save and reload a dry atmospheric state::

    from notus.io import save_restart, load_restart

    save_restart("restart.npz", state)
    state, ocean_sst = load_restart("restart.npz")

Save a coupled state with ocean SST::

    save_restart("restart.npz", state, ocean_sst=surface.ocean.surface_temperature)

Save a coupled state with ocean SST and sea ice::

    save_restart(
        "restart.npz",
        state,
        ocean_sst=surface.ocean.surface_temperature,
        ice_thickness=surface.ice.ice_thickness,
        ice_fraction=surface.ice.ice_fraction,
    )
"""

from __future__ import annotations

from typing import Literal, overload

import jax.numpy as jnp
import numpy as np

from notus.state import PrimitiveEquationState


def save_restart(
    path: str,
    state: PrimitiveEquationState,
    *,
    ocean_sst: jnp.ndarray | np.ndarray | None = None,
    ice_thickness: jnp.ndarray | np.ndarray | None = None,
    ice_fraction: jnp.ndarray | np.ndarray | None = None,
) -> None:
    """Save atmospheric (and optionally ocean/sea-ice) state to a restart file.

    The restart file is a ``.npz`` archive containing the spectral
    coefficients for all prognostic fields.  Coupled runs should save
    ``ocean_sst`` so the slab ocean can be warm-started too.

    Parameters
    ----------
    path : str
        Output file path (should end in ``.npz``).
    state : PrimitiveEquationState
        Atmospheric state in spectral space.
    ocean_sst : jnp.ndarray or None
        Ocean surface temperature [K] to include in the restart.
    ice_thickness : jnp.ndarray or None
        Sea-ice thickness [m] to include in the restart.
    ice_fraction : jnp.ndarray or None
        Sea-ice fraction [0, 1] to include in the restart.
    """
    if (ice_thickness is None) != (ice_fraction is None):
        msg = "ice_thickness and ice_fraction must be provided together"
        raise ValueError(msg)

    data: dict[str, np.ndarray] = {
        "vorticity": np.asarray(state.vorticity),
        "divergence": np.asarray(state.divergence),
        "temperature": np.asarray(state.temperature),
        "log_surface_pressure": np.asarray(state.log_surface_pressure),
    }
    if state.humidity is not None:
        data["humidity"] = np.asarray(state.humidity)
    if ocean_sst is not None:
        data["ocean_sst"] = np.asarray(ocean_sst)
    if ice_thickness is not None and ice_fraction is not None:
        data["ice_thickness"] = np.asarray(ice_thickness)
        data["ice_fraction"] = np.asarray(ice_fraction)
    # numpy stubs type **kwargs as ndarray but reject dict[str, ndarray] expansion
    np.savez(path, **data)  # type: ignore[arg-type]


@overload
def load_restart(
    path: str,
    *,
    include_sea_ice: Literal[False] = False,
) -> tuple[PrimitiveEquationState, jnp.ndarray | None]: ...


@overload
def load_restart(
    path: str,
    *,
    include_sea_ice: Literal[True],
) -> tuple[
    PrimitiveEquationState,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
]: ...


def load_restart(
    path: str,
    *,
    include_sea_ice: bool = False,
) -> (
    tuple[PrimitiveEquationState, jnp.ndarray | None]
    | tuple[
        PrimitiveEquationState,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
    ]
):
    """Load atmospheric (and optionally ocean/sea-ice) state from a restart file.

    Parameters
    ----------
    path : str
        Path to the ``.npz`` restart file.

    Returns
    -------
    state : PrimitiveEquationState
        Atmospheric state in spectral space.
    ocean_sst : jnp.ndarray or None
        Ocean surface temperature [K], or ``None`` if not saved.
    ice_thickness : jnp.ndarray or None
        Sea-ice thickness [m], returned only when ``include_sea_ice=True``.
    ice_fraction : jnp.ndarray or None
        Sea-ice fraction [0, 1], returned only when ``include_sea_ice=True``.
    """
    data = np.load(path)
    humidity = jnp.array(data["humidity"]) if "humidity" in data else None
    state = PrimitiveEquationState(
        vorticity=jnp.array(data["vorticity"]),
        divergence=jnp.array(data["divergence"]),
        temperature=jnp.array(data["temperature"]),
        log_surface_pressure=jnp.array(data["log_surface_pressure"]),
        humidity=humidity,
    )
    ocean_sst = jnp.array(data["ocean_sst"]) if "ocean_sst" in data else None
    if not include_sea_ice:
        return state, ocean_sst

    ice_thickness = jnp.array(data["ice_thickness"]) if "ice_thickness" in data else None
    ice_fraction = jnp.array(data["ice_fraction"]) if "ice_fraction" in data else None
    return state, ocean_sst, ice_thickness, ice_fraction
