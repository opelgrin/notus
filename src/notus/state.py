"""Immutable model state containers.

States are frozen dataclasses registered as JAX pytrees, so they can be
passed through ``jax.jit``, ``jax.vmap``, etc.  Use ``.replace()`` to
create modified copies (functional update pattern).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, slots=True)
class ShallowWaterState:
    """State for the shallow water equations on the sphere.

    All fields are in spectral space (complex, lower-triangular).

    Attributes
    ----------
    vorticity : jnp.ndarray
        Relative vorticity ζ, shape ``(n_spectral,)``.
    divergence : jnp.ndarray
        Divergence δ, shape ``(n_spectral,)``.
    geopotential : jnp.ndarray
        Geopotential Φ = g·h, shape ``(n_spectral,)``.
    """

    vorticity: jnp.ndarray
    divergence: jnp.ndarray
    geopotential: jnp.ndarray

    def replace(self, **kwargs: jnp.ndarray) -> ShallowWaterState:
        """Return a new state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True, slots=True)
class PrimitiveEquationState:
    """State for the hydrostatic primitive equations.

    Horizontal fields (vorticity, divergence, temperature) are 2D arrays
    with shape ``(n_levels, n_spectral)`` — one spectral field per sigma level.

    Surface pressure is a single spectral field.

    Attributes
    ----------
    vorticity : jnp.ndarray
        Relative vorticity ζ, shape ``(n_levels, n_spectral)``.
    divergence : jnp.ndarray
        Divergence δ, shape ``(n_levels, n_spectral)``.
    temperature : jnp.ndarray
        Absolute temperature T, shape ``(n_levels, n_spectral)``.
    log_surface_pressure : jnp.ndarray
        ln(pₛ/p₀), shape ``(n_spectral,)``.
    """

    vorticity: jnp.ndarray
    divergence: jnp.ndarray
    temperature: jnp.ndarray
    log_surface_pressure: jnp.ndarray

    @property
    def n_levels(self) -> int:
        """Number of vertical sigma levels."""
        return self.vorticity.shape[0]

    def replace(self, **kwargs: jnp.ndarray) -> PrimitiveEquationState:
        """Return a new state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


# Register as JAX pytrees so they work with jit/vmap/grad.
def _sw_flatten(state: ShallowWaterState) -> tuple[tuple[jnp.ndarray, ...], None]:
    return (state.vorticity, state.divergence, state.geopotential), None


def _sw_unflatten(_aux: None, children: tuple[jnp.ndarray, ...]) -> ShallowWaterState:
    return ShallowWaterState(*children)


def _pe_flatten(
    state: PrimitiveEquationState,
) -> tuple[tuple[jnp.ndarray, ...], None]:
    return (
        state.vorticity,
        state.divergence,
        state.temperature,
        state.log_surface_pressure,
    ), None


def _pe_unflatten(_aux: None, children: tuple[jnp.ndarray, ...]) -> PrimitiveEquationState:
    return PrimitiveEquationState(*children)


jax.tree_util.register_pytree_node(ShallowWaterState, _sw_flatten, _sw_unflatten)
jax.tree_util.register_pytree_node(PrimitiveEquationState, _pe_flatten, _pe_unflatten)
