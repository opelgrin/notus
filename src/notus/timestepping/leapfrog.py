"""Leapfrog time integration with Robert-Asselin-Williams (RAW) filter.

The leapfrog scheme is a second-order, three-level explicit method:

    x^{n+1} = x^{n-1} + 2·dt · F(x^n)

It has a computational mode (spurious oscillation between even and odd
timesteps) that must be damped by the Robert-Asselin filter.  The Williams
(2009) modification reduces the amplitude error of the original filter.

RAW filter (Williams 2009):

    D = x^{n-1} - 2·x^n + x^{n+1}
    x^n_filtered    = x^n    + (ν·α/2)·D
    x^{n+1}_filtered = x^{n+1} - (ν·(1-α)/2)·D

where ν is the Robert-Asselin coefficient and α is the Williams parameter.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from notus.state import ShallowWaterState


@dataclasses.dataclass(frozen=True, slots=True)
class LeapfrogState:
    """Two-level state for leapfrog timestepping.

    Attributes
    ----------
    current : ShallowWaterState
        State at time t (used for tendency evaluation).
    previous : ShallowWaterState
        State at time t - dt.
    """

    current: ShallowWaterState
    previous: ShallowWaterState

    def replace(self, **kwargs: ShallowWaterState) -> LeapfrogState:
        """Return a new state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


def euler_step(
    state: ShallowWaterState,
    tendency: ShallowWaterState,
    dt: float,
) -> LeapfrogState:
    """Forward Euler step to initialize the leapfrog scheme.

    Computes x^1 = x^0 + dt · F(x^0) and returns a LeapfrogState
    with previous = x^0, current = x^1.

    Parameters
    ----------
    state : ShallowWaterState
        Initial state x^0 (spectral).
    tendency : ShallowWaterState
        Tendencies F(x^0) (spectral).
    dt : float
        Timestep [s].

    Returns
    -------
    LeapfrogState
        Two-level state ready for leapfrog stepping.
    """
    new_state = jax.tree.map(lambda x, f: x + dt * f, state, tendency)
    return LeapfrogState(current=new_state, previous=state)


def leapfrog_step(
    state: LeapfrogState,
    tendency: ShallowWaterState,
    dt: float,
    robert_coeff: float = 0.04,
    williams_coeff: float = 0.53,
) -> LeapfrogState:
    """Advance one leapfrog step with RAW filter.

    Parameters
    ----------
    state : LeapfrogState
        Contains current (t) and previous (t-dt) states.
    tendency : ShallowWaterState
        Tendencies evaluated at time t.
    dt : float
        Timestep [s].
    robert_coeff : float
        Robert-Asselin filter coefficient ν (typically 0.04).
    williams_coeff : float
        Williams modification α (typically 0.53). Setting α=1 recovers
        the original Robert-Asselin filter.

    Returns
    -------
    LeapfrogState
        Updated state: previous = filtered x^n, current = filtered x^{n+1}.
    """

    def _raw_step(
        x_prev: jnp.ndarray,
        x_curr: jnp.ndarray,
        tend: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Leapfrog update
        x_new = x_prev + 2.0 * dt * tend

        # RAW filter
        d = x_prev - 2.0 * x_curr + x_new
        nu_half = robert_coeff / 2.0
        x_curr_filtered = x_curr + nu_half * williams_coeff * d
        x_new_filtered = x_new - nu_half * (1.0 - williams_coeff) * d

        return x_curr_filtered, x_new_filtered

    # Apply to all fields via tree_map
    filtered_curr, filtered_new = jax.tree.map(
        _raw_step,
        state.previous,
        state.current,
        tendency,
    )

    return LeapfrogState(current=filtered_new, previous=filtered_curr)


# Register as JAX pytree
def _lf_flatten(state: LeapfrogState) -> tuple[tuple[ShallowWaterState, ...], None]:
    return (state.current, state.previous), None


def _lf_unflatten(_aux: None, children: tuple[ShallowWaterState, ...]) -> LeapfrogState:
    return LeapfrogState(current=children[0], previous=children[1])


jax.tree_util.register_pytree_node(LeapfrogState, _lf_flatten, _lf_unflatten)
