"""Prescribed surface temperature profiles for aquaplanet experiments."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, slots=True)
class PrescribedSST:
    """Frierson et al. (2006) aquaplanet SST profile.

    The sea-surface temperature varies only with latitude::

        T_s(φ) = T_max − ΔT · sin²φ

    Parameters
    ----------
    t_max : float
        Maximum (equatorial) SST [K].
    delta_t : float
        Equator-to-pole SST difference [K].
    """

    t_max: float = 285.0
    delta_t: float = 40.0


def compute_sst(config: PrescribedSST, sin_lat: jnp.ndarray) -> jnp.ndarray:
    """Compute SST at each latitude.

    Parameters
    ----------
    config : PrescribedSST
        SST profile parameters.
    sin_lat : jnp.ndarray
        Sine of latitude, shape ``(n_lat,)``.

    Returns
    -------
    jnp.ndarray
        SST in Kelvin, shape ``(n_lat,)``. Zonally uniform.
    """
    return config.t_max - config.delta_t * sin_lat**2
