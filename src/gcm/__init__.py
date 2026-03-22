"""GCM — A spectral-transform General Circulation Model in JAX.

This package provides a from-scratch implementation of a spectral-transform
atmospheric model, suitable for idealized simulations (e.g. Held-Suarez)
on any rotating planet with an ideal-gas atmosphere.

Core components
---------------
- Grid: :class:`GaussianGrid`
- Transforms: :class:`SpectralTransform`
- Constants: :class:`PlanetaryConstants`, :data:`EARTH`
- Operators: :func:`laplacian`, :func:`inverse_laplacian`, etc.
- State: :class:`ShallowWaterState`, :class:`PrimitiveEquationState`
"""

from gcm.constants import EARTH, PlanetaryConstants
from gcm.grid import GaussianGrid
from gcm.operators import (
    hyperdiffusion,
    inverse_laplacian,
    laplacian,
    zonal_derivative,
)
from gcm.state import PrimitiveEquationState, ShallowWaterState
from gcm.transforms import SpectralTransform


__all__ = [
    "EARTH",
    "GaussianGrid",
    "PlanetaryConstants",
    "PrimitiveEquationState",
    "ShallowWaterState",
    "SpectralTransform",
    "hyperdiffusion",
    "inverse_laplacian",
    "laplacian",
    "zonal_derivative",
]
