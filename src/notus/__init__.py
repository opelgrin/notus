"""Notus — A spectral-transform General Circulation Model in JAX.

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

from notus.constants import EARTH, PlanetaryConstants
from notus.grid import GaussianGrid
from notus.operators import (
    hyperdiffusion,
    inverse_laplacian,
    laplacian,
    meridional_derivative,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
    zonal_derivative,
)
from notus.sigma import SigmaLevels, standard_sigma_levels, uniform_sigma_levels
from notus.state import PrimitiveEquationState, ShallowWaterState
from notus.transforms import SpectralTransform
from notus.vertical import (
    geopotential,
    omega_over_pressure,
    sigma_dot,
    surface_pressure_tendency,
    vertical_advection,
)


__all__ = [
    "EARTH",
    "GaussianGrid",
    "PlanetaryConstants",
    "PrimitiveEquationState",
    "ShallowWaterState",
    "SigmaLevels",
    "SpectralTransform",
    "geopotential",
    "hyperdiffusion",
    "inverse_laplacian",
    "laplacian",
    "meridional_derivative",
    "omega_over_pressure",
    "sigma_dot",
    "spectral_curl",
    "spectral_divergence",
    "standard_sigma_levels",
    "surface_pressure_tendency",
    "uniform_sigma_levels",
    "uv_from_vordiv",
    "vertical_advection",
    "zonal_derivative",
]
