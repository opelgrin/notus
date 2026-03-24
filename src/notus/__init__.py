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
from notus.diagnostics import (
    ConservationDiagnostics,
    ZonalMeanState,
    compute_conservation_diagnostics,
    compute_zonal_mean_state,
    sigma_integral,
    spherical_integral,
)
from notus.grid import GaussianGrid
from notus.initial_conditions import (
    JWConfig,
    held_suarez_initial_state,
    jablonowski_williamson_perturbation,
    jablonowski_williamson_steady_state,
    moist_aquaplanet_initial_state,
    simple_physics_initial_state,
)
from notus.operators import (
    OperatorArrays,
    hyperdiffusion,
    inverse_laplacian,
    laplacian,
    meridional_derivative,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
    zonal_derivative,
)
from notus.physics import HeldSuarez, SimplePhysics, SimplePhysicsConfig
from notus.state import PrimitiveEquationState, ShallowWaterState
from notus.transforms import SpectralTransform
from notus.vertical import (
    geopotential,
    omega_over_pressure,
    sigma_dot,
    surface_pressure_tendency,
    vertical_advection,
)
from notus.vertical.sigma import SigmaLevels, standard_sigma_levels, uniform_sigma_levels


__all__ = [
    "EARTH",
    "ConservationDiagnostics",
    "GaussianGrid",
    "HeldSuarez",
    "JWConfig",
    "OperatorArrays",
    "PlanetaryConstants",
    "PrimitiveEquationState",
    "ShallowWaterState",
    "SigmaLevels",
    "SimplePhysics",
    "SimplePhysicsConfig",
    "SpectralTransform",
    "ZonalMeanState",
    "compute_conservation_diagnostics",
    "compute_zonal_mean_state",
    "geopotential",
    "held_suarez_initial_state",
    "hyperdiffusion",
    "inverse_laplacian",
    "jablonowski_williamson_perturbation",
    "jablonowski_williamson_steady_state",
    "laplacian",
    "meridional_derivative",
    "moist_aquaplanet_initial_state",
    "omega_over_pressure",
    "sigma_dot",
    "sigma_integral",
    "simple_physics_initial_state",
    "spectral_curl",
    "spectral_divergence",
    "spherical_integral",
    "standard_sigma_levels",
    "surface_pressure_tendency",
    "uniform_sigma_levels",
    "uv_from_vordiv",
    "vertical_advection",
    "zonal_derivative",
]
