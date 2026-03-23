"""Dynamical core: tendency computations for the spectral-transform GCM."""

from notus.dynamics.primitive_equations import primitive_equation_tendencies
from notus.dynamics.shallow_water import (
    shallow_water_tendencies,
    shallow_water_tendencies_eager,
)


__all__ = [
    "primitive_equation_tendencies",
    "shallow_water_tendencies",
    "shallow_water_tendencies_eager",
]
