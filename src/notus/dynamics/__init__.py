"""Dynamical core: tendency computations for the spectral-transform GCM."""

from notus.dynamics.shallow_water import (
    shallow_water_tendencies,
    shallow_water_tendencies_eager,
)


__all__ = ["shallow_water_tendencies", "shallow_water_tendencies_eager"]
