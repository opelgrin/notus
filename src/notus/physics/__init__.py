"""Physics parameterizations (Held-Suarez forcing, radiation, etc.)."""

from notus.physics.forcing import Forcing, HeldSuarez
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig


__all__ = ["Forcing", "HeldSuarez", "SimplePhysics", "SimplePhysicsConfig"]
