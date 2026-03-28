"""Planetary constants and configuration.

All physical constants are encapsulated in a `PlanetaryConstants` dataclass
so the model is not hardcoded to Earth. Downstream code receives constants
via dependency injection, never via module-level globals.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True, slots=True)
class PlanetaryConstants:
    """Physical constants for a rotating planet with an ideal-gas atmosphere.

    Parameters
    ----------
    name : str
        Human-readable label.
    radius : float
        Mean radius [m].
    rotation_rate : float
        Angular velocity of rotation [rad/s].
    gravity : float
        Surface gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air, R_d [J/(kg·K)].
    specific_heat_cp : float
        Specific heat at constant pressure, c_p [J/(kg·K)].
    reference_pressure : float
        Reference surface pressure, p₀ [Pa].
    solar_constant : float
        Total solar irradiance, S₀ [W/m²].
    surface_albedo : float
        Surface albedo (dimensionless, 0-1).  For aquaplanet
        configurations this should be the ocean surface albedo (~0.06),
        not the planetary Bond albedo.
    """

    name: str
    radius: float
    rotation_rate: float
    gravity: float
    gas_constant: float
    specific_heat_cp: float
    reference_pressure: float = 1.0e5
    solar_constant: float = 1360.0
    surface_albedo: float = 0.06
    latent_heat_vaporization: float = 2.5e6
    gas_constant_vapor: float = 461.5
    von_karman: float = 0.4

    @property
    def epsilon_moisture(self) -> float:
        """Ratio R_d / R_v, used in saturation specific humidity."""
        return self.gas_constant / self.gas_constant_vapor

    @property
    def kappa(self) -> float:
        """Ratio R/c_p, used in potential temperature and Exner function."""
        return self.gas_constant / self.specific_heat_cp

    @property
    def specific_heat_cv(self) -> float:
        """Specific heat at constant volume, c_v = c_p - R [J/(kg·K)]."""
        return self.specific_heat_cp - self.gas_constant

    @property
    def angular_frequency(self) -> float:
        """Alias for rotation_rate (Ω) [rad/s]."""
        return self.rotation_rate

    @property
    def day_length(self) -> float:
        """Sidereal day length [s]."""
        return 2.0 * math.pi / self.rotation_rate


EARTH = PlanetaryConstants(
    name="Earth",
    radius=6.371e6,
    rotation_rate=7.292e-5,
    gravity=9.80616,
    gas_constant=287.04,
    specific_heat_cp=1004.64,
    reference_pressure=1.0e5,
)
