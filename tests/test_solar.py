"""Tests for solar geometry (orbital parameters, daily-mean insolation)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from notus.physics.solar import (
    EARTH_ORBIT,
    OrbitalParameters,
    daily_mean_insolation,
    earth_sun_distance_factor,
    solar_declination,
)


jax.config.update("jax_enable_x64", True)

SOLAR_CONSTANT = 1360.0


class TestSolarDeclination:
    """Verify solar declination angle."""

    def test_range(self) -> None:
        """Declination should stay within [-obliquity, +obliquity]."""
        days = jnp.linspace(0, 365.25, 366)
        decl = jax.vmap(lambda d: solar_declination(d, EARTH_ORBIT))(days)
        assert float(jnp.max(jnp.abs(decl))) <= EARTH_ORBIT.obliquity + 0.01

    def test_zero_obliquity_gives_zero(self) -> None:
        """With zero obliquity, declination should always be zero."""
        orbital = OrbitalParameters(obliquity=0.0)
        days = jnp.linspace(0, 365, 100)
        decl = jax.vmap(lambda d: solar_declination(d, orbital))(days)
        np.testing.assert_allclose(decl, 0.0, atol=1e-15)

    def test_solstice_magnitude(self) -> None:
        """Declination magnitude should reach obliquity at solstices."""
        days = jnp.linspace(0, 365.25, 3660)
        decl = jax.vmap(lambda d: solar_declination(d, EARTH_ORBIT))(days)
        max_decl = float(jnp.max(jnp.abs(decl)))
        np.testing.assert_allclose(max_decl, EARTH_ORBIT.obliquity, atol=0.01)


class TestEarthSunDistance:
    """Verify Earth-Sun distance factor."""

    def test_circular_orbit_is_unity(self) -> None:
        """With zero eccentricity, distance factor should be 1.0."""
        orbital = OrbitalParameters(eccentricity=0.0)
        days = jnp.linspace(0, 365, 100)
        dist = jax.vmap(lambda d: earth_sun_distance_factor(d, orbital))(days)
        np.testing.assert_allclose(dist, 1.0, atol=1e-15)

    def test_annual_mean_near_unity(self) -> None:
        """Annual mean of (a/r)^2 should be close to 1."""
        days = jnp.linspace(0, 365.25, 3660)
        dist = jax.vmap(lambda d: earth_sun_distance_factor(d, EARTH_ORBIT))(days)
        np.testing.assert_allclose(float(jnp.mean(dist)), 1.0, atol=0.01)

    def test_variation_magnitude(self) -> None:
        """Range should be approximately 1 ± 2e for small eccentricity."""
        days = jnp.linspace(0, 365.25, 3660)
        dist = jax.vmap(lambda d: earth_sun_distance_factor(d, EARTH_ORBIT))(days)
        e = EARTH_ORBIT.eccentricity
        assert float(jnp.max(dist)) > 1.0 + e
        assert float(jnp.min(dist)) < 1.0 - e


class TestDailyMeanInsolation:
    """Verify daily-mean insolation calculation."""

    def test_nonnegative(self) -> None:
        """Insolation should be non-negative everywhere."""
        sin_lat = jnp.linspace(-1.0, 1.0, 64)
        days = jnp.linspace(0, 365, 365)
        for d in days:
            insol = daily_mean_insolation(sin_lat, d, SOLAR_CONSTANT, EARTH_ORBIT)
            assert bool(jnp.all(insol >= 0.0))

    def test_equinox_symmetric(self) -> None:
        """At equinox, insolation should be symmetric about the equator."""
        sin_lat = jnp.linspace(-0.9, 0.9, 32)
        # Find equinox: declination ≈ 0
        insol = daily_mean_insolation(sin_lat, jnp.float64(0.0), SOLAR_CONSTANT, EARTH_ORBIT)
        # Compare NH and SH (reversed)
        np.testing.assert_allclose(
            np.asarray(insol),
            np.asarray(insol[::-1]),
            atol=20.0,  # some asymmetry from eccentricity
        )

    def test_polar_night(self) -> None:
        """Winter pole should have zero insolation."""
        # NH winter solstice: declination at most negative
        # Find the day with most negative declination
        days = jnp.linspace(0, 365.25, 3660)
        decls = jax.vmap(lambda d: solar_declination(d, EARTH_ORBIT))(days)
        winter_day = days[jnp.argmin(decls)]

        sin_lat = jnp.array([1.0])  # North pole
        insol = daily_mean_insolation(sin_lat, winter_day, SOLAR_CONSTANT, EARTH_ORBIT)
        np.testing.assert_allclose(float(insol[0]), 0.0, atol=1e-10)

    def test_polar_day(self) -> None:
        """Summer pole should have positive insolation."""
        days = jnp.linspace(0, 365.25, 3660)
        decls = jax.vmap(lambda d: solar_declination(d, EARTH_ORBIT))(days)
        summer_day = days[jnp.argmax(decls)]

        sin_lat = jnp.array([1.0])  # North pole
        insol = daily_mean_insolation(sin_lat, summer_day, SOLAR_CONSTANT, EARTH_ORBIT)
        assert float(insol[0]) > 0.0

    def test_annual_global_mean(self) -> None:
        """Annual and global mean insolation should approximate S0/4."""
        # Use uniform sin(lat) grid — d(sinφ) is the area element,
        # so the global mean is the unweighted mean over sin(lat).
        sin_lat = jnp.linspace(-1.0, 1.0, 128)
        days = jnp.linspace(0, 365.25, 366)

        total = jnp.zeros(len(sin_lat))
        for d in days:
            total += daily_mean_insolation(sin_lat, d, SOLAR_CONSTANT, EARTH_ORBIT)
        annual_mean = total / len(days)

        global_mean = float(jnp.mean(annual_mean))
        np.testing.assert_allclose(global_mean, SOLAR_CONSTANT / 4.0, rtol=0.02)

    def test_zero_obliquity_no_seasonal_variation(self) -> None:
        """With zero obliquity, insolation should not vary with day."""
        orbital = OrbitalParameters(obliquity=0.0, eccentricity=0.0)
        sin_lat = jnp.array([0.0, 0.5, -0.5])

        insol_d0 = daily_mean_insolation(sin_lat, jnp.float64(0.0), SOLAR_CONSTANT, orbital)
        insol_d180 = daily_mean_insolation(sin_lat, jnp.float64(182.0), SOLAR_CONSTANT, orbital)
        np.testing.assert_allclose(insol_d0, insol_d180, rtol=1e-10)

    def test_jit_compatible(self) -> None:
        """Solar geometry functions should be JIT-compilable."""
        sin_lat = jnp.linspace(-0.9, 0.9, 16)

        @jax.jit
        def compute(day):
            return daily_mean_insolation(sin_lat, day, SOLAR_CONSTANT, EARTH_ORBIT)

        result = compute(jnp.float64(100.0))
        assert result.shape == (16,)
        assert bool(jnp.all(jnp.isfinite(result)))
