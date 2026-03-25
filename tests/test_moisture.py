"""Tests for moisture thermodynamics and tracer transport (Phase 6).

Unit tests for saturation vapor pressure, specific humidity, and moist
adiabat.  Integration tests for passive moisture tracer advection in
the primitive equations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import held_suarez_initial_state, moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.physics.convection import (
    betts_miller_convection,
    large_scale_condensation,
)
from notus.physics.forcing import HeldSuarez
from notus.physics.moisture import (
    moist_adiabat,
    saturation_specific_humidity,
    saturation_vapor_pressure,
)
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.surface import surface_latent_heat_flux
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels, uniform_sigma_levels


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def levels() -> SigmaLevels:
    return uniform_sigma_levels(20)


@pytest.fixture(scope="module")
def grid() -> GaussianGrid:
    return GaussianGrid(truncation=21)


@pytest.fixture(scope="module")
def transform(grid: GaussianGrid) -> SpectralTransform:
    return SpectralTransform(grid, EARTH.radius)


# ---------------------------------------------------------------------------
# Saturation vapor pressure (Bolton 1980)
# ---------------------------------------------------------------------------


class TestSaturationVaporPressure:
    def test_freezing_point(self) -> None:
        """e_sat(273.15 K) ≈ 611.2 Pa (Bolton fit at 0 °C)."""
        e = saturation_vapor_pressure(jnp.array(273.15))
        np.testing.assert_allclose(float(e), 611.2, rtol=1e-3)

    def test_boiling_point(self) -> None:
        """e_sat(373.15 K) ≈ 101325 Pa (1 atm at 100 °C).

        Bolton fit is calibrated for -35 to +35 °C so this is an
        extrapolation, but it should still be in the right ballpark.
        """
        e = saturation_vapor_pressure(jnp.array(373.15))
        np.testing.assert_allclose(float(e), 101325.0, rtol=0.15)

    def test_monotonically_increasing(self) -> None:
        """e_sat must increase with temperature."""
        t = jnp.linspace(240.0, 320.0, 100)
        e = saturation_vapor_pressure(t)
        assert jnp.all(jnp.diff(e) > 0)

    def test_cold_stratosphere(self) -> None:
        """At 200 K, e_sat should be very small (< 1 Pa)."""
        e = saturation_vapor_pressure(jnp.array(200.0))
        assert float(e) < 1.0

    def test_warm_tropics(self) -> None:
        """At 300 K, e_sat should be ~3500 Pa (standard tables)."""
        e = saturation_vapor_pressure(jnp.array(300.0))
        np.testing.assert_allclose(float(e), 3530.0, rtol=0.02)


# ---------------------------------------------------------------------------
# Saturation specific humidity
# ---------------------------------------------------------------------------


class TestSaturationSpecificHumidity:
    def test_tropical_surface(self) -> None:
        """q_sat(300 K, 1000 hPa) ≈ 22 g/kg (standard atmosphere)."""
        q = saturation_specific_humidity(jnp.array(300.0), jnp.array(1.0e5), EARTH.epsilon_moisture)
        np.testing.assert_allclose(float(q), 0.022, atol=0.003)

    def test_stratosphere_nearly_zero(self) -> None:
        """q_sat(200 K, 100 hPa) should be very small."""
        q = saturation_specific_humidity(jnp.array(200.0), jnp.array(1.0e4), EARTH.epsilon_moisture)
        assert float(q) < 1e-4  # < 0.1 g/kg

    def test_increases_with_temperature(self) -> None:
        """q_sat increases with T at fixed p."""
        t = jnp.linspace(250.0, 310.0, 50)
        p = jnp.full_like(t, 1.0e5)
        q = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        assert jnp.all(jnp.diff(q) > 0)

    def test_decreases_with_pressure(self) -> None:
        """q_sat decreases with p at fixed T."""
        p = jnp.linspace(5.0e4, 1.0e5, 50)
        t = jnp.full_like(p, 280.0)
        q = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        assert jnp.all(jnp.diff(q) < 0)

    def test_epsilon_relationship(self) -> None:
        """In the low-humidity limit, q_sat ≈ ε e_sat / p."""
        t = jnp.array(230.0)
        p = jnp.array(5.0e4)
        eps = EARTH.epsilon_moisture
        q = saturation_specific_humidity(t, p, eps)
        e_sat = saturation_vapor_pressure(t)
        q_approx = eps * e_sat / p
        np.testing.assert_allclose(float(q), float(q_approx), rtol=0.01)


# ---------------------------------------------------------------------------
# Moist adiabat
# ---------------------------------------------------------------------------


class TestMoistAdiabat:
    def test_surface_temperature_recovered(self) -> None:
        """Temperature at surface pressure should equal t_surface."""
        p_levels = jnp.array([1.0e5])
        t = moist_adiabat(
            300.0,
            p_levels,
            1.0e5,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        np.testing.assert_allclose(float(t[0]), 300.0, atol=0.5)

    def test_temperature_decreases_upward(self) -> None:
        """Moist adiabat must cool with decreasing pressure."""
        p_levels = jnp.linspace(2.0e4, 1.0e5, 20)
        t = moist_adiabat(
            300.0,
            p_levels,
            1.0e5,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        # p_levels ordered small to large, so T should increase
        assert jnp.all(jnp.diff(t) > 0)

    def test_lapse_rate_between_dry_and_moist(self) -> None:
        """Lower-troposphere moist lapse rate should be 4-7 K/km.

        The dry adiabatic lapse rate is ~10 K/km.  The moist adiabat
        should be significantly less, typically 5-6 K/km at tropical
        surface temperatures.
        """
        p_levels = jnp.array([9.0e4, 1.0e5])
        t = moist_adiabat(
            300.0,
            p_levels,
            1.0e5,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        # Approximate height difference: dp = 10000 Pa ≈ 1 km
        dz_approx = 1000.0  # m
        lapse_rate = (float(t[1]) - float(t[0])) / dz_approx * 1000.0
        # K/km, should be ~5-7
        assert 3.0 < lapse_rate < 8.0

    def test_approaches_dry_adiabat_at_cold_temps(self) -> None:
        """At cold temperatures, moist ≈ dry adiabat (little moisture)."""
        p_levels = jnp.array([1.0e4, 2.0e4])
        t = moist_adiabat(
            300.0,
            p_levels,
            1.0e5,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        # At these cold upper-tropospheric temperatures,
        # the lapse rate should approach the dry value (~10 K/km)
        dz_approx = EARTH.gas_constant * float(t.mean()) / EARTH.gravity * jnp.log(2.0e4 / 1.0e4)
        lapse = (float(t[1]) - float(t[0])) / float(dz_approx) * 1000.0
        assert lapse > 7.0  # closer to dry (~10 K/km)


# ---------------------------------------------------------------------------
# State with humidity — pytree operations
# ---------------------------------------------------------------------------


class TestStatePytree:
    def test_dry_state_unchanged(self) -> None:
        """Dry state (no humidity) round-trips through pytree correctly."""
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((5, 10), dtype=jnp.complex128),
            divergence=jnp.zeros((5, 10), dtype=jnp.complex128),
            temperature=jnp.ones((5, 10), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(10, dtype=jnp.complex128),
        )
        leaves, treedef = jax.tree_util.tree_flatten(state)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert not restored.has_humidity
        assert restored.humidity is None

    def test_moist_state_roundtrip(self) -> None:
        """State with humidity round-trips through pytree correctly."""
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((5, 10), dtype=jnp.complex128),
            divergence=jnp.zeros((5, 10), dtype=jnp.complex128),
            temperature=jnp.ones((5, 10), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(10, dtype=jnp.complex128),
            humidity=jnp.ones((5, 10), dtype=jnp.complex128) * 0.01,
        )
        leaves, treedef = jax.tree_util.tree_flatten(state)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.has_humidity
        np.testing.assert_array_equal(restored.humidity, state.humidity)

    def test_tree_map_scales_humidity(self) -> None:
        """jax.tree.map on moist state should include humidity."""
        state = PrimitiveEquationState(
            vorticity=jnp.ones((3, 5), dtype=jnp.complex128),
            divergence=jnp.ones((3, 5), dtype=jnp.complex128),
            temperature=jnp.ones((3, 5), dtype=jnp.complex128),
            log_surface_pressure=jnp.ones(5, dtype=jnp.complex128),
            humidity=jnp.ones((3, 5), dtype=jnp.complex128) * 2.0,
        )
        scaled = jax.tree.map(lambda x: 3.0 * x, state)
        assert scaled.humidity is not None
        np.testing.assert_allclose(scaled.humidity, 6.0)

    def test_tree_map_add_dry_states(self) -> None:
        """Adding dry states via tree_map should produce dry result."""
        s1 = PrimitiveEquationState(
            vorticity=jnp.ones((3, 5), dtype=jnp.complex128),
            divergence=jnp.zeros((3, 5), dtype=jnp.complex128),
            temperature=jnp.ones((3, 5), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(5, dtype=jnp.complex128),
        )
        s2 = jax.tree.map(lambda x: x, s1)
        result = jax.tree.map(jnp.add, s1, s2)
        assert not result.has_humidity


# ---------------------------------------------------------------------------
# Passive tracer transport
# ---------------------------------------------------------------------------


def _make_moist_state(
    transform: SpectralTransform,
    levels: SigmaLevels,
    planet: type[object] = EARTH,
) -> tuple[PrimitiveEquationState, np.ndarray, jnp.ndarray]:
    """Create an initial state with a humidity tracer.

    Starts from the standard Held-Suarez initial condition and adds
    a specific humidity profile: 80% RH in the troposphere, decaying
    above σ = 0.3.
    """
    state, ref_temps, surf_geo = held_suarez_initial_state(transform, EARTH, levels)
    n_levels = levels.n_levels
    n_spec = transform.grid.n_spectral_coeffs

    # Simple q profile: 0.01 kg/kg at surface, decaying with sigma
    sigma_full = np.asarray(levels.sigma_full)
    q_profile = 0.01 * np.maximum(0.0, (sigma_full - 0.1) / 0.9) ** 2

    # Put the zonal-mean profile into spectral mode (0,0) only
    q_spec = jnp.zeros((n_levels, n_spec), dtype=jnp.complex128)
    # Mode (0,0) spectral coefficient = global mean * sqrt(4π)
    # For a constant field, the (0,0) coefficient is value * sqrt(4π)
    sqrt4pi = jnp.sqrt(4.0 * jnp.pi)
    q_spec = q_spec.at[:, 0].set(jnp.array(q_profile, dtype=jnp.complex128) * sqrt4pi)

    moist_state = state.replace(humidity=q_spec)
    return moist_state, ref_temps, surf_geo


class TestPassiveTracerTransport:
    def test_resting_atmosphere_no_change(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Uniform q on a resting isothermal atmosphere → no tendency."""
        n_levels = levels.n_levels
        n_spec = transform.grid.n_spectral_coeffs

        # Resting state: zero winds, uniform temperature
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=jnp
            .full((n_levels, n_spec), 0.0, dtype=jnp.complex128)
            .at[:, 0]
            .set(264.0 * jnp.sqrt(4.0 * jnp.pi)),
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
            humidity=jnp
            .full((n_levels, n_spec), 0.0, dtype=jnp.complex128)
            .at[:, 0]
            .set(0.01 * jnp.sqrt(4.0 * jnp.pi)),
        )

        from notus.dynamics.primitive_equations import (
            primitive_equation_tendencies,
        )

        ref_temps = np.full(n_levels, 264.0)
        surf_geo = jnp.zeros(n_spec, dtype=jnp.complex128)

        tend_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            diffusion_order=0,
        )
        tendency = tend_fn(state)

        # Humidity tendency should be near zero (no winds → no advection)
        assert tendency.humidity is not None
        max_q_tend = float(jnp.max(jnp.abs(tendency.humidity)))
        assert max_q_tend < 1e-15

    def test_short_integration_stability(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """10-step integration with moisture tracer should remain stable."""
        state, ref_temps, surf_geo = _make_moist_state(transform, levels)
        dt = 1200.0
        filt = exponential_filter(transform.arrays, dt)
        forcing = HeldSuarez(transform, EARTH, levels)
        init_fn, step_fn = build_pe_stepper(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            dt=dt,
            spectral_filter=filt,
            forcing=forcing,
        )

        prev, curr = init_fn(state)
        for _ in range(10):
            prev, curr = step_fn(prev, curr)

        # Should still have humidity
        assert curr.humidity is not None
        # Check finite values
        assert jnp.all(jnp.isfinite(curr.humidity))
        # Check humidity didn't blow up
        q_grid = jax.vmap(transform.spectral_to_grid)(curr.humidity)
        assert float(jnp.max(jnp.abs(q_grid))) < 1.0  # < 1 kg/kg

    def test_humidity_has_tendency(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """With active dynamics, humidity should have nonzero tendency."""
        state, ref_temps, surf_geo = _make_moist_state(transform, levels)

        from notus.dynamics.primitive_equations import (
            primitive_equation_tendencies,
        )

        # First evolve the dry fields a bit to get nonzero winds
        dt = 1200.0
        filt = exponential_filter(transform.arrays, dt)
        forcing = HeldSuarez(transform, EARTH, levels)
        init_fn, step_fn = build_pe_stepper(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            dt=dt,
            spectral_filter=filt,
            forcing=forcing,
        )

        prev, curr = init_fn(state)
        for _ in range(5):
            prev, curr = step_fn(prev, curr)

        # Now compute explicit tendency on the evolved state
        tend_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
        )
        tendency = tend_fn(curr)

        assert tendency.humidity is not None
        max_tend = float(jnp.max(jnp.abs(tendency.humidity)))
        assert max_tend > 1e-20  # should be nonzero

    def test_dry_state_backward_compatible(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Dry state (no humidity) still works through the stepper."""
        state, ref_temps, surf_geo = held_suarez_initial_state(transform, EARTH, levels)
        dt = 1200.0
        filt = exponential_filter(transform.arrays, dt)
        forcing = HeldSuarez(transform, EARTH, levels)
        init_fn, step_fn = build_pe_stepper(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            dt=dt,
            spectral_filter=filt,
            forcing=forcing,
        )

        prev, curr = init_fn(state)
        for _ in range(5):
            prev, curr = step_fn(prev, curr)

        assert not curr.has_humidity
        assert curr.humidity is None
        assert jnp.all(jnp.isfinite(curr.temperature))


# ---------------------------------------------------------------------------
# Surface latent heat flux
# ---------------------------------------------------------------------------


class TestSurfaceLatentHeatFlux:
    def test_positive_upward_flux(self) -> None:
        """Flux should be positive (moistening) when surface is moister."""
        n_lat, n_lon = 8, 16
        t_surface = jnp.full(n_lat, 300.0)
        q_air = jnp.full((n_lat, n_lon), 0.010)  # < q_sat(300K)
        wind_speed = jnp.full((n_lat, n_lon), 5.0)
        ps = jnp.full((n_lat, n_lon), 1.0e5)

        flux = surface_latent_heat_flux(
            t_surface,
            q_air,
            wind_speed,
            ps,
            EARTH.gravity,
            EARTH.gas_constant,
            0.05,
            EARTH.epsilon_moisture,
            drag_coefficient=0.0015,
        )
        assert jnp.all(flux > 0)

    def test_zero_flux_at_equilibrium(self) -> None:
        """Flux should vanish when q_air = q_sat(T_s)."""
        n_lat, n_lon = 8, 16
        t_surface = jnp.full(n_lat, 300.0)
        ps = jnp.full((n_lat, n_lon), 1.0e5)

        q_sat = saturation_specific_humidity(
            jnp.array(300.0),
            jnp.array(1.0e5),
            EARTH.epsilon_moisture,
        )
        q_air = jnp.full((n_lat, n_lon), float(q_sat))
        wind_speed = jnp.full((n_lat, n_lon), 5.0)

        flux = surface_latent_heat_flux(
            t_surface,
            q_air,
            wind_speed,
            ps,
            EARTH.gravity,
            EARTH.gas_constant,
            0.05,
            EARTH.epsilon_moisture,
            drag_coefficient=0.0015,
        )
        np.testing.assert_allclose(flux, 0.0, atol=1e-10)

    def test_scales_with_wind_speed(self) -> None:
        """Flux should scale linearly with wind speed."""
        n_lat, n_lon = 4, 8
        t_surface = jnp.full(n_lat, 300.0)
        q_air = jnp.full((n_lat, n_lon), 0.010)
        ps = jnp.full((n_lat, n_lon), 1.0e5)

        flux_5 = surface_latent_heat_flux(
            t_surface,
            q_air,
            jnp.full((n_lat, n_lon), 5.0),
            ps,
            EARTH.gravity,
            EARTH.gas_constant,
            0.05,
            EARTH.epsilon_moisture,
            drag_coefficient=0.0015,
        )
        flux_10 = surface_latent_heat_flux(
            t_surface,
            q_air,
            jnp.full((n_lat, n_lon), 10.0),
            ps,
            EARTH.gravity,
            EARTH.gas_constant,
            0.05,
            EARTH.epsilon_moisture,
            drag_coefficient=0.0015,
        )
        np.testing.assert_allclose(flux_10 / flux_5, 2.0, rtol=0.01)

    def test_tropical_magnitude(self) -> None:
        """Tropical evaporation should be ~100-200 W/m² equivalent.

        E [W/m²] = L * rho * C_D * |v| * (q_sat - q_a)
        With T_s=300K, q_a=15g/kg, |v|=5m/s → E ~ 100-150 W/m²
        """
        n_lat, n_lon = 4, 8
        t_surface = jnp.full(n_lat, 300.0)
        q_air = jnp.full((n_lat, n_lon), 0.015)
        wind_speed = jnp.full((n_lat, n_lon), 5.0)
        ps = jnp.full((n_lat, n_lon), 1.0e5)

        dq_dt = surface_latent_heat_flux(
            t_surface,
            q_air,
            wind_speed,
            ps,
            EARTH.gravity,
            EARTH.gas_constant,
            0.05,
            EARTH.epsilon_moisture,
            drag_coefficient=0.0015,
        )
        # Convert tendency to W/m²: E_flux = dq/dt * dp/g * L
        dp = 0.05 * 1.0e5
        e_wm2 = float(jnp.mean(dq_dt)) * dp / EARTH.gravity * EARTH.latent_heat_vaporization
        assert 50.0 < e_wm2 < 300.0


# ---------------------------------------------------------------------------
# Large-scale condensation
# ---------------------------------------------------------------------------


class TestLargeScaleCondensation:
    def test_supersaturated_removes_excess(self) -> None:
        """Supersaturated column should be brought to saturation."""
        n_levels, n_lat, n_lon = 10, 4, 8
        t = jnp.full((n_levels, n_lat, n_lon), 280.0)
        p = jnp.full((n_levels, n_lat, n_lon), 8.0e4)
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 1.1  # 110% RH

        t_new, q_new, _condensate = large_scale_condensation(
            t,
            q,
            p,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        # q should be at or below saturation
        q_sat_new = saturation_specific_humidity(t_new, p, EARTH.epsilon_moisture)
        excess = q_new - q_sat_new
        assert float(jnp.max(excess)) < 1e-6

    def test_subsaturated_unchanged(self) -> None:
        """Subsaturated column should not be modified."""
        n_levels, n_lat, n_lon = 10, 4, 8
        t = jnp.full((n_levels, n_lat, n_lon), 280.0)
        p = jnp.full((n_levels, n_lat, n_lon), 8.0e4)
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 0.8  # 80% RH

        t_new, q_new, condensate = large_scale_condensation(
            t,
            q,
            p,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        np.testing.assert_allclose(t_new, t, atol=1e-12)
        np.testing.assert_allclose(q_new, q, atol=1e-12)
        np.testing.assert_allclose(condensate, 0.0, atol=1e-12)

    def test_energy_conservation(self) -> None:
        """cp*ΔT + L*Δq = 0 at each grid point."""
        n_levels, n_lat, n_lon = 10, 4, 8
        t = jnp.full((n_levels, n_lat, n_lon), 280.0)
        p = jnp.full((n_levels, n_lat, n_lon), 8.0e4)
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 1.2  # 120% RH

        t_new, q_new, _condensate = large_scale_condensation(
            t,
            q,
            p,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        energy_change = EARTH.specific_heat_cp * (t_new - t) + EARTH.latent_heat_vaporization * (
            q_new - q
        )
        np.testing.assert_allclose(energy_change, 0.0, atol=1e-6)

    def test_temperature_increases(self) -> None:
        """Condensation should warm the air (latent heat release)."""
        n_levels, n_lat, n_lon = 5, 2, 4
        t = jnp.full((n_levels, n_lat, n_lon), 280.0)
        p = jnp.full((n_levels, n_lat, n_lon), 8.0e4)
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 1.1

        t_new, _q_new, _condensate = large_scale_condensation(
            t,
            q,
            p,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        assert jnp.all(t_new >= t)

    def test_cold_temperature_robustness(self) -> None:
        """Condensation should handle very cold temperatures."""
        n_levels, n_lat, n_lon = 5, 2, 4
        t = jnp.full((n_levels, n_lat, n_lon), 200.0)
        p = jnp.full((n_levels, n_lat, n_lon), 1.0e4)
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 2.0  # very supersaturated

        t_new, q_new, _condensate = large_scale_condensation(
            t,
            q,
            p,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        assert jnp.all(jnp.isfinite(t_new))
        assert jnp.all(jnp.isfinite(q_new))

    def test_condensate_matches_moisture_removal(self) -> None:
        """Condensate should equal moisture removed per level."""
        n_levels, n_lat, n_lon = 10, 4, 8
        t = jnp.full((n_levels, n_lat, n_lon), 280.0)
        p = jnp.full((n_levels, n_lat, n_lon), 8.0e4)
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 1.15

        _t_new, q_new, condensate = large_scale_condensation(
            t,
            q,
            p,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )
        np.testing.assert_allclose(condensate, q - q_new, atol=1e-12)


# ---------------------------------------------------------------------------
# Betts-Miller convection
# ---------------------------------------------------------------------------


class TestBettsMillerConvection:
    def test_stable_column_no_convection(self) -> None:
        """Stable column should produce zero tendencies."""
        n_levels, n_lat, n_lon = 10, 4, 8
        # Very stable: cold at bottom, warm aloft (inverted)
        sigma = jnp.linspace(0.05, 0.95, n_levels)
        p = jnp.broadcast_to(sigma[:, None, None] * 1.0e5, (n_levels, n_lat, n_lon)).copy()
        # Temperature increasing upward (stable against convection)
        t = jnp.broadcast_to(
            (250.0 + 50.0 * (1.0 - sigma))[:, None, None],
            (n_levels, n_lat, n_lon),
        ).copy()
        # Completely dry column → Pq = ∫(0 - q_ref)dσ < 0 → no trigger
        q = jnp.zeros_like(t)
        dsigma = jnp.full(n_levels, 1.0 / n_levels)

        dt, dq = betts_miller_convection(
            t,
            q,
            p,
            dsigma,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        # Should be zero — dry column does not trigger (Pq < 0)
        assert float(jnp.max(jnp.abs(dt))) < 1e-8
        assert float(jnp.max(jnp.abs(dq))) < 1e-12

    def test_unstable_column_produces_tendencies(self) -> None:
        """Column with steep lapse rate should trigger BM convection."""
        n_levels, n_lat, n_lon = 20, 4, 8
        sigma = jnp.linspace(0.025, 0.975, n_levels)
        p = jnp.broadcast_to(sigma[:, None, None] * 1.0e5, (n_levels, n_lat, n_lon)).copy()
        # Steep lapse rate: warm surface (300K), cold aloft (220K).
        # The moist adiabat from 300K cools to ~230K at the top,
        # so the environment (220K) is colder → parcel is buoyant.
        t = jnp.broadcast_to(
            (300.0 - 80.0 * (1.0 - sigma))[:, None, None],
            (n_levels, n_lat, n_lon),
        ).copy()
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 0.95
        dsigma = jnp.full(n_levels, 1.0 / n_levels)

        dt, dq = betts_miller_convection(
            t,
            q,
            p,
            dsigma,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        assert float(jnp.max(jnp.abs(dt))) > 1e-6
        assert float(jnp.max(jnp.abs(dq))) > 1e-10

    def test_enthalpy_conservation(self) -> None:
        """Column-integrated moist enthalpy tendency should be near zero."""
        n_levels, n_lat, n_lon = 20, 4, 8
        sigma = jnp.linspace(0.025, 0.975, n_levels)
        p = jnp.broadcast_to(sigma[:, None, None] * 1.0e5, (n_levels, n_lat, n_lon)).copy()
        t = jnp.broadcast_to(
            (300.0 - 80.0 * (1.0 - sigma))[:, None, None],
            (n_levels, n_lat, n_lon),
        ).copy()
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 0.95
        dsigma = jnp.full(n_levels, 1.0 / n_levels)

        dt, dq = betts_miller_convection(
            t,
            q,
            p,
            dsigma,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        # Column enthalpy tendency: ∫(cp*dT + L*dq) dσ ≈ 0
        enthalpy_tend = jnp.sum(
            (EARTH.specific_heat_cp * dt + EARTH.latent_heat_vaporization * dq)
            * dsigma[:, None, None],
            axis=0,
        )
        # Normalized by column enthalpy
        enthalpy_col = jnp.sum(
            (EARTH.specific_heat_cp * t + EARTH.latent_heat_vaporization * q)
            * dsigma[:, None, None],
            axis=0,
        )
        relative = jnp.abs(enthalpy_tend) / jnp.abs(enthalpy_col)
        assert float(jnp.max(relative)) < 0.01

    def test_relaxation_timescale(self) -> None:
        """Tendency magnitude should be proportional to 1/tau."""
        n_levels, n_lat, n_lon = 20, 2, 4
        sigma = jnp.linspace(0.025, 0.975, n_levels)
        p = jnp.broadcast_to(sigma[:, None, None] * 1.0e5, (n_levels, n_lat, n_lon)).copy()
        t = jnp.broadcast_to(
            (300.0 - 80.0 * (1.0 - sigma))[:, None, None],
            (n_levels, n_lat, n_lon),
        ).copy()
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 0.95
        dsigma = jnp.full(n_levels, 1.0 / n_levels)

        dt_fast, _ = betts_miller_convection(
            t,
            q,
            p,
            dsigma,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            tau_bm=3600.0,
        )
        dt_slow, _ = betts_miller_convection(
            t,
            q,
            p,
            dsigma,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            tau_bm=7200.0,
        )

        # Fast should be ~2x stronger than slow
        ratio = float(jnp.max(jnp.abs(dt_fast)) / jnp.max(jnp.abs(dt_slow)))
        np.testing.assert_allclose(ratio, 2.0, rtol=0.1)

    def test_no_tendencies_above_lzb(self) -> None:
        """Tendencies should be zero above the level of zero buoyancy."""
        n_levels, n_lat, n_lon = 20, 4, 8
        sigma = jnp.linspace(0.025, 0.975, n_levels)
        p = jnp.broadcast_to(sigma[:, None, None] * 1.0e5, (n_levels, n_lat, n_lon)).copy()
        # Steep lapse rate — buoyant only in lower troposphere
        t = jnp.broadcast_to(
            (300.0 - 80.0 * (1.0 - sigma))[:, None, None],
            (n_levels, n_lat, n_lon),
        ).copy()
        q_sat = saturation_specific_humidity(t, p, EARTH.epsilon_moisture)
        q = q_sat * 0.95
        dsigma = jnp.full(n_levels, 1.0 / n_levels)

        dt, dq = betts_miller_convection(
            t,
            q,
            p,
            dsigma,
            EARTH.epsilon_moisture,
            EARTH.latent_heat_vaporization,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
        )

        # Upper levels (σ < 0.2) should have zero tendencies
        upper = sigma < 0.2
        assert float(jnp.max(jnp.abs(dt[upper]))) < 1e-15
        assert float(jnp.max(jnp.abs(dq[upper]))) < 1e-15


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestSimplePhysicsConfigValidation:
    def test_tau_bm_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="tau_bm"):
            SimplePhysicsConfig(tau_bm=0.0)

    def test_rh_ref_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="rh_ref"):
            SimplePhysicsConfig(rh_ref=1.5)

    def test_rh_ref_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="rh_ref"):
            SimplePhysicsConfig(rh_ref=0.0)

    def test_n_condensation_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_condensation_iterations"):
            SimplePhysicsConfig(n_condensation_iterations=0)

    def test_rh_condensation_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="rh_condensation"):
            SimplePhysicsConfig(rh_condensation=1.1)

    def test_valid_config_accepted(self) -> None:
        cfg = SimplePhysicsConfig(
            tau_bm=3600.0,
            rh_ref=0.8,
            n_condensation_iterations=5,
            rh_condensation=0.95,
        )
        np.testing.assert_allclose(cfg.tau_bm, 3600.0)


# ---------------------------------------------------------------------------
# Moist aquaplanet initial conditions
# ---------------------------------------------------------------------------


class TestMoistInitialConditions:
    def test_has_humidity(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Moist initial condition should have humidity field."""
        state, _ref, _geo = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
        )
        assert state.has_humidity
        assert state.humidity is not None
        assert state.humidity.shape == state.temperature.shape

    def test_humidity_positive_in_troposphere(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Humidity should be positive in the troposphere."""
        state, _ref, _geo = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
        )
        assert state.humidity is not None
        q_grid = jax.vmap(transform.spectral_to_grid)(state.humidity)

        # Lower troposphere (sigma > 0.5) should have positive q
        sigma = np.asarray(levels.sigma_full)
        tropo_mask = sigma > 0.5
        q_tropo = q_grid[tropo_mask]
        # Mean should be positive (some Gibbs ringing possible)
        assert float(jnp.mean(q_tropo.real)) > 0

    def test_humidity_bounded_and_positive_mean(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Global mean humidity should be positive and bounded."""
        state, _ref, _geo = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
        )
        assert state.humidity is not None
        q_grid = jax.vmap(transform.spectral_to_grid)(state.humidity)

        global_mean = float(jnp.mean(q_grid.real))
        assert global_mean > 0
        assert global_mean < 0.05  # < 50 g/kg


# ---------------------------------------------------------------------------
# Moist aquaplanet integration
# ---------------------------------------------------------------------------


class TestMoistAquaplanetIntegration:
    def test_10_day_stability(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Moist aquaplanet should remain stable for 10 days at T21 L20."""
        state, ref_temps, surf_geo = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
        )
        dt = 600.0
        filt = exponential_filter(transform.arrays, dt)
        forcing = SimplePhysics(transform, EARTH, levels)
        init_fn, step_fn = build_pe_stepper(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            dt=dt,
            spectral_filter=filt,
            forcing=forcing,
        )

        prev, curr = init_fn(state)
        n_steps = int(10 * 86400 / dt)  # 10 days
        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        # All fields should be finite
        assert jnp.all(jnp.isfinite(curr.temperature))
        assert jnp.all(jnp.isfinite(curr.vorticity))
        assert curr.humidity is not None
        assert jnp.all(jnp.isfinite(curr.humidity))

        # Temperature should be in a reasonable range
        t_grid = jax.vmap(transform.spectral_to_grid)(curr.temperature)
        assert float(jnp.min(t_grid)) > 150.0  # not frozen
        assert float(jnp.max(t_grid)) < 350.0  # not boiling

    def test_humidity_physically_reasonable(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """After 10 days, humidity should remain physically plausible."""
        state, ref_temps, surf_geo = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
        )
        dt = 600.0
        filt = exponential_filter(transform.arrays, dt)
        forcing = SimplePhysics(transform, EARTH, levels)
        init_fn, step_fn = build_pe_stepper(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            dt=dt,
            spectral_filter=filt,
            forcing=forcing,
        )

        prev, curr = init_fn(state)
        n_steps = int(10 * 86400 / dt)
        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        assert curr.humidity is not None
        q_grid = jax.vmap(transform.spectral_to_grid)(curr.humidity)

        # Humidity should not explode
        assert float(jnp.max(jnp.abs(q_grid))) < 0.1  # < 100 g/kg

        # Mean tropospheric humidity should be positive
        sigma = np.asarray(levels.sigma_full)
        tropo_mask = sigma > 0.5
        q_tropo_mean = float(jnp.mean(q_grid[tropo_mask].real))
        assert q_tropo_mean > 0

    def test_dry_aquaplanet_still_works(
        self,
        transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Dry SimplePhysics (no humidity) still works after moist changes."""
        from notus.initial_conditions import simple_physics_initial_state

        state, ref_temps, surf_geo = simple_physics_initial_state(
            transform,
            EARTH,
            levels,
        )
        dt = 600.0
        filt = exponential_filter(transform.arrays, dt)
        forcing = SimplePhysics(transform, EARTH, levels)
        init_fn, step_fn = build_pe_stepper(
            transform,
            EARTH,
            levels,
            ref_temps,
            surf_geo,
            dt=dt,
            spectral_filter=filt,
            forcing=forcing,
        )

        prev, curr = init_fn(state)
        for _ in range(50):
            prev, curr = step_fn(prev, curr)

        assert not curr.has_humidity
        assert jnp.all(jnp.isfinite(curr.temperature))
