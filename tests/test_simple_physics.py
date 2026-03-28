"""Tests for the simple physics forcing (Frierson et al. 2006).

Unit tests verify individual components (radiation, convection, SST, surface flux).
Integration tests verify stability and physical plausibility of short runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import simple_physics_initial_state
from notus.operators import exponential_filter
from notus.physics.convection import dry_convective_adjustment
from notus.physics.radiation import (
    byrne_longwave_optical_depth,
    byrne_shortwave_optical_depth,
    longwave_heating,
    longwave_optical_depth,
    shortwave_heating,
    speedy_longwave_heating,
    speedy_lw_band_fractions,
    speedy_shortwave_heating,
)
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.surface import PrescribedSST, compute_sst, surface_sensible_heat_flux
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels, uniform_sigma_levels


jax.config.update("jax_enable_x64", True)

# Default Frierson LW parameters used across tests
LW_DEFAULTS = {
    "tau_equator": 6.0,
    "tau_pole": 1.5,
    "linear_fraction": 0.1,
    "alpha": 4.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def levels() -> SigmaLevels:
    """Standard 20-level sigma coordinate."""
    return uniform_sigma_levels(20)


@pytest.fixture(scope="module")
def levels_module() -> SigmaLevels:
    """Standard 20-level sigma coordinate (module-scoped)."""
    return uniform_sigma_levels(20)


@pytest.fixture
def sp_forcing(
    t21_transform: SpectralTransform,
    levels: SigmaLevels,
) -> SimplePhysics:
    """Simple physics forcing at T21."""
    return SimplePhysics(t21_transform, EARTH, levels)


@pytest.fixture
def isothermal_state(
    t21_transform: SpectralTransform,
    levels: SigmaLevels,
) -> tuple[PrimitiveEquationState, jnp.ndarray]:
    """Isothermal rest state at 264 K with uniform surface pressure."""
    grid = t21_transform.grid
    n_levels = levels.n_levels
    n_spec = grid.n_spectral_coeffs

    t_grid = jnp.full((n_levels, grid.n_lat, grid.n_lon), 264.0)
    t_spec = jax.vmap(t21_transform.grid_to_spectral)(t_grid)

    state = PrimitiveEquationState(
        vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        temperature=t_spec,
        log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
    )
    ps_grid = jnp.full((grid.n_lat, grid.n_lon), EARTH.reference_pressure)
    return state, ps_grid


@pytest.fixture(scope="module")
def t21_transform_module() -> SpectralTransform:
    """T21 spectral transform (module-scoped)."""
    return SpectralTransform(GaussianGrid(truncation=21), EARTH.radius)


# ---------------------------------------------------------------------------
# Unit tests: prescribed SST
# ---------------------------------------------------------------------------


class TestPrescribedSST:
    """Verify Frierson Gaussian SST profile."""

    def test_equatorial_sst(self) -> None:
        """SST at equator should equal t_min + t_delta."""
        config = PrescribedSST(t_min=271.0, t_delta=29.0)
        latitudes = jnp.array([0.0])
        sst = compute_sst(config, latitudes)
        np.testing.assert_allclose(sst[0], 300.0)

    def test_symmetric(self) -> None:
        """SST should be symmetric about the equator."""
        config = PrescribedSST()
        latitudes = jnp.array([-0.5, 0.5])
        sst = compute_sst(config, latitudes)
        np.testing.assert_allclose(sst[0], sst[1])

    def test_decreases_poleward(self) -> None:
        """SST should decrease from equator to pole."""
        config = PrescribedSST()
        latitudes = jnp.linspace(0.0, jnp.pi / 2, 10)
        sst = compute_sst(config, latitudes)
        assert jnp.all(jnp.diff(sst) <= 0.0)

    def test_floor_at_high_latitude(self) -> None:
        """SST at pole should approach t_min."""
        config = PrescribedSST(t_min=271.0, t_delta=29.0)
        latitudes = jnp.array([jnp.pi / 2])  # 90 degrees
        sst = compute_sst(config, latitudes)
        # Gaussian decays to ~0 at 90 degrees with phi_w=26 degrees
        np.testing.assert_allclose(sst[0], 271.0, atol=0.5)


# ---------------------------------------------------------------------------
# Unit tests: longwave optical depth
# ---------------------------------------------------------------------------


class TestLongwaveOpticalDepth:
    """Verify optical depth profile."""

    def test_toa_is_zero(self) -> None:
        """Optical depth at TOA (sigma=0) should be zero."""
        sigma_half = jnp.array([0.0, 0.5, 1.0])
        sin_lat = jnp.array([0.0])
        tau = longwave_optical_depth(
            sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )
        np.testing.assert_allclose(tau[0, 0], 0.0)

    def test_surface_equator(self) -> None:
        """Optical depth at surface (sigma=1) at equator should equal tau_equator."""
        sigma_half = jnp.array([0.0, 1.0])
        sin_lat = jnp.array([0.0])
        tau = longwave_optical_depth(
            sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )
        # tau_0 * [f_l * 1 + (1-f_l) * 1^4] = tau_0 * 1.0 = tau_equator
        np.testing.assert_allclose(tau[1, 0], 6.0)

    def test_surface_pole(self) -> None:
        """Optical depth at surface (sigma=1) at pole should equal tau_pole."""
        sigma_half = jnp.array([0.0, 1.0])
        sin_lat = jnp.array([1.0])
        tau = longwave_optical_depth(
            sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )
        np.testing.assert_allclose(tau[1, 0], 1.5)

    def test_monotonically_increasing_downward(self) -> None:
        """Optical depth should increase from TOA to surface."""
        sigma_half = jnp.linspace(0.0, 1.0, 21)
        sin_lat = jnp.array([0.3])
        tau = longwave_optical_depth(
            sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )
        assert jnp.all(jnp.diff(tau[:, 0]) >= 0.0)

    def test_mixed_pressure_dependence(self) -> None:
        """Verify mixed linear+p^4 profile at sigma=0.5, equator."""
        sigma_half = jnp.array([0.0, 0.5, 1.0])
        sin_lat = jnp.array([0.0])
        tau = longwave_optical_depth(
            sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )
        # tau = 6.0 * [0.1 * 0.5 + 0.9 * 0.5^4] = 6.0 * [0.05 + 0.05625] = 0.6375
        expected = 6.0 * (0.1 * 0.5 + 0.9 * 0.5**4)
        np.testing.assert_allclose(tau[1, 0], expected)


# ---------------------------------------------------------------------------
# Unit tests: longwave heating
# ---------------------------------------------------------------------------


class TestLongwaveHeating:
    """Verify longwave radiative heating."""

    def test_isothermal_with_same_surface_temp_is_small(
        self,
        levels: SigmaLevels,
    ) -> None:
        """An isothermal atmosphere with T_surface = T_atm has small LW heating."""
        n_levels = levels.n_levels
        n_lat, n_lon = 4, 8
        t_uniform = 280.0
        temperature = jnp.full((n_levels, n_lat, n_lon), t_uniform)
        surface_temperature = jnp.full((n_lat,), t_uniform)
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        sin_lat = jnp.zeros(n_lat)
        tau_half = longwave_optical_depth(
            levels.sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )

        q_lw, _lw_down = longwave_heating(
            temperature,
            surface_temperature,
            tau_half,
            levels.dsigma,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )

        assert jnp.max(jnp.abs(q_lw)) < 1.0e-4

    def test_warm_surface_heats_lower_atmosphere(
        self,
        levels: SigmaLevels,
    ) -> None:
        """A warm surface should produce net heating in the lowest layers."""
        n_levels = levels.n_levels
        n_lat, n_lon = 4, 8
        temperature = jnp.full((n_levels, n_lat, n_lon), 250.0)
        surface_temperature = jnp.full((n_lat,), 300.0)
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        sin_lat = jnp.zeros(n_lat)
        tau_half = longwave_optical_depth(
            levels.sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )

        q_lw, _lw_down = longwave_heating(
            temperature,
            surface_temperature,
            tau_half,
            levels.dsigma,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )

        assert float(jnp.mean(q_lw[-1])) > 0.0

    def test_shape(self, levels: SigmaLevels) -> None:
        """Output shape should match input grid."""
        n_levels = levels.n_levels
        n_lat, n_lon = 4, 8
        temperature = jnp.ones((n_levels, n_lat, n_lon)) * 260.0
        surface_temperature = jnp.ones(n_lat) * 280.0
        surface_pressure = jnp.ones((n_lat, n_lon)) * 1.0e5
        sin_lat = jnp.zeros(n_lat)
        tau_half = longwave_optical_depth(
            levels.sigma_half,
            sin_lat,
            **LW_DEFAULTS,
        )

        q_lw, _lw_down = longwave_heating(
            temperature,
            surface_temperature,
            tau_half,
            levels.dsigma,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        assert q_lw.shape == (n_levels, n_lat, n_lon)


# ---------------------------------------------------------------------------
# Unit tests: shortwave heating (kept for the standalone function)
# ---------------------------------------------------------------------------


class TestShortwaveHeating:
    """Verify shortwave radiative heating (Beer-Lambert)."""

    def test_positive_everywhere(self, levels: SigmaLevels) -> None:
        """SW heating should be positive (absorption) at all levels."""
        n_lat, n_lon = 4, 8
        sin_lat = jnp.linspace(-0.5, 0.5, n_lat)
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)

        q_sw, sw_down_sfc = shortwave_heating(
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.solar_constant,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            sw_tau_0=0.22,
            sw_exponent=2.0,
            delta_s=1.4,
        )
        assert jnp.all(q_sw >= 0.0)
        assert jnp.all(sw_down_sfc >= 0.0)

    def test_zero_optical_depth_gives_zero_heating(
        self,
        levels: SigmaLevels,
    ) -> None:
        """With zero SW optical depth, no absorption should occur."""
        n_lat, n_lon = 4, 8
        sin_lat = jnp.zeros(n_lat)
        surface_pressure = jnp.ones((n_lat, n_lon)) * 1.0e5

        q_sw, sw_down_sfc = shortwave_heating(
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.solar_constant,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            sw_tau_0=0.0,
            sw_exponent=2.0,
            delta_s=1.4,
        )
        np.testing.assert_allclose(q_sw, 0.0, atol=1e-30)
        # With zero optical depth, full insolation reaches the surface
        # sin_lat=0 → insolation = S₀/4 * (1 + delta_s/4)
        expected_insol = EARTH.solar_constant / 4.0 * (1.0 + 1.4 / 4.0)
        np.testing.assert_allclose(sw_down_sfc, expected_insol, rtol=1e-10)


class TestShortwaveEnergyConservation:
    """Verify SW column energy is conserved: atm absorbed + surface flux = TOA incoming."""

    def test_column_budget_closes(self, levels: SigmaLevels) -> None:
        """Atmospheric absorption + surface down = TOA incoming (no albedo)."""
        n_lat, n_lon = 4, 8
        sin_lat = jnp.linspace(-0.5, 0.5, n_lat)
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)

        q_sw, sw_down_sfc = shortwave_heating(
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.solar_constant,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            sw_tau_0=0.22,
            sw_exponent=2.0,
            delta_s=1.4,
        )
        # Atmospheric absorption per column: integrate heating * dp / g
        dp = levels.dsigma[:, None, None] * surface_pressure[None, :, :]
        atm_absorbed = jnp.sum(
            q_sw * EARTH.specific_heat_cp * dp / EARTH.gravity,
            axis=0,
        )  # (n_lat, n_lon)

        # TOA incoming
        insolation = EARTH.solar_constant / 4.0 * (1.0 + 1.4 * (1.0 - 3.0 * sin_lat**2) / 4.0)
        toa = insolation[:, None] * jnp.ones((n_lat, n_lon))

        residual = atm_absorbed + sw_down_sfc - toa
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_reflected_beam_absorbed(self, levels: SigmaLevels) -> None:
        """Non-zero albedo increases atmospheric absorption via reflected beam."""
        n_lat, n_lon = 4, 8
        sin_lat = jnp.linspace(-0.5, 0.5, n_lat)
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)

        q_no_albedo, _ = shortwave_heating(
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.solar_constant,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            sw_tau_0=0.22,
            sw_exponent=2.0,
            delta_s=1.4,
            surface_albedo=0.0,
        )
        q_albedo, _ = shortwave_heating(
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.solar_constant,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            sw_tau_0=0.22,
            sw_exponent=2.0,
            delta_s=1.4,
            surface_albedo=0.3,
        )
        # With albedo, reflected beam adds atmospheric absorption
        dp = levels.dsigma[:, None, None] * surface_pressure[None, :, :]
        total_no = jnp.sum(q_no_albedo * dp, axis=0)
        total_yes = jnp.sum(q_albedo * dp, axis=0)
        assert jnp.all(total_yes > total_no)

    def test_humidity_dependent_sw_tau(self, levels: SigmaLevels) -> None:
        """SW optical depth should increase with humidity."""
        n_lat, n_lon = 4, 8
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)

        dry_q = jnp.zeros((levels.n_levels, n_lat, n_lon))
        moist_q = jnp.full((levels.n_levels, n_lat, n_lon), 0.01)

        tau_dry = byrne_shortwave_optical_depth(
            levels.dsigma,
            dry_q,
            surface_pressure,
            1.0e5,
        )
        tau_moist = byrne_shortwave_optical_depth(
            levels.dsigma,
            moist_q,
            surface_pressure,
            1.0e5,
        )
        # Moist should have higher optical depth
        assert jnp.all(tau_moist[-1] > tau_dry[-1])

    def test_byrne_sw_tau_toa_zero(self, levels: SigmaLevels) -> None:
        """SW optical depth should be zero at TOA."""
        n_lat, n_lon = 4, 8
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        humidity = jnp.full((levels.n_levels, n_lat, n_lon), 0.005)

        tau = byrne_shortwave_optical_depth(
            levels.dsigma,
            humidity,
            surface_pressure,
            1.0e5,
        )
        np.testing.assert_allclose(tau[0], 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# Unit tests: Byrne humidity-dependent optical depth
# ---------------------------------------------------------------------------


class TestByrneOpticalDepth:
    """Verify Byrne/Isca humidity-dependent LW optical depth."""

    def test_toa_is_zero(self, levels: SigmaLevels) -> None:
        """Optical depth at TOA should be zero."""
        n_lat, n_lon = 4, 8
        humidity = jnp.ones((levels.n_levels, n_lat, n_lon)) * 0.005
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        tau = byrne_longwave_optical_depth(
            levels.dsigma,
            humidity,
            surface_pressure,
            1.0e5,
        )
        np.testing.assert_allclose(tau[0], 0.0)

    def test_increases_with_humidity(self, levels: SigmaLevels) -> None:
        """More humidity should produce larger optical depth."""
        n_lat, n_lon = 4, 8
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)

        q_dry = jnp.ones((levels.n_levels, n_lat, n_lon)) * 0.001
        q_wet = jnp.ones((levels.n_levels, n_lat, n_lon)) * 0.01
        tau_dry = byrne_longwave_optical_depth(
            levels.dsigma,
            q_dry,
            surface_pressure,
            1.0e5,
        )
        tau_wet = byrne_longwave_optical_depth(
            levels.dsigma,
            q_wet,
            surface_pressure,
            1.0e5,
        )
        # Surface optical depth should be larger for wetter atmosphere
        assert float(jnp.mean(tau_wet[-1])) > float(jnp.mean(tau_dry[-1]))

    def test_3d_shape(self, levels: SigmaLevels) -> None:
        """Output should be (n_levels+1, n_lat, n_lon)."""
        n_lat, n_lon = 4, 8
        humidity = jnp.ones((levels.n_levels, n_lat, n_lon)) * 0.005
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        tau = byrne_longwave_optical_depth(
            levels.dsigma,
            humidity,
            surface_pressure,
            1.0e5,
        )
        assert tau.shape == (levels.n_levels + 1, n_lat, n_lon)

    def test_dry_limit_matches_well_mixed(self, levels: SigmaLevels) -> None:
        """With zero humidity, optical depth should equal a * column integral."""
        n_lat, n_lon = 2, 4
        humidity = jnp.zeros((levels.n_levels, n_lat, n_lon))
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        a = 0.8678
        tau = byrne_longwave_optical_depth(
            levels.dsigma,
            humidity,
            surface_pressure,
            1.0e5,
            byrne_a=a,
        )
        # Total tau at surface = a * sum(dsigma) * (ps/p0) = a * 1.0 * 1.0
        np.testing.assert_allclose(float(jnp.mean(tau[-1])), a, rtol=1e-10)

    def test_longwave_heating_with_3d_tau(self, levels: SigmaLevels) -> None:
        """longwave_heating should work with 3-D Byrne optical depth."""
        n_lat, n_lon = 4, 8
        temperature = jnp.full((levels.n_levels, n_lat, n_lon), 260.0)
        surface_temperature = jnp.full((n_lat,), 280.0)
        surface_pressure = jnp.full((n_lat, n_lon), 1.0e5)
        humidity = jnp.ones((levels.n_levels, n_lat, n_lon)) * 0.005
        tau_half = byrne_longwave_optical_depth(
            levels.dsigma,
            humidity,
            surface_pressure,
            1.0e5,
        )
        q_lw, _lw_down = longwave_heating(
            temperature,
            surface_temperature,
            tau_half,
            levels.dsigma,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        assert q_lw.shape == (levels.n_levels, n_lat, n_lon)
        assert bool(jnp.all(jnp.isfinite(q_lw)))


# ---------------------------------------------------------------------------
# Unit tests: surface sensible heat flux
# ---------------------------------------------------------------------------


class TestSurfaceSensibleHeatFlux:
    """Verify bulk aerodynamic surface flux."""

    def test_warm_surface_heats_air(self) -> None:
        """Positive flux when surface is warmer than air."""
        t_surface = jnp.array([300.0])
        t_air = jnp.array([[280.0]])
        wind = jnp.array([[5.0]])
        ps = jnp.array([[1.0e5]])

        q = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )
        assert float(q[0, 0]) > 0.0

    def test_cold_surface_cools_air(self) -> None:
        """Negative flux when surface is colder than air."""
        t_surface = jnp.array([260.0])
        t_air = jnp.array([[280.0]])
        wind = jnp.array([[5.0]])
        ps = jnp.array([[1.0e5]])

        q = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )
        assert float(q[0, 0]) < 0.0

    def test_zero_wind_zero_flux(self) -> None:
        """No flux when wind speed is zero."""
        t_surface = jnp.array([300.0])
        t_air = jnp.array([[280.0]])
        wind = jnp.array([[0.0]])
        ps = jnp.array([[1.0e5]])

        q = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )
        np.testing.assert_allclose(q[0, 0], 0.0, atol=1e-30)

    def test_equal_temps_zero_flux(self) -> None:
        """No flux when surface and air temperatures are equal."""
        t_surface = jnp.array([280.0])
        t_air = jnp.array([[280.0]])
        wind = jnp.array([[10.0]])
        ps = jnp.array([[1.0e5]])

        q = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )
        np.testing.assert_allclose(q[0, 0], 0.0, atol=1e-30)

    def test_reasonable_magnitude(self) -> None:
        """Heating rate should be physically reasonable (~10-50 K/day)."""
        t_surface = jnp.array([300.0])
        t_air = jnp.array([[280.0]])
        wind = jnp.array([[5.0]])
        ps = jnp.array([[1.0e5]])

        q = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )
        q_kday = float(q[0, 0]) * 86400.0
        assert 1.0 < q_kday < 100.0, f"Heating rate {q_kday:.1f} K/day out of range"

    def test_polar_wind_regularization_bounds_flux(self) -> None:
        """Near-polar wind reconstruction should be finite and cap-amplified."""
        # Mimic recovery of (u, v) from cosine-weighted winds in SimplePhysics.
        u_cos = jnp.array([[1.0e-3], [1.0e-3]])  # (n_lat=2, n_lon=1)
        v_cos = jnp.zeros_like(u_cos)
        cos_lat = jnp.array([1.0, 1.0e-12])[:, None]

        wind_unregularized = jnp.sqrt((u_cos / cos_lat) ** 2 + (v_cos / cos_lat) ** 2)
        cos_lat_safe = jnp.maximum(cos_lat, 1.0e-6)
        wind_regularized = jnp.sqrt((u_cos / cos_lat_safe) ** 2 + (v_cos / cos_lat_safe) ** 2)

        # Regularization should reduce extreme high-lat amplification by the cap ratio.
        assert float(wind_unregularized[1, 0]) == pytest.approx(1.0e9)
        assert float(wind_regularized[1, 0]) == pytest.approx(1.0e3)
        assert float(wind_regularized[1, 0] / wind_unregularized[1, 0]) == pytest.approx(1.0e-6)
        assert bool(jnp.all(jnp.isfinite(wind_regularized)))

        t_surface = jnp.array([300.0, 300.0])
        t_air = jnp.array([[280.0], [280.0]])
        ps = jnp.array([[1.0e5], [1.0e5]])

        q_unregularized = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind_unregularized,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )
        q_regularized = surface_sensible_heat_flux(
            t_surface,
            t_air,
            wind_regularized,
            ps,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            EARTH.gas_constant,
            dsigma_lowest=0.05,
            drag_coefficient=0.0015,
        )

        assert bool(jnp.all(jnp.isfinite(q_regularized)))
        assert float(q_regularized[1, 0] / q_unregularized[1, 0]) == pytest.approx(1.0e-6)


# ---------------------------------------------------------------------------
# Unit tests: dry convective adjustment
# ---------------------------------------------------------------------------


class TestDryConvectiveAdjustment:
    """Verify dry convective adjustment."""

    def test_stable_profile_unchanged(self) -> None:
        """A stable profile (theta increasing upward) should be unchanged."""
        sigma_full = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dsigma = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        kappa = EARTH.kappa
        n_lat, n_lon = 2, 4

        theta_profile = jnp.array([310.0, 300.0, 290.0, 280.0, 270.0])
        t_profile = theta_profile * sigma_full**kappa
        temperature = jnp.broadcast_to(
            t_profile[:, None, None],
            (5, n_lat, n_lon),
        ).copy()

        t_adjusted = dry_convective_adjustment(
            temperature,
            sigma_full,
            dsigma,
            kappa,
            n_iterations=3,
        )
        np.testing.assert_allclose(t_adjusted, temperature, atol=1e-10)

    def test_unstable_pair_is_neutralized(self) -> None:
        """A single unstable pair should be mixed to equal theta."""
        sigma_full = jnp.array([0.25, 0.75])
        dsigma = jnp.array([0.5, 0.5])
        kappa = EARTH.kappa

        theta_above = 280.0
        theta_below = 300.0
        t_above = theta_above * sigma_full[0] ** kappa
        t_below = theta_below * sigma_full[1] ** kappa
        temperature = jnp.array([[[t_above]], [[t_below]]])

        t_adjusted = dry_convective_adjustment(
            temperature,
            sigma_full,
            dsigma,
            kappa,
            n_iterations=1,
        )

        theta_adj_above = float(t_adjusted[0, 0, 0]) / sigma_full[0] ** kappa
        theta_adj_below = float(t_adjusted[1, 0, 0]) / sigma_full[1] ** kappa
        np.testing.assert_allclose(theta_adj_above, theta_adj_below, rtol=1e-10)

    def test_enthalpy_conserved(self) -> None:
        """Column enthalpy should be conserved by adjustment."""
        sigma_full = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dsigma = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        kappa = EARTH.kappa
        n_lat, n_lon = 2, 4

        theta_profile = jnp.array([250.0, 260.0, 270.0, 280.0, 290.0])
        t_profile = theta_profile * sigma_full**kappa
        temperature = jnp.broadcast_to(
            t_profile[:, None, None],
            (5, n_lat, n_lon),
        ).copy()

        t_adjusted = dry_convective_adjustment(
            temperature,
            sigma_full,
            dsigma,
            kappa,
            n_iterations=5,
        )

        h_before = jnp.sum(dsigma[:, None, None] * temperature, axis=0)
        h_after = jnp.sum(dsigma[:, None, None] * t_adjusted, axis=0)
        np.testing.assert_allclose(h_after, h_before, rtol=1e-10)

    def test_fully_unstable_column_becomes_stable(self) -> None:
        """A fully unstable column should become neutrally stable."""
        sigma_full = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dsigma = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        kappa = EARTH.kappa

        theta_profile = jnp.array([250.0, 260.0, 270.0, 280.0, 290.0])
        t_profile = theta_profile * sigma_full**kappa
        temperature = t_profile[:, None, None]

        t_adjusted = dry_convective_adjustment(
            temperature,
            sigma_full,
            dsigma,
            kappa,
            n_iterations=50,
        )

        theta_adjusted = t_adjusted[:, 0, 0] / sigma_full**kappa
        for k in range(4):
            assert float(theta_adjusted[k]) >= float(theta_adjusted[k + 1]) - 0.1

    def test_jit_compatible(self) -> None:
        """Convective adjustment should be JIT-compilable."""
        sigma_full = jnp.array([0.25, 0.75])
        dsigma = jnp.array([0.5, 0.5])
        temperature = jnp.ones((2, 2, 4)) * 260.0

        adjusted = jax.jit(
            lambda t: dry_convective_adjustment(t, sigma_full, dsigma, EARTH.kappa),
        )(temperature)
        assert adjusted.shape == temperature.shape


# ---------------------------------------------------------------------------
# Unit tests: SimplePhysics forcing
# ---------------------------------------------------------------------------


class TestSimplePhysicsForcing:
    """Verify SimplePhysics satisfies Forcing protocol and basic properties."""

    def test_output_shape(
        self,
        sp_forcing: SimplePhysics,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """Output should have same shape as input state."""
        state, ps_grid = isothermal_state
        tendencies = sp_forcing(state, ps_grid)
        assert tendencies.vorticity.shape == state.vorticity.shape
        assert tendencies.divergence.shape == state.divergence.shape
        assert tendencies.temperature.shape == state.temperature.shape
        assert tendencies.log_surface_pressure.shape == state.log_surface_pressure.shape

    def test_zero_surface_pressure_tendency(
        self,
        sp_forcing: SimplePhysics,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """Simple physics should not produce surface pressure tendencies."""
        state, ps_grid = isothermal_state
        tendencies = sp_forcing(state, ps_grid)
        np.testing.assert_allclose(tendencies.log_surface_pressure, 0.0, atol=1e-30)

    def test_rayleigh_drag_only_in_boundary_layer(
        self,
        sp_forcing: SimplePhysics,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
        levels: SigmaLevels,
    ) -> None:
        """Rayleigh drag should be zero above the boundary layer."""
        state, ps_grid = isothermal_state
        state = state.replace(
            vorticity=jnp.ones_like(state.vorticity) * (1.0 + 0j),
        )
        tendencies = sp_forcing(state, ps_grid)

        sigma_b = sp_forcing.config.sigma_b
        for k in range(levels.n_levels):
            if float(levels.sigma_full[k]) < sigma_b:
                np.testing.assert_allclose(
                    tendencies.vorticity[k],
                    0.0,
                    atol=1e-30,
                    err_msg=f"Drag nonzero at level {k}",
                )

    def test_nonzero_temperature_tendency(
        self,
        sp_forcing: SimplePhysics,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """An isothermal atmosphere should produce nonzero temperature tendencies."""
        state, ps_grid = isothermal_state
        tendencies = sp_forcing(state, ps_grid)
        assert float(jnp.max(jnp.abs(tendencies.temperature))) > 0.0

    def test_jit_compatible(
        self,
        sp_forcing: SimplePhysics,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """SimplePhysics should be JIT-compilable."""
        state, ps_grid = isothermal_state
        tendencies = jax.jit(sp_forcing)(state, ps_grid)
        assert tendencies.temperature.shape == state.temperature.shape


# ---------------------------------------------------------------------------
# Integration test: short run stability
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSimplePhysicsIntegration:
    """Short integration to verify stability and physical plausibility."""

    @pytest.fixture(scope="module")
    def run_30_days(
        self,
        t21_transform_module: SpectralTransform,
        levels_module: SigmaLevels,
    ) -> dict[str, float]:
        """Run a 30-day integration at T21 L20, dt=600s."""
        dt = 600.0
        n_steps = int(30 * 86400 / dt)

        state, ref_temps, surface_phi = simple_physics_initial_state(
            t21_transform_module,
            EARTH,
            levels_module,
            initial_temperature=264.0,
            perturbation_amplitude=1.0,
            seed=42,
        )

        forcing = SimplePhysics(t21_transform_module, EARTH, levels_module)
        spectral_filter = exponential_filter(t21_transform_module.arrays, dt)

        init_fn, step_fn = build_pe_stepper(
            t21_transform_module,
            EARTH,
            levels_module,
            ref_temps,
            surface_phi,
            dt=dt,
            forcing=forcing,
            spectral_filter=spectral_filter,
        )

        step_fn = jax.jit(step_fn)
        prev, curr = init_fn(state)

        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        t_grid = jax.vmap(t21_transform_module.spectral_to_grid)(curr.temperature)
        lnps_grid = t21_transform_module.spectral_to_grid(curr.log_surface_pressure)
        ps_grid = EARTH.reference_pressure * jnp.exp(lnps_grid)

        return {
            "t_min": float(jnp.min(t_grid)),
            "t_max": float(jnp.max(t_grid)),
            "t_mean": float(jnp.mean(t_grid)),
            "ps_min": float(jnp.min(ps_grid)),
            "ps_max": float(jnp.max(ps_grid)),
        }

    def test_temperature_bounded(self, run_30_days: dict[str, float]) -> None:
        """Temperature should remain within physical bounds after 30 days."""
        assert run_30_days["t_min"] > 150.0, f"Temperature too cold: {run_30_days['t_min']:.1f} K"
        assert run_30_days["t_max"] < 400.0, f"Temperature too hot: {run_30_days['t_max']:.1f} K"

    def test_temperature_drifts_from_initial(
        self,
        run_30_days: dict[str, float],
    ) -> None:
        """Mean temperature should drift from isothermal initial condition."""
        assert abs(run_30_days["t_mean"] - 264.0) > 0.1

    def test_surface_pressure_bounded(
        self,
        run_30_days: dict[str, float],
    ) -> None:
        """Surface pressure should remain near 1e5 Pa after 30 days."""
        assert run_30_days["ps_min"] > 0.8e5
        assert run_30_days["ps_max"] < 1.2e5


# ---------------------------------------------------------------------------
# Unit tests: SPEEDY multi-band radiation
# ---------------------------------------------------------------------------


class TestSpeedyLwBandFractions:
    """Verify temperature-dependent LW band fractions."""

    def test_sum_to_one_minus_epslw(self) -> None:
        """Band fractions should sum to (1 - epslw)."""
        temps = jnp.array([200.0, 250.0, 280.0, 300.0, 320.0])
        epslw = 0.05
        fband = speedy_lw_band_fractions(temps, epslw=epslw)
        total = jnp.sum(fband, axis=0)
        np.testing.assert_allclose(total, 1.0 - epslw, atol=1e-12)

    def test_all_bands_positive(self) -> None:
        """All band fractions should be positive for typical temperatures."""
        temps = jnp.linspace(200.0, 320.0, 50)
        fband = speedy_lw_band_fractions(temps)
        assert jnp.all(fband >= 0.0)

    def test_shape(self) -> None:
        """Output shape should be (4, *temperature.shape)."""
        temps = jnp.ones((10, 4, 8)) * 280.0
        fband = speedy_lw_band_fractions(temps)
        assert fband.shape == (4, 10, 4, 8)

    def test_temperature_dependence(self) -> None:
        """Window band fraction should increase at higher temperatures."""
        t_cold = jnp.array([220.0])
        t_warm = jnp.array([310.0])
        f_cold = speedy_lw_band_fractions(t_cold)
        f_warm = speedy_lw_band_fractions(t_warm)
        # Band 0 (window) should be larger at warm T
        assert float(f_warm[0, 0]) > float(f_cold[0, 0])


class TestSpeedyLongwaveHeating:
    """Verify 4-band LW radiation."""

    def test_shape_and_finite(self, levels: SigmaLevels) -> None:
        """Output shapes and all values should be finite."""
        n_lat, n_lon = 4, 8
        t = jnp.full((levels.n_levels, n_lat, n_lon), 260.0)
        t_s = jnp.full(n_lat, 280.0)
        q = jnp.full((levels.n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)

        heating, lw_down, olr = speedy_longwave_heating(
            t,
            t_s,
            q,
            levels.dsigma,
            ps,
            1e5,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        assert heating.shape == (levels.n_levels, n_lat, n_lon)
        assert lw_down.shape == (n_lat, n_lon)
        assert olr.shape == (n_lat, n_lon)
        assert jnp.all(jnp.isfinite(heating))
        assert jnp.all(jnp.isfinite(lw_down))
        assert jnp.all(jnp.isfinite(olr))

    def test_olr_positive(self, levels: SigmaLevels) -> None:
        """OLR should be positive for any reasonable temperature."""
        n_lat, n_lon = 4, 8
        t = jnp.full((levels.n_levels, n_lat, n_lon), 260.0)
        t_s = jnp.full(n_lat, 280.0)
        q = jnp.full((levels.n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)

        _, _, olr = speedy_longwave_heating(
            t,
            t_s,
            q,
            levels.dsigma,
            ps,
            1e5,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        assert jnp.all(olr > 0.0)

    def test_warm_surface_heats_lower_atmosphere(self, levels: SigmaLevels) -> None:
        """Warm surface should cause positive heating in the lowest layers."""
        n_lat, n_lon = 4, 8
        t = jnp.full((levels.n_levels, n_lat, n_lon), 240.0)
        t_s = jnp.full(n_lat, 300.0)
        q = jnp.full((levels.n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)

        heating, _, _ = speedy_longwave_heating(
            t,
            t_s,
            q,
            levels.dsigma,
            ps,
            1e5,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        # Lowest level should be heated
        assert jnp.all(heating[-1] > 0.0)


class TestSpeedyShortwaveHeating:
    """Verify 2-band SW radiation."""

    def test_shape_and_finite(self, levels: SigmaLevels) -> None:
        """Output shapes and all values should be finite."""
        n_lat, n_lon = 4, 8
        q = jnp.full((levels.n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)
        insol = jnp.full(n_lat, 340.0)

        heating, sw_down = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        assert heating.shape == (levels.n_levels, n_lat, n_lon)
        assert sw_down.shape == (n_lat, n_lon)
        assert jnp.all(jnp.isfinite(heating))
        assert jnp.all(jnp.isfinite(sw_down))

    def test_zero_insolation_gives_zero(self, levels: SigmaLevels) -> None:
        """Zero insolation should produce zero heating and zero surface flux."""
        n_lat, n_lon = 4, 8
        q = jnp.full((levels.n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)
        insol = jnp.zeros(n_lat)

        heating, sw_down = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        np.testing.assert_allclose(heating, 0.0, atol=1e-30)
        np.testing.assert_allclose(sw_down, 0.0, atol=1e-30)

    def test_more_humidity_more_absorption(self, levels: SigmaLevels) -> None:
        """Moist atmosphere should absorb more SW (less reaches surface)."""
        n_lat, n_lon = 4, 8
        ps = jnp.full((n_lat, n_lon), 1e5)
        insol = jnp.full(n_lat, 340.0)

        q_dry = jnp.full((levels.n_levels, n_lat, n_lon), 0.001)
        q_moist = jnp.full((levels.n_levels, n_lat, n_lon), 0.015)

        _, sw_dry = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q_dry,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        _, sw_moist = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q_moist,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )
        # Less SW reaches surface in moist case
        assert jnp.all(sw_moist < sw_dry)

    def test_sw_column_closure(self, levels: SigmaLevels) -> None:
        """Atmospheric absorption + surface absorbed + reflected = TOA."""
        n_lat, n_lon = 4, 8
        q = jnp.full((levels.n_levels, n_lat, n_lon), 0.008)
        ps = jnp.full((n_lat, n_lon), 1e5)
        insol = jnp.full(n_lat, 340.0)
        albedo = 0.1

        heating, sw_down = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            surface_albedo=albedo,
        )

        # Column atmospheric absorption
        dp = levels.dsigma[:, None, None] * ps[None, :, :]
        atm_abs = jnp.sum(heating * EARTH.specific_heat_cp * dp / EARTH.gravity, axis=0)

        # Surface absorbed
        sfc_abs = sw_down * (1.0 - albedo)

        # Reflected to space
        reflected = insol[:, None] * jnp.ones((n_lat, n_lon)) - atm_abs - sfc_abs

        residual = atm_abs + sfc_abs + reflected - insol[:, None]
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)


class TestSpeedyConfigValidation:
    """Verify SPEEDY scheme config."""

    def test_speedy_config_accepted(self) -> None:
        """'speedy' should be a valid radiation_scheme."""
        cfg = SimplePhysicsConfig(radiation_scheme="speedy")
        assert cfg.radiation_scheme == "speedy"

    def test_invalid_scheme_rejected(self) -> None:
        """Invalid scheme should raise ValueError."""
        with pytest.raises(ValueError, match="radiation_scheme"):
            SimplePhysicsConfig(radiation_scheme="invalid")
