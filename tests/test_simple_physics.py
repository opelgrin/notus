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
    longwave_heating,
    longwave_optical_depth,
    shortwave_heating,
)
from notus.physics.simple_physics import SimplePhysics
from notus.physics.surface import PrescribedSST, compute_sst, surface_sensible_heat_flux
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels, uniform_sigma_levels


jax.config.update("jax_enable_x64", True)

# Default Frierson LW parameters used across tests
LW_DEFAULTS = {
    "tau_equator": 6.0,
    "tau_pole": 0.1,
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
        np.testing.assert_allclose(tau[1, 0], 0.1)

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

        q_lw = longwave_heating(
            temperature,
            surface_temperature,
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            **LW_DEFAULTS,
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

        q_lw = longwave_heating(
            temperature,
            surface_temperature,
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            **LW_DEFAULTS,
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

        q_lw = longwave_heating(
            temperature,
            surface_temperature,
            levels.sigma_half,
            levels.dsigma,
            sin_lat,
            surface_pressure,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            **LW_DEFAULTS,
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

        q_sw = shortwave_heating(
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

    def test_zero_optical_depth_gives_zero_heating(
        self,
        levels: SigmaLevels,
    ) -> None:
        """With zero SW optical depth, no absorption should occur."""
        n_lat, n_lon = 4, 8
        sin_lat = jnp.zeros(n_lat)
        surface_pressure = jnp.ones((n_lat, n_lon)) * 1.0e5

        q_sw = shortwave_heating(
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
