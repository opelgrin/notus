"""Tests for the simple physics forcing (Frierson et al. 2006).

Unit tests verify individual components (radiation, convection, SST).
Integration tests verify stability and physical plausibility of short runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.initial_conditions import simple_physics_initial_state
from notus.operators import exponential_filter
from notus.physics.convection import dry_convective_adjustment
from notus.physics.radiation import (
    longwave_heating,
    longwave_optical_depth,
    shortwave_heating,
)
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.surface import PrescribedSST, compute_sst
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels, uniform_sigma_levels


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def levels() -> SigmaLevels:
    """Standard 20-level sigma coordinate."""
    return uniform_sigma_levels(20)


@pytest.fixture
def sp_config() -> SimplePhysicsConfig:
    """Default simple physics configuration."""
    return SimplePhysicsConfig()


@pytest.fixture
def sp_forcing(
    t21_transform: SpectralTransform, levels: SigmaLevels,
) -> SimplePhysics:
    """Simple physics forcing at T21 with dt=1200s."""
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


# ---------------------------------------------------------------------------
# Unit tests: prescribed SST
# ---------------------------------------------------------------------------


class TestPrescribedSST:
    """Verify SST profile against analytical values."""

    def test_equatorial_sst(self) -> None:
        """SST at equator should equal t_max."""
        config = PrescribedSST(t_max=285.0, delta_t=40.0)
        sin_lat = jnp.array([0.0])  # equator
        sst = compute_sst(config, sin_lat)
        np.testing.assert_allclose(sst[0], 285.0)

    def test_polar_sst(self) -> None:
        """SST at pole should equal t_max - delta_t."""
        config = PrescribedSST(t_max=285.0, delta_t=40.0)
        sin_lat = jnp.array([1.0])  # north pole
        sst = compute_sst(config, sin_lat)
        np.testing.assert_allclose(sst[0], 245.0)

    def test_symmetric(self) -> None:
        """SST should be symmetric about the equator."""
        config = PrescribedSST()
        sin_lat = jnp.array([-0.5, 0.5])
        sst = compute_sst(config, sin_lat)
        np.testing.assert_allclose(sst[0], sst[1])

    def test_monotone_equator_to_pole(self) -> None:
        """SST should decrease from equator to pole."""
        config = PrescribedSST()
        sin_lat = jnp.linspace(0.0, 1.0, 10)
        sst = compute_sst(config, sin_lat)
        assert jnp.all(jnp.diff(sst) <= 0.0)


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
            sigma_half, sin_lat, tau_equator=7.2, tau_pole=1.8, alpha=4.0,
        )
        np.testing.assert_allclose(tau[0, 0], 0.0)

    def test_surface_equator(self) -> None:
        """Optical depth at surface (sigma=1) at equator should equal tau_equator."""
        sigma_half = jnp.array([0.0, 1.0])
        sin_lat = jnp.array([0.0])
        tau = longwave_optical_depth(
            sigma_half, sin_lat, tau_equator=7.2, tau_pole=1.8, alpha=4.0,
        )
        np.testing.assert_allclose(tau[1, 0], 7.2)

    def test_surface_pole(self) -> None:
        """Optical depth at surface (sigma=1) at pole should equal tau_pole."""
        sigma_half = jnp.array([0.0, 1.0])
        sin_lat = jnp.array([1.0])
        tau = longwave_optical_depth(
            sigma_half, sin_lat, tau_equator=7.2, tau_pole=1.8, alpha=4.0,
        )
        np.testing.assert_allclose(tau[1, 0], 1.8)

    def test_monotonically_increasing_downward(self) -> None:
        """Optical depth should increase from TOA to surface."""
        sigma_half = jnp.linspace(0.0, 1.0, 21)
        sin_lat = jnp.array([0.3])
        tau = longwave_optical_depth(
            sigma_half, sin_lat, tau_equator=7.2, tau_pole=1.8, alpha=4.0,
        )
        assert jnp.all(jnp.diff(tau[:, 0]) >= 0.0)

    def test_hand_computed_midlevel(self) -> None:
        """Verify against hand computation at sigma=0.5, equator."""
        sigma_half = jnp.array([0.0, 0.5, 1.0])
        sin_lat = jnp.array([0.0])
        tau = longwave_optical_depth(
            sigma_half, sin_lat, tau_equator=7.2, tau_pole=1.8, alpha=4.0,
        )
        expected = 7.2 * 0.5**4  # = 0.45
        np.testing.assert_allclose(tau[1, 0], expected)


# ---------------------------------------------------------------------------
# Unit tests: longwave heating
# ---------------------------------------------------------------------------


class TestLongwaveHeating:
    """Verify longwave radiative heating."""

    def test_isothermal_with_same_surface_temp_is_small(
        self, levels: SigmaLevels,
    ) -> None:
        """An isothermal atmosphere with T_surface = T_atm should have small LW heating.

        Not exactly zero because optical depth varies with level, but net
        heating should be small compared to typical radiative rates.
        """
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
            tau_equator=7.2,
            tau_pole=1.8,
            alpha=4.0,
        )

        # Heating rates should be small (< 1 K/day = ~1.2e-5 K/s)
        assert jnp.max(jnp.abs(q_lw)) < 1.0e-4

    def test_warm_surface_heats_lower_atmosphere(
        self, levels: SigmaLevels,
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
            tau_equator=7.2,
            tau_pole=1.8,
            alpha=4.0,
        )

        # Bottom level should be heated (warm surface radiates up)
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
            tau_equator=7.2,
            tau_pole=1.8,
            alpha=4.0,
        )
        assert q_lw.shape == (n_levels, n_lat, n_lon)


# ---------------------------------------------------------------------------
# Unit tests: shortwave heating
# ---------------------------------------------------------------------------


class TestShortwaveHeating:
    """Verify shortwave radiative heating."""

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

    def test_equator_greater_than_pole(self, levels: SigmaLevels) -> None:
        """Column-integrated SW heating should be larger at equator than poles."""
        n_lat, n_lon = 4, 8
        sin_lat = jnp.array([0.0, 0.3, 0.6, 0.9])
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
        # Column sum at equator (lat index 0) > column sum at high lat (index 3)
        col_eq = float(jnp.sum(q_sw[:, 0, 0]))
        col_pole = float(jnp.sum(q_sw[:, 3, 0]))
        assert col_eq > col_pole

    def test_shape(self, levels: SigmaLevels) -> None:
        """Output shape should match input grid."""
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
            sw_tau_0=0.22,
            sw_exponent=2.0,
            delta_s=1.4,
        )
        assert q_sw.shape == (levels.n_levels, n_lat, n_lon)

    def test_zero_optical_depth_gives_zero_heating(
        self, levels: SigmaLevels,
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

        # Stable: T increases downward faster than dry adiabat
        # theta = T * sigma^(-kappa); for stability, theta must increase upward
        # Set T so that theta is uniform (neutral) then add stability
        theta_profile = jnp.array([310.0, 300.0, 290.0, 280.0, 270.0])
        t_profile = theta_profile * sigma_full**kappa
        temperature = jnp.broadcast_to(
            t_profile[:, None, None], (5, n_lat, n_lon),
        ).copy()

        t_adjusted = dry_convective_adjustment(
            temperature, sigma_full, dsigma, kappa, n_iterations=3,
        )
        np.testing.assert_allclose(t_adjusted, temperature, atol=1e-10)

    def test_unstable_pair_is_neutralized(self) -> None:
        """A single unstable pair should be mixed to equal theta."""
        sigma_full = jnp.array([0.25, 0.75])
        dsigma = jnp.array([0.5, 0.5])
        kappa = EARTH.kappa

        # Make unstable: lower theta above than below
        theta_above = 280.0
        theta_below = 300.0
        t_above = theta_above * sigma_full[0] ** kappa
        t_below = theta_below * sigma_full[1] ** kappa
        temperature = jnp.array([[[t_above]], [[t_below]]])

        t_adjusted = dry_convective_adjustment(
            temperature, sigma_full, dsigma, kappa, n_iterations=1,
        )

        # After adjustment, thetas should be equal
        theta_adj_above = float(t_adjusted[0, 0, 0]) / sigma_full[0] ** kappa
        theta_adj_below = float(t_adjusted[1, 0, 0]) / sigma_full[1] ** kappa
        np.testing.assert_allclose(theta_adj_above, theta_adj_below, rtol=1e-10)

    def test_enthalpy_conserved(self) -> None:
        """Column enthalpy should be conserved by adjustment."""
        sigma_full = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dsigma = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        kappa = EARTH.kappa
        n_lat, n_lon = 2, 4

        # Fully unstable: theta increases downward
        theta_profile = jnp.array([250.0, 260.0, 270.0, 280.0, 290.0])
        t_profile = theta_profile * sigma_full**kappa
        temperature = jnp.broadcast_to(
            t_profile[:, None, None], (5, n_lat, n_lon),
        ).copy()

        t_adjusted = dry_convective_adjustment(
            temperature, sigma_full, dsigma, kappa, n_iterations=5,
        )

        # Column enthalpy: sum(dsigma * T) should be conserved
        h_before = jnp.sum(dsigma[:, None, None] * temperature, axis=0)
        h_after = jnp.sum(dsigma[:, None, None] * t_adjusted, axis=0)
        np.testing.assert_allclose(h_after, h_before, rtol=1e-10)

    def test_fully_unstable_column_becomes_stable(self) -> None:
        """A fully unstable column should become neutrally stable after adjustment."""
        sigma_full = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dsigma = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        kappa = EARTH.kappa

        # theta increasing downward (unstable)
        theta_profile = jnp.array([250.0, 260.0, 270.0, 280.0, 290.0])
        t_profile = theta_profile * sigma_full**kappa
        temperature = t_profile[:, None, None]  # (5, 1, 1)

        t_adjusted = dry_convective_adjustment(
            temperature, sigma_full, dsigma, kappa, n_iterations=50,
        )

        # Check all adjacent pairs have theta_above >= theta_below
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
        """Rayleigh drag should be zero above the boundary layer (sigma < sigma_b)."""
        state, ps_grid = isothermal_state

        # Set nonzero vorticity to detect drag
        state = state.replace(
            vorticity=jnp.ones_like(state.vorticity) * (1.0 + 0j),
        )
        tendencies = sp_forcing(state, ps_grid)

        sigma_b = sp_forcing.config.sigma_b
        for k in range(levels.n_levels):
            if float(levels.sigma_full[k]) < sigma_b:
                np.testing.assert_allclose(
                    tendencies.vorticity[k], 0.0, atol=1e-30,
                    err_msg=f"Drag nonzero at level {k} (sigma={float(levels.sigma_full[k]):.2f})",
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

    @pytest.fixture
    def run_10_days(
        self,
        t21_transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> dict[str, float]:
        """Run a 10-day integration at T21 L20 and return diagnostics."""
        dt = 1200.0
        n_steps = int(10 * 86400 / dt)  # 10 days

        state, ref_temps, surface_phi = simple_physics_initial_state(
            t21_transform, EARTH, levels,
            initial_temperature=264.0,
            perturbation_amplitude=1.0,
            seed=42,
        )

        forcing = SimplePhysics(t21_transform, EARTH, levels)

        spectral_filter = exponential_filter(t21_transform.arrays, dt)

        init_fn, step_fn = build_pe_stepper(
            t21_transform, EARTH, levels,
            ref_temps, surface_phi,
            dt=dt,
            forcing=forcing,
            spectral_filter=spectral_filter,
        )

        step_fn = jax.jit(step_fn)
        prev, curr = init_fn(state)

        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        # Compute diagnostics
        t_grid = jax.vmap(t21_transform.spectral_to_grid)(curr.temperature)
        t_min = float(jnp.min(t_grid))
        t_max = float(jnp.max(t_grid))
        t_mean = float(jnp.mean(t_grid))

        lnps_grid = t21_transform.spectral_to_grid(curr.log_surface_pressure)
        ps_grid = EARTH.reference_pressure * jnp.exp(lnps_grid)
        ps_min = float(jnp.min(ps_grid))
        ps_max = float(jnp.max(ps_grid))

        return {
            "t_min": t_min,
            "t_max": t_max,
            "t_mean": t_mean,
            "ps_min": ps_min,
            "ps_max": ps_max,
        }

    def test_temperature_bounded(self, run_10_days: dict[str, float]) -> None:
        """Temperature should remain within physical bounds after 10 days."""
        assert run_10_days["t_min"] > 150.0, (
            f"Temperature too cold: {run_10_days['t_min']:.1f} K"
        )
        assert run_10_days["t_max"] < 400.0, (
            f"Temperature too hot: {run_10_days['t_max']:.1f} K"
        )

    def test_temperature_drifts_from_initial(
        self, run_10_days: dict[str, float],
    ) -> None:
        """Mean temperature should drift from isothermal initial condition."""
        assert abs(run_10_days["t_mean"] - 264.0) > 0.1, (
            "Temperature should evolve from initial state"
        )

    def test_surface_pressure_bounded(
        self, run_10_days: dict[str, float],
    ) -> None:
        """Surface pressure should remain near 1e5 Pa after 10 days."""
        assert run_10_days["ps_min"] > 0.8e5
        assert run_10_days["ps_max"] < 1.2e5
