"""Tests for the Jablonowski-Williamson baroclinic instability test case.

Validates:
1. Analytic initial conditions match the paper's formulas
2. Grid-point fields match independent analytic evaluation
3. Steady state remains stationary under integration
4. Perturbation triggers baroclinic wave development
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import (
    JWConfig,
    _reference_geopotential,
    _reference_temperature,
    jablonowski_williamson_perturbation,
    jablonowski_williamson_steady_state,
)
from notus.operators import exponential_filter
from notus.vertical.sigma import uniform_sigma_levels
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform


# Enable float64 for precision
jax.config.update("jax_enable_x64", True)


def _notus_to_grid(
    transform: SpectralTransform,
    spectral_field: jnp.ndarray,
) -> np.ndarray:
    """Convert spectral field(s) to grid points.

    Handles both 1D (n_spectral,) and 2D (n_levels, n_spectral) inputs.
    """
    if spectral_field.ndim == 1:
        return np.asarray(transform.spectral_to_grid(spectral_field))
    result = np.zeros((spectral_field.shape[0], transform.grid.n_lat, transform.grid.n_lon))
    for k in range(spectral_field.shape[0]):
        result[k] = np.asarray(transform.spectral_to_grid(spectral_field[k]))
    return result


# =====================================================================
# Unit tests for analytic formulas
# =====================================================================


class TestAnalyticFormulas:
    """Verify reference profiles against known values from J-W 2006."""

    def test_reference_temperature_troposphere(self) -> None:
        """T_ref in troposphere (eta >= eta_tropo) follows simple power law."""
        config = JWConfig()
        eta = 0.5  # well within troposphere
        t_ref = _reference_temperature(eta, EARTH, config)
        exponent = EARTH.gas_constant * config.gamma / EARTH.gravity
        expected = config.t0 * eta**exponent
        np.testing.assert_allclose(t_ref, expected, rtol=1e-14)

    def test_reference_temperature_stratosphere(self) -> None:
        """T_ref in stratosphere (eta < eta_tropo) has additional delta_t term."""
        config = JWConfig()
        eta = 0.1  # above tropopause
        t_ref = _reference_temperature(eta, EARTH, config)
        exponent = EARTH.gas_constant * config.gamma / EARTH.gravity
        t_mean = config.t0 * eta**exponent
        # Should have extra term
        assert t_ref > t_mean
        expected = t_mean + config.delta_t * (config.eta_tropo - eta) ** 5
        np.testing.assert_allclose(t_ref, expected, rtol=1e-14)

    def test_reference_temperature_at_tropopause(self) -> None:
        """T_ref should be continuous at the tropopause."""
        config = JWConfig()
        t_above = _reference_temperature(config.eta_tropo - 1e-10, EARTH, config)
        t_at = _reference_temperature(config.eta_tropo, EARTH, config)
        np.testing.assert_allclose(t_above, t_at, rtol=1e-6)

    def test_reference_geopotential_surface(self) -> None:
        """Phi_ref at surface (eta=1) should be zero (z=0 at surface)."""
        config = JWConfig()
        phi = _reference_geopotential(1.0, EARTH, config)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_reference_geopotential_tropopause_continuity(self) -> None:
        """Phi_ref should be continuous at the tropopause."""
        config = JWConfig()
        phi_above = _reference_geopotential(config.eta_tropo - 1e-10, EARTH, config)
        phi_at = _reference_geopotential(config.eta_tropo, EARTH, config)
        np.testing.assert_allclose(phi_above, phi_at, rtol=1e-5)

    def test_reference_temperature_profile_is_physical(self) -> None:
        """Temperature should decrease with height in troposphere."""
        config = JWConfig()
        etas = np.linspace(0.3, 1.0, 20)
        temps = [_reference_temperature(float(e), EARTH, config) for e in etas]
        # Temperature should increase toward surface in troposphere
        for i in range(len(temps) - 1):
            assert temps[i] < temps[i + 1], (
                f"T_ref should increase with eta: T({etas[i]:.2f})={temps[i]:.1f} "
                f">= T({etas[i + 1]:.2f})={temps[i + 1]:.1f}"
            )


# =====================================================================
# Grid-point field validation against analytic formulas
# =====================================================================


class TestGridPointFields:
    """Verify that spectral→grid fields match the analytic J-W expressions."""

    def test_vorticity_matches_analytic(self) -> None:
        """Grid-point vorticity should match ζ(lat, η) after spectral roundtrip.

        The analytic field is evaluated on the grid, transformed to spectral
        (truncated at T21), and back.  This tests that our implementation
        evaluates the J-W formula correctly.
        """
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        config = JWConfig()

        state, _, _ = jablonowski_williamson_steady_state(transform, EARTH, levels, config)
        vort_grid = _notus_to_grid(transform, state.vorticity)

        lat = np.array(grid.latitudes)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        etas = np.array(levels.sigma_full)

        for k, eta in enumerate(etas):
            eta_nu = (float(eta) - config.eta0) * np.pi / 2.0
            cos_enu = np.cos(eta_nu)
            analytic = (
                (-4.0 * config.u0 / EARTH.radius)
                * cos_enu**1.5
                * sin_lat
                * cos_lat
                * (2.0 - 5.0 * sin_lat**2)
            )
            # Apply same spectral roundtrip to the analytic field
            analytic_2d = np.broadcast_to(analytic[:, None], (grid.n_lat, grid.n_lon))
            expected = np.asarray(
                transform.spectral_to_grid(transform.grid_to_spectral(jnp.array(analytic_2d)))
            )
            np.testing.assert_allclose(
                vort_grid[k],
                expected,
                rtol=1e-12,
                err_msg=f"Vorticity mismatch at level {k} (eta={eta:.2f})",
            )

    def test_temperature_matches_analytic(self) -> None:
        """Grid-point temperature should match T_ref + T'(lat, η) after roundtrip."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        config = JWConfig()

        state, ref_temps, _ = jablonowski_williamson_steady_state(transform, EARTH, levels, config)
        temp_grid = _notus_to_grid(transform, state.temperature)

        lat = np.array(grid.latitudes)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        etas = np.array(levels.sigma_full)

        a_lat = -2.0 * sin_lat**6 * (cos_lat**2 + 1.0 / 3.0) + 10.0 / 63.0
        b_lat = 1.6 * cos_lat**3 * (sin_lat**2 + 2.0 / 3.0) - np.pi / 4.0

        for k, eta in enumerate(etas):
            eta_nu = (float(eta) - config.eta0) * np.pi / 2.0
            cos_enu = np.cos(eta_nu)
            sin_enu = np.sin(eta_nu)
            t_var = (
                0.75
                * (float(eta) * np.pi * config.u0 / EARTH.gas_constant)
                * sin_enu
                * np.sqrt(cos_enu)
                * (
                    a_lat * 2.0 * config.u0 * cos_enu**1.5
                    + b_lat * EARTH.radius * EARTH.rotation_rate
                )
            )
            analytic = ref_temps[k] + t_var
            analytic_2d = np.broadcast_to(analytic[:, None], (grid.n_lat, grid.n_lon))
            expected = np.asarray(
                transform.spectral_to_grid(transform.grid_to_spectral(jnp.array(analytic_2d)))
            )
            np.testing.assert_allclose(
                temp_grid[k],
                expected,
                rtol=1e-12,
                err_msg=f"Temperature mismatch at level {k} (eta={eta:.2f})",
            )

    def test_divergence_is_zero(self) -> None:
        """Initial divergence should be identically zero."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        state, _, _ = jablonowski_williamson_steady_state(transform, EARTH, levels)
        np.testing.assert_allclose(np.asarray(state.divergence), 0.0, atol=1e-30)

    def test_log_surface_pressure_is_zero(self) -> None:
        """Initial ln(ps/p0) should be zero (uniform surface pressure)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        state, _, _ = jablonowski_williamson_steady_state(transform, EARTH, levels)
        np.testing.assert_allclose(np.asarray(state.log_surface_pressure), 0.0, atol=1e-30)

    def test_vorticity_is_zonally_symmetric(self) -> None:
        """Steady-state vorticity should have no longitude dependence (m=0 only)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        state, _, _ = jablonowski_williamson_steady_state(transform, EARTH, levels)
        vort_grid = _notus_to_grid(transform, state.vorticity)
        for k in range(levels.n_levels):
            zonal_std = np.std(vort_grid[k], axis=1)  # std along lon
            np.testing.assert_allclose(
                zonal_std,
                0.0,
                atol=1e-15,
                err_msg=f"Vorticity should be zonally symmetric at level {k}",
            )

    def test_surface_geopotential_nonzero(self) -> None:
        """Surface geopotential Φ_s should be nonzero (balanced orography)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        _, _, surface_phi = jablonowski_williamson_steady_state(transform, EARTH, levels)
        phi_grid = np.asarray(transform.spectral_to_grid(surface_phi))
        # Should have latitude-dependent structure
        assert np.max(np.abs(phi_grid)) > 100.0, (
            "Surface geopotential should have significant magnitude"
        )

    def test_reference_temperatures_physical_range(self) -> None:
        """Reference temperatures should be in a physically reasonable range."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(20)
        _, ref_temps, _ = jablonowski_williamson_steady_state(transform, EARTH, levels)
        assert np.all(ref_temps > 180.0), "T_ref should be > 180 K everywhere"
        assert np.all(ref_temps < 300.0), "T_ref should be < 300 K everywhere"
        # Should decrease with height (increasing index = increasing eta = lower)
        assert ref_temps[0] < ref_temps[-1], "T_ref should be colder at top than surface"

    def test_perturbation_is_localized(self) -> None:
        """Perturbation vorticity should be localized near (lon_c, lat_c)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        config_lon = np.pi / 9.0
        config_lat = 2.0 * np.pi / 9.0

        pert = jablonowski_williamson_perturbation(transform, EARTH, levels)
        vort_grid = _notus_to_grid(transform, pert.vorticity)

        # Find the grid point closest to the perturbation center
        lon = np.array(grid.longitudes)
        lat = np.array(grid.latitudes)
        j_lat = int(np.argmin(np.abs(lat - config_lat)))
        i_lon = int(np.argmin(np.abs(lon - config_lon)))

        # Peak should be near the perturbation center
        peak_idx = np.unravel_index(np.argmax(np.abs(vort_grid[0])), vort_grid[0].shape)
        assert abs(peak_idx[0] - j_lat) <= 2, (
            f"Vorticity peak lat index {peak_idx[0]} too far from center {j_lat}"
        )
        assert abs(peak_idx[1] - i_lon) <= 2, (
            f"Vorticity peak lon index {peak_idx[1]} too far from center {i_lon}"
        )

    def test_perturbation_is_level_independent(self) -> None:
        """Perturbation should be the same at all levels."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)

        pert = jablonowski_williamson_perturbation(transform, EARTH, levels)
        vort_grid = _notus_to_grid(transform, pert.vorticity)
        for k in range(1, levels.n_levels):
            np.testing.assert_allclose(
                vort_grid[k],
                vort_grid[0],
                atol=1e-30,
                err_msg=f"Perturbation should be identical at all levels (k={k})",
            )


# =====================================================================
# Stationarity test — balanced state should remain balanced
# =====================================================================


class TestStationarity:
    """The J-W balanced state should remain stationary under integration."""

    def test_steady_state_stationary_6_hours(self) -> None:
        """Balanced J-W state at T21L20 should stay nearly unchanged for 6 hours.

        Uses 600s timestep, 36 steps = 6 hours.  All fields should remain
        close to their initial values.  20 levels properly resolves the
        tropopause at η ≈ 0.2.
        """
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(20)
        dt = 600.0
        n_steps = 36  # 6 hours

        state, ref_temps, surface_phi = jablonowski_williamson_steady_state(
            transform, EARTH, levels
        )

        filt = exponential_filter(grid.truncation, dt)
        init_fn, step_fn = build_pe_stepper(
            transform=transform,
            planet=EARTH,
            levels=levels,
            reference_temperature=ref_temps,
            surface_geopotential=surface_phi,
            dt=dt,
            spectral_filter=filt,
        )

        # Store initial fields for comparison
        init_vort = np.asarray(state.vorticity)
        init_temp = np.asarray(state.temperature)

        prev, curr = init_fn(state)
        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        # Divergence should remain near zero
        div_max = float(jnp.max(jnp.abs(curr.divergence)))
        assert div_max < 1e-3, f"Divergence grew too large: {div_max:.2e}"

        # Vorticity should stay close to initial
        np.testing.assert_allclose(
            np.asarray(curr.vorticity),
            init_vort,
            atol=5e-4,
            err_msg="Vorticity drifted from initial balanced state",
        )

        # Temperature should stay close to initial
        np.testing.assert_allclose(
            np.asarray(curr.temperature),
            init_temp,
            atol=5e-2,
            err_msg="Temperature drifted from initial balanced state",
        )

        # Log surface pressure should remain near zero
        lnps_max = float(jnp.max(jnp.abs(curr.log_surface_pressure)))
        assert lnps_max < 1e-4, f"ln(ps) grew too large: {lnps_max:.2e}"


# =====================================================================
# Baroclinic wave development
# =====================================================================


class TestBaroclinicWave:
    """Test that the perturbation triggers realistic baroclinic instability."""

    def test_perturbation_breaks_symmetry(self) -> None:
        """The perturbation should introduce longitude-dependent structure."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)

        pert = jablonowski_williamson_perturbation(transform, EARTH, levels)

        # Should have nonzero m>0 modes (longitude dependence)
        m1_idx = grid.spectral_index(1, 1)
        assert float(jnp.abs(pert.vorticity[0, m1_idx])) > 0, (
            "Perturbation should have m>0 spectral modes"
        )

    def test_perturbed_state_develops_baroclinic_wave(self) -> None:
        """Perturbed state at T21L5 should show growing disturbance after 1 day.

        The surface pressure field should develop structure beyond the
        initial perturbation, indicating baroclinic wave growth.
        """
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(5)
        dt = 600.0
        n_steps = 144  # 1 day

        state, ref_temps, surface_phi = jablonowski_williamson_steady_state(
            transform, EARTH, levels
        )
        pert = jablonowski_williamson_perturbation(transform, EARTH, levels)

        # Add perturbation to steady state
        perturbed = jax.tree.map(jnp.add, state, pert)

        filt = exponential_filter(grid.truncation, dt)
        init_fn, step_fn = build_pe_stepper(
            transform=transform,
            planet=EARTH,
            levels=levels,
            reference_temperature=ref_temps,
            surface_geopotential=surface_phi,
            dt=dt,
            spectral_filter=filt,
        )

        prev, curr = init_fn(perturbed)
        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        # The integration should remain stable (no NaN/Inf)
        for field_name in [
            "vorticity",
            "divergence",
            "temperature",
            "log_surface_pressure",
        ]:
            field = getattr(curr, field_name)
            assert jnp.all(jnp.isfinite(field)), (
                f"{field_name} contains non-finite values after 1 day"
            )

        # Divergence should have grown from the initial perturbation
        # (baroclinic instability amplifies the disturbance)
        div_max = float(jnp.max(jnp.abs(curr.divergence)))
        assert div_max > 1e-10, f"Divergence should grow from perturbation, got max={div_max:.2e}"

        # Surface pressure should show some structure
        lnps_max = float(jnp.max(jnp.abs(curr.log_surface_pressure)))
        assert lnps_max > 1e-12, (
            f"Surface pressure should develop structure, got max(|lnps|)={lnps_max:.2e}"
        )
