"""Validation tests for dynamics over orography.

Tests that the model remains stable and conserves quantities when
integrating over idealized mountain topography.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus import (
    EARTH,
    GaussianGrid,
    PhysicsSuite,
    PhysicsSuiteConfig,
    SpectralTransform,
    build_pe_stepper,
    compute_conservation_diagnostics,
    exponential_filter,
    gaussian_mountain,
    grid_surface_pressure,
    held_suarez_initial_state,
    moist_aquaplanet_initial_state,
    orographic_log_surface_pressure,
    sinusoidal_mountains,
    smooth_orography,
    uniform_sigma_levels,
)
from notus.physics.surface import diagnose_precipitation


jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRUNCATION = 21
_N_LEVELS = 20
_DT = 600.0


def _mountain_initial_state(
    surface_geopotential: jnp.ndarray,
    transform: SpectralTransform,
    levels,
    ref_temp: float = 264.0,
):
    """Build a rest-state IC with orographic surface pressure adjustment."""
    state, ref_temps, _ = held_suarez_initial_state(
        transform,
        EARTH,
        levels,
        initial_temperature=ref_temp,
        perturbation_amplitude=0.0,
    )
    # Replace flat ln(ps) with hydrostatically adjusted field
    ln_ps = orographic_log_surface_pressure(surface_geopotential, transform, EARTH, ref_temp)
    state = state.replace(log_surface_pressure=ln_ps)
    return state, ref_temps


def _run_mountain_integration(
    surface_geopotential: jnp.ndarray,
    n_days: int = 5,
):
    """Run dry dynamics over a mountain for n_days, return daily diagnostics."""
    grid = GaussianGrid(truncation=_TRUNCATION)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(_N_LEVELS)

    state, ref_temps = _mountain_initial_state(surface_geopotential, transform, levels)

    filt = exponential_filter(transform.arrays, _DT)
    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_geopotential,
        dt=_DT,
        spectral_filter=filt,
    )

    prev, curr, _diags = init_fn(state)
    steps_per_day = int(86400 / _DT)

    daily_diags = []
    for _day in range(n_days):
        for _ in range(steps_per_day):
            prev, curr, _diags = step_fn(prev, curr)

        diag = compute_conservation_diagnostics(
            curr, transform, EARTH, levels, surface_geopotential
        )
        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
        ps_grid = np.asarray(grid_surface_pressure(curr, transform, EARTH))

        daily_diags.append({
            "mass": diag.mass,
            "total_energy": diag.total_energy,
            "t_min": float(t_grid.min()),
            "t_max": float(t_grid.max()),
            "ps_min": float(ps_grid.min()),
            "ps_max": float(ps_grid.max()),
        })

    return daily_diags


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGaussianMountainIntegration:
    """Integration tests with an isolated Gaussian mountain."""

    @pytest.fixture(scope="class")
    def results(self):
        grid = GaussianGrid(truncation=_TRUNCATION)
        transform = SpectralTransform(grid, EARTH.radius)
        phi_s = gaussian_mountain(transform, EARTH, height=2000.0, half_width=np.pi / 9)
        phi_s = smooth_orography(phi_s, transform.arrays, method="lanczos")
        return _run_mountain_integration(phi_s, n_days=5)

    def test_stable_5_days(self, results):
        """Model should not blow up over 5 days."""
        for day, d in enumerate(results):
            assert np.isfinite(d["t_min"]), f"Day {day + 1}: T_min not finite"
            assert np.isfinite(d["t_max"]), f"Day {day + 1}: T_max not finite"
            assert np.isfinite(d["mass"]), f"Day {day + 1}: mass not finite"

    def test_temperatures_bounded(self, results):
        """Temperatures should stay physically plausible."""
        for day, d in enumerate(results):
            assert d["t_min"] > 150.0, f"Day {day + 1}: T_min={d['t_min']:.1f} K too cold"
            assert d["t_max"] < 350.0, f"Day {day + 1}: T_max={d['t_max']:.1f} K too hot"

    def test_surface_pressure_reduced_over_mountain(self, results):
        """Surface pressure minimum should be below p₀ (mountain effect)."""
        for d in results:
            assert d["ps_min"] < EARTH.reference_pressure

    def test_mass_conservation(self, results):
        """Mass should be conserved to within 0.01% over 5 days."""
        m0 = results[0]["mass"]
        for day, d in enumerate(results):
            rel_change = abs(d["mass"] - m0) / m0
            assert rel_change < 1e-4, f"Day {day + 1}: mass drift {rel_change:.2e} exceeds 0.01%"

    def test_energy_bounded(self, results):
        """Total energy should not drift wildly (within 1% of initial)."""
        e0 = results[0]["total_energy"]
        for day, d in enumerate(results):
            rel_change = abs(d["total_energy"] - e0) / abs(e0)
            assert rel_change < 0.01, f"Day {day + 1}: energy drift {rel_change:.2e} exceeds 1%"


@pytest.mark.slow
class TestSinusoidalMountainIntegration:
    """Integration tests with wavenumber-2 sinusoidal mountains."""

    @pytest.fixture(scope="class")
    def results(self):
        grid = GaussianGrid(truncation=_TRUNCATION)
        transform = SpectralTransform(grid, EARTH.radius)
        phi_s = sinusoidal_mountains(transform, EARTH, height=2000.0, zonal_wavenumber=2)
        phi_s = smooth_orography(phi_s, transform.arrays, method="lanczos")
        return _run_mountain_integration(phi_s, n_days=5)

    def test_stable_5_days(self, results):
        """Model should not blow up over 5 days."""
        for day, d in enumerate(results):
            assert np.isfinite(d["t_min"]), f"Day {day + 1}: T_min not finite"
            assert np.isfinite(d["t_max"]), f"Day {day + 1}: T_max not finite"

    def test_temperatures_bounded(self, results):
        """Temperatures should stay physically plausible."""
        for day, d in enumerate(results):
            assert d["t_min"] > 150.0, f"Day {day + 1}: T_min={d['t_min']:.1f} K too cold"
            assert d["t_max"] < 350.0, f"Day {day + 1}: T_max={d['t_max']:.1f} K too hot"

    def test_mass_conservation(self, results):
        """Mass should be conserved to within 0.01%."""
        m0 = results[0]["mass"]
        for day, d in enumerate(results):
            rel_change = abs(d["mass"] - m0) / m0
            assert rel_change < 1e-4, f"Day {day + 1}: mass drift {rel_change:.2e} exceeds 0.01%"


def _run_moist_mountain_integration(
    surface_geopotential: jnp.ndarray,
    n_days: int = 10,
    dt: float = 600.0,
):
    """Run moist dynamics+physics over a mountain, return daily precipitation maps."""
    grid = GaussianGrid(truncation=_TRUNCATION)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(_N_LEVELS)

    # Moist initial state with orographic pressure adjustment
    ref_temp = 264.0
    state, ref_temps, _ = moist_aquaplanet_initial_state(
        transform, EARTH, levels, initial_temperature=ref_temp, perturbation_amplitude=1.0
    )
    ln_ps = orographic_log_surface_pressure(surface_geopotential, transform, EARTH, ref_temp)
    state = state.replace(log_surface_pressure=ln_ps)

    config = PhysicsSuiteConfig()
    physics = PhysicsSuite(transform, EARTH, levels, config)
    filt = exponential_filter(transform.arrays, dt)

    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_geopotential,
        dt=dt,
        spectral_filter=filt,
        forcing=physics,
    )

    prev, curr, _diags = init_fn(state)
    steps_per_day = int(86400 / dt)

    daily_data: list[dict] = []
    for _day in range(n_days):
        for _ in range(steps_per_day):
            prev, curr, _diags = step_fn(prev, curr)

        # Diagnose precipitation from the current state
        t_grid = jax.vmap(transform.spectral_to_grid)(curr.temperature)
        q_grid = jax.vmap(transform.spectral_to_grid)(curr.humidity)
        ps_grid = EARTH.reference_pressure * jnp.exp(
            transform.spectral_to_grid(curr.log_surface_pressure)
        )
        sigma_full = jnp.array(levels.sigma_full)
        pressure = sigma_full[:, None, None] * ps_grid[None, :, :]

        precip = diagnose_precipitation(
            t_grid,
            q_grid,
            pressure,
            levels.dsigma,
            ps_grid,
            gravity=EARTH.gravity,
            epsilon=EARTH.epsilon_moisture,
            latent_heat=EARTH.latent_heat_vaporization,
            specific_heat_cp=EARTH.specific_heat_cp,
            gas_constant=EARTH.gas_constant,
        )

        daily_data.append({
            "precip": np.asarray(precip),
            "t_min": float(np.asarray(t_grid).min()),
            "t_max": float(np.asarray(t_grid).max()),
            "ps_grid": np.asarray(ps_grid),
        })

    return daily_data, grid, transform


@pytest.mark.slow
class TestMoistMountainIntegration:
    """Moist integration over a mountain to verify orographic precipitation."""

    @pytest.fixture(scope="class")
    def results(self):
        grid = GaussianGrid(truncation=_TRUNCATION)
        transform = SpectralTransform(grid, EARTH.radius)
        phi_s = gaussian_mountain(
            transform, EARTH, height=2500.0, center_lat=np.pi / 6, half_width=np.pi / 9
        )
        phi_s = smooth_orography(phi_s, transform.arrays, method="lanczos")
        return _run_moist_mountain_integration(phi_s, n_days=10)

    def test_stable_10_days(self, results):
        """Moist model with mountain should not blow up over 10 days."""
        daily_data, _, _ = results
        for day, d in enumerate(daily_data):
            assert np.isfinite(d["t_min"]), f"Day {day + 1}: T_min not finite"
            assert np.isfinite(d["t_max"]), f"Day {day + 1}: T_max not finite"

    def test_temperatures_bounded(self, results):
        """Temperatures should stay physically plausible."""
        daily_data, _, _ = results
        for day, d in enumerate(daily_data):
            assert d["t_min"] > 150.0, f"Day {day + 1}: T_min={d['t_min']:.1f} K too cold"
            assert d["t_max"] < 350.0, f"Day {day + 1}: T_max={d['t_max']:.1f} K too hot"

    def test_precipitation_develops(self, results):
        """Some precipitation should develop over the 10-day integration."""
        daily_data, _, _ = results
        final_precip = daily_data[-1]["precip"]
        assert final_precip.max() > 0.0, "No precipitation developed"

    def test_precipitation_non_negative(self, results):
        """Precipitation should always be non-negative."""
        daily_data, _, _ = results
        for day, d in enumerate(daily_data):
            assert d["precip"].min() >= 0.0, f"Day {day + 1}: negative precipitation"

    def test_surface_pressure_reduced_over_mountain(self, results):
        """Surface pressure should be lower over the mountain."""
        daily_data, _, _ = results
        ps = daily_data[-1]["ps_grid"]
        assert ps.min() < EARTH.reference_pressure

    def test_mountain_breaks_zonal_symmetry_in_precip(self, results):
        """Precipitation near the mountain should not be zonally uniform.

        The mountain forces spatially varying vertical motion, breaking
        the zonal symmetry that an aquaplanet would otherwise have.
        """
        daily_data, grid, _ = results
        precip = daily_data[-1]["precip"]
        lat = np.asarray(grid.latitudes)
        # Find latitude band near the mountain (within ~15°)
        center_lat = np.pi / 6
        lat_band = np.abs(lat - center_lat) < np.radians(15)
        precip_band = precip[lat_band, :]
        # Zonal standard deviation should be nonzero
        zonal_std = np.std(precip_band, axis=1)
        assert zonal_std.max() > 0.0, "Precipitation is zonally uniform despite mountain"


# ---------------------------------------------------------------------------
# Sharp terrain stability tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSharpTerrainStability:
    """Stability tests with steep, narrow mountains.

    Sharp topography stresses the spectral representation through Gibbs
    ringing and large pressure-gradient forcing.  These tests verify
    that the model stays stable with physically extreme but resolvable
    terrain, and that spectral smoothing is sufficient to tame Gibbs
    artefacts.
    """

    @pytest.fixture(scope="class")
    def steep_mountain_smoothed(self):
        """4 km peak with half-width ~5° — steep but smoothed."""
        grid = GaussianGrid(truncation=_TRUNCATION)
        transform = SpectralTransform(grid, EARTH.radius)
        phi_s = gaussian_mountain(transform, EARTH, height=4000.0, half_width=np.pi / 36)
        phi_s = smooth_orography(phi_s, transform.arrays, method="lanczos", order=2)
        return _run_mountain_integration(phi_s, n_days=5)

    @pytest.fixture(scope="class")
    def steep_mountain_unsmoothed(self):
        """4 km peak with half-width ~5° — no spectral smoothing."""
        grid = GaussianGrid(truncation=_TRUNCATION)
        transform = SpectralTransform(grid, EARTH.radius)
        phi_s = gaussian_mountain(transform, EARTH, height=4000.0, half_width=np.pi / 36)
        return _run_mountain_integration(phi_s, n_days=5)

    # -- smoothed steep mountain ------------------------------------------------

    def test_smoothed_stable_5_days(self, steep_mountain_smoothed):
        """Smoothed steep mountain should remain stable for 5 days."""
        for day, d in enumerate(steep_mountain_smoothed):
            assert np.isfinite(d["t_min"]), f"Day {day + 1}: T_min not finite"
            assert np.isfinite(d["t_max"]), f"Day {day + 1}: T_max not finite"
            assert np.isfinite(d["mass"]), f"Day {day + 1}: mass not finite"

    def test_smoothed_temperatures_bounded(self, steep_mountain_smoothed):
        """Temperatures should stay within plausible bounds."""
        for day, d in enumerate(steep_mountain_smoothed):
            assert d["t_min"] > 150.0, f"Day {day + 1}: T_min={d['t_min']:.1f} K too cold"
            assert d["t_max"] < 350.0, f"Day {day + 1}: T_max={d['t_max']:.1f} K too hot"

    def test_smoothed_mass_conservation(self, steep_mountain_smoothed):
        """Mass should be conserved to within 0.01% even with steep terrain."""
        m0 = steep_mountain_smoothed[0]["mass"]
        for day, d in enumerate(steep_mountain_smoothed):
            rel_change = abs(d["mass"] - m0) / m0
            assert rel_change < 1e-4, f"Day {day + 1}: mass drift {rel_change:.2e} exceeds 0.01%"

    def test_smoothed_energy_bounded(self, steep_mountain_smoothed):
        """Total energy should not drift more than 2% over steep terrain.

        Steep terrain generates stronger gravity waves that can amplify
        energy drift, so we allow a slightly larger tolerance than the
        moderate-mountain tests.
        """
        e0 = steep_mountain_smoothed[0]["total_energy"]
        for day, d in enumerate(steep_mountain_smoothed):
            rel_change = abs(d["total_energy"] - e0) / abs(e0)
            assert rel_change < 0.02, f"Day {day + 1}: energy drift {rel_change:.2e} exceeds 2%"

    # -- unsmoothed steep mountain (Gibbs ringing stress test) ------------------

    def test_unsmoothed_stable_5_days(self, steep_mountain_unsmoothed):
        """Unsmoothed steep mountain: check whether Gibbs ringing causes blow-up.

        Without spectral smoothing, sharp orography produces ringing that
        can seed growing oscillations.  This test documents whether the
        model survives — a failure here indicates that smoothing is
        essential for sharp terrain and should not be silently omitted.
        """
        for day, d in enumerate(steep_mountain_unsmoothed):
            assert np.isfinite(d["t_min"]), f"Day {day + 1}: T_min not finite"
            assert np.isfinite(d["t_max"]), f"Day {day + 1}: T_max not finite"
            assert np.isfinite(d["mass"]), f"Day {day + 1}: mass not finite"

    def test_unsmoothed_temperatures_bounded(self, steep_mountain_unsmoothed):
        """Unsmoothed steep terrain should still have bounded temperatures."""
        for day, d in enumerate(steep_mountain_unsmoothed):
            assert d["t_min"] > 100.0, (
                f"Day {day + 1}: T_min={d['t_min']:.1f} K — "
                "extreme cold suggests Gibbs-driven instability"
            )
            assert d["t_max"] < 400.0, (
                f"Day {day + 1}: T_max={d['t_max']:.1f} K — "
                "extreme heat suggests Gibbs-driven instability"
            )
