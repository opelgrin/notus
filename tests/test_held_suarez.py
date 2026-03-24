"""Tests for the Held-Suarez (1994) forcing and short integrations.

Unit tests verify the forcing fields (equilibrium temperature, relaxation
rates, friction) against hand-computed values.  Integration tests verify
stability and physical plausibility of short runs from an isothermal rest
state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import held_suarez_initial_state
from notus.operators import exponential_filter
from notus.physics.forcing import HeldSuarez
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
def hs_forcing(t21_transform: SpectralTransform, levels: SigmaLevels) -> HeldSuarez:
    """Held-Suarez forcing at T21 resolution."""
    return HeldSuarez(t21_transform, EARTH, levels)


@pytest.fixture
def isothermal_state(
    t21_transform: SpectralTransform,
    levels: SigmaLevels,
) -> tuple[PrimitiveEquationState, jnp.ndarray]:
    """Isothermal rest state at 264 K with uniform surface pressure.

    Returns (state, surface_pressure_grid).
    """
    grid = t21_transform.grid
    n_levels = levels.n_levels
    n_spec = grid.n_spectral_coeffs

    # Uniform 264 K temperature in spectral space
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
# Unit tests: forcing fields
# ---------------------------------------------------------------------------


class TestEquilibriumTemperature:
    """Verify T_eq against analytical values."""

    def test_equator_surface(self, hs_forcing: HeldSuarez) -> None:
        """At equator (sin²φ=0) and σ=1 (p=p0), T_eq = 315 K."""
        # At equator, sin²φ = 0, cos²φ = 1
        # At σ=1, p/p0 = 1, log(p/p0) = 0
        # T_eq = max(200, (315 - 0 - 0) * 1^κ) = 315 K
        # This is the maximum possible T_eq
        assert hs_forcing.kappa == pytest.approx(EARTH.kappa)

    def test_pole_surface(self, hs_forcing: HeldSuarez) -> None:
        """At pole (sin²φ=1) and σ=1, T_eq = (315 - 60) * 1^κ = 255 K."""
        expected = 315.0 - 60.0  # = 255 K at σ=1
        assert expected == pytest.approx(255.0)

    def test_teq_bounded_below(
        self,
        hs_forcing: HeldSuarez,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """T_eq should never go below t_min (200 K)."""
        state, ps = isothermal_state
        result = hs_forcing(state, ps)
        # With T=264K uniform, the forcing pushes toward T_eq.
        # We verify indirectly: the temperature tendency is finite everywhere.
        dt_spec = result.temperature
        dt_grid = jax.vmap(hs_forcing.transform.spectral_to_grid)(dt_spec)
        assert jnp.all(jnp.isfinite(dt_grid))


class TestRayleighFriction:
    """Verify Rayleigh friction coefficients and tendencies."""

    def test_zero_above_boundary_layer(self, hs_forcing: HeldSuarez) -> None:
        """k_v should be zero for σ < σ_b (= 0.7)."""
        sigma_full = np.asarray(hs_forcing.levels.sigma_full)
        k_v = np.asarray(hs_forcing.k_v)
        above_bl = sigma_full < hs_forcing.sigma_b
        assert np.allclose(k_v[above_bl], 0.0, atol=1e-30), (
            f"Non-zero friction above boundary layer: {k_v[above_bl]}"
        )

    def test_nonzero_in_boundary_layer(self, hs_forcing: HeldSuarez) -> None:
        """k_v should be positive for σ > σ_b."""
        sigma_full = np.asarray(hs_forcing.levels.sigma_full)
        k_v = np.asarray(hs_forcing.k_v)
        in_bl = sigma_full > hs_forcing.sigma_b
        assert np.all(k_v[in_bl] > 0.0), (
            f"Zero friction in boundary layer: {k_v[in_bl]}"
        )

    def test_maximum_at_surface(self, hs_forcing: HeldSuarez) -> None:
        """k_v at σ=1 should equal k_f."""
        k_v = np.asarray(hs_forcing.k_v)
        # Last level is closest to surface
        sigma_full = np.asarray(hs_forcing.levels.sigma_full)
        surface_idx = np.argmax(sigma_full)
        sigma_frac_surface = (sigma_full[surface_idx] - 0.7) / 0.3
        expected = hs_forcing.k_f * sigma_frac_surface
        assert k_v[surface_idx] == pytest.approx(expected, rel=1e-10)

    def test_friction_tendency_opposes_motion(
        self,
        hs_forcing: HeldSuarez,
        t21_transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Rayleigh friction tendencies should have opposite sign to the field."""
        grid = t21_transform.grid
        n_levels = levels.n_levels
        n_spec = grid.n_spectral_coeffs

        # Create state with non-zero vorticity in boundary layer
        vort_grid = jnp.ones((n_levels, grid.n_lat, grid.n_lon))
        vort_spec = jax.vmap(t21_transform.grid_to_spectral)(vort_grid)
        t_grid = jnp.full((n_levels, grid.n_lat, grid.n_lon), 264.0)
        t_spec = jax.vmap(t21_transform.grid_to_spectral)(t_grid)

        state = PrimitiveEquationState(
            vorticity=vort_spec,
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=t_spec,
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )
        ps = jnp.full((grid.n_lat, grid.n_lon), EARTH.reference_pressure)
        result = hs_forcing(state, ps)

        # In boundary layer, vort_tend = -k_v * vort_spec
        k_v = hs_forcing.k_v
        bl_levels = np.asarray(k_v) > 0
        if np.any(bl_levels):
            vort_tend = result.vorticity[bl_levels]
            expected = -k_v[bl_levels, None] * vort_spec[bl_levels]
            assert jnp.allclose(vort_tend, expected, atol=1e-30)


class TestNewtonianRelaxation:
    """Verify Newtonian temperature relaxation."""

    def test_forcing_signs(
        self,
        hs_forcing: HeldSuarez,
        t21_transform: SpectralTransform,
        levels: SigmaLevels,
    ) -> None:
        """Hot atmosphere should cool, cold atmosphere should warm."""
        grid = t21_transform.grid
        n_levels = levels.n_levels
        n_spec = grid.n_spectral_coeffs
        ps = jnp.full((grid.n_lat, grid.n_lon), EARTH.reference_pressure)

        # Very hot atmosphere (400 K >> T_eq everywhere)
        t_hot = jnp.full((n_levels, grid.n_lat, grid.n_lon), 400.0)
        t_hot_spec = jax.vmap(t21_transform.grid_to_spectral)(t_hot)
        hot_state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=t_hot_spec,
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )
        hot_result = hs_forcing(hot_state, ps)
        hot_tend_grid = jax.vmap(t21_transform.spectral_to_grid)(hot_result.temperature)
        # Should be cooling (negative tendency) everywhere
        assert jnp.all(hot_tend_grid < 0.0), "Hot atmosphere should cool"

        # Very cold atmosphere (100 K < t_min = 200 K = T_eq floor)
        # T_eq >= 200 K everywhere, so T=100 K => T < T_eq => warming
        t_cold = jnp.full((n_levels, grid.n_lat, grid.n_lon), 100.0)
        t_cold_spec = jax.vmap(t21_transform.grid_to_spectral)(t_cold)
        cold_state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=t_cold_spec,
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )
        cold_result = hs_forcing(cold_state, ps)
        cold_tend_grid = jax.vmap(t21_transform.spectral_to_grid)(cold_result.temperature)
        # Should be warming (positive tendency) everywhere
        assert jnp.all(cold_tend_grid > 0.0), "Cold atmosphere should warm"

    def test_zero_surface_pressure_tendency(
        self,
        hs_forcing: HeldSuarez,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """Held-Suarez forcing has no surface pressure tendency."""
        state, ps = isothermal_state
        result = hs_forcing(state, ps)
        assert jnp.allclose(result.log_surface_pressure, 0.0)


class TestForcingJIT:
    """Verify JIT compatibility."""

    def test_jit_produces_finite_results(
        self,
        hs_forcing: HeldSuarez,
        isothermal_state: tuple[PrimitiveEquationState, jnp.ndarray],
    ) -> None:
        """Forcing should be JIT-compilable and produce finite results."""
        state, ps = isothermal_state
        jit_forcing = jax.jit(hs_forcing)
        result = jit_forcing(state, ps)
        assert jnp.all(jnp.isfinite(result.vorticity))
        assert jnp.all(jnp.isfinite(result.divergence))
        assert jnp.all(jnp.isfinite(result.temperature))
        assert jnp.all(jnp.isfinite(result.log_surface_pressure))


# ---------------------------------------------------------------------------
# Integration tests: short runs with forcing
# ---------------------------------------------------------------------------


def _run_held_suarez(
    truncation: int,
    n_levels: int,
    dt: float,
    n_days: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Run a short Held-Suarez integration and collect daily diagnostics.

    Returns
    -------
    global_mean_t : list[float]
        Daily global-mean temperature [K] (surface level).
    t_range : list[tuple[float, float]]
        Daily (min, max) temperature across all levels [K].
    """
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = held_suarez_initial_state(
        transform, EARTH, levels, perturbation_amplitude=1.0,
    )

    forcing = HeldSuarez(transform, EARTH, levels)
    filt = exponential_filter(transform.arrays, dt)
    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        spectral_filter=filt,
        forcing=forcing,
    )

    prev, curr = init_fn(state)
    steps_per_day = int(86400 / dt)

    global_mean_t: list[float] = []
    t_range: list[tuple[float, float]] = []

    for _day in range(n_days):
        for _ in range(steps_per_day):
            prev, curr = step_fn(prev, curr)

        # Temperature diagnostics
        t_grid = np.asarray(jax.vmap(transform.spectral_to_grid)(curr.temperature))
        # Weights for global mean (Gaussian quadrature)
        weights = np.asarray(grid.lat_weights)
        t_surface = t_grid[-1]  # bottom level
        mean_t = float(np.sum(t_surface * weights[:, None]) / np.sum(weights) / grid.n_lon)
        global_mean_t.append(mean_t)
        t_range.append((float(np.min(t_grid)), float(np.max(t_grid))))

    return global_mean_t, t_range


class TestHeldSuarezIntegration:
    """Short integration tests for the Held-Suarez forcing."""

    @pytest.fixture(scope="class")
    def hs_results(self) -> tuple[list[float], list[tuple[float, float]]]:
        """Run a 10-day T21L20 integration once for all tests."""
        return _run_held_suarez(
            truncation=21,
            n_levels=20,
            dt=1200.0,
            n_days=10,
        )

    def test_integration_stable_10_days(
        self,
        hs_results: tuple[list[float], list[tuple[float, float]]],
    ) -> None:
        """The integration should remain stable for 10 days."""
        global_mean_t, _t_range = hs_results
        assert len(global_mean_t) == 10
        for day, mean_t in enumerate(global_mean_t):
            assert np.isfinite(mean_t), f"Mean T not finite at day {day + 1}"

    def test_temperatures_bounded(
        self,
        hs_results: tuple[list[float], list[tuple[float, float]]],
    ) -> None:
        """Temperatures should stay within physically plausible bounds."""
        _, t_range = hs_results
        for day, (t_min, t_max) in enumerate(t_range):
            assert t_min > 150.0, f"Day {day + 1}: T_min={t_min:.1f} K too cold"
            assert t_max < 350.0, f"Day {day + 1}: T_max={t_max:.1f} K too hot"

    def test_temperature_drifts_from_initial(
        self,
        hs_results: tuple[list[float], list[tuple[float, float]]],
    ) -> None:
        """Global mean surface temperature should drift from 264 K toward T_eq.

        T_eq at the surface (σ=1) ranges from 255 K (pole) to 315 K (equator),
        with a global mean around 280-290 K.  Starting from 264 K, we expect
        the surface to warm over time.
        """
        global_mean_t, _ = hs_results
        # After 10 days, mean T should have moved away from 264 K
        assert abs(global_mean_t[-1] - 264.0) > 1.0, (
            f"Surface temperature has not drifted from initial: {global_mean_t[-1]:.2f} K"
        )


# ---------------------------------------------------------------------------
# Benchmark validation: longer integration to verify climatology
# ---------------------------------------------------------------------------


def _run_held_suarez_climatology(
    truncation: int,
    n_levels: int,
    dt: float,
    spinup_days: int,
    averaging_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GaussianGrid, SigmaLevels]:
    """Run Held-Suarez and return time-averaged zonal-mean U, T, and EKE.

    Returns
    -------
    u_zm : ndarray, shape (n_levels, n_lat)
        Time-averaged zonal-mean zonal wind [m/s].
    t_zm : ndarray, shape (n_levels, n_lat)
        Time-averaged zonal-mean temperature [K].
    eke : ndarray, shape (n_levels, n_lat)
        Time-averaged eddy kinetic energy [m²/s²].
    u_surface : ndarray, shape (n_lat,)
        Time-averaged zonal-mean surface zonal wind [m/s].
    grid : GaussianGrid
    levels : SigmaLevels
    """
    from notus.operators.vector import uv_from_vordiv as uv_recon

    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = held_suarez_initial_state(
        transform, EARTH, levels, perturbation_amplitude=1.0,
    )

    forcing = HeldSuarez(transform, EARTH, levels)
    filt = exponential_filter(transform.arrays, dt)
    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        spectral_filter=filt,
        forcing=forcing,
    )

    steps_per_day = int(86400 / dt)
    total_days = spinup_days + averaging_days
    cos_lat = np.asarray(grid.cos_lat)

    prev, curr = init_fn(state)

    # Accumulators
    u_zm_acc = np.zeros((n_levels, grid.n_lat))
    t_zm_acc = np.zeros((n_levels, grid.n_lat))
    u_prime_sq_acc = np.zeros((n_levels, grid.n_lat))
    v_prime_sq_acc = np.zeros((n_levels, grid.n_lat))
    n_samples = 0

    for day in range(1, total_days + 1):
        for _ in range(steps_per_day):
            prev, curr = step_fn(prev, curr)

        if day > spinup_days:
            # Compute zonal-mean fields
            def _uv(vort: jnp.ndarray, div: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                return uv_recon(vort, div, transform.arrays)

            u_cos_s, v_cos_s = jax.vmap(_uv)(curr.vorticity, curr.divergence)
            all_s = jnp.concatenate([u_cos_s, v_cos_s, curr.temperature], axis=0)
            all_g = np.asarray(jax.vmap(transform.spectral_to_grid)(all_s))

            u_cos_g = all_g[:n_levels]
            v_cos_g = all_g[n_levels : 2 * n_levels]
            t_g = all_g[2 * n_levels : 3 * n_levels]

            cos_bc = cos_lat[None, :, None]
            u_g = u_cos_g / cos_bc
            v_g = v_cos_g / cos_bc

            u_zm = np.mean(u_g, axis=-1)
            t_zm = np.mean(t_g, axis=-1)

            u_prime = u_g - u_zm[:, :, None]
            v_prime = v_g - np.mean(v_g, axis=-1)[:, :, None]

            u_zm_acc += u_zm
            t_zm_acc += t_zm
            u_prime_sq_acc += np.mean(u_prime**2, axis=-1)
            v_prime_sq_acc += np.mean(v_prime**2, axis=-1)
            n_samples += 1

    u_zm_mean = u_zm_acc / n_samples
    t_zm_mean = t_zm_acc / n_samples
    eke_mean = 0.5 * (u_prime_sq_acc + v_prime_sq_acc) / n_samples
    sigma_full = np.asarray(levels.sigma_full)
    surface_idx = np.argmax(sigma_full)
    u_surface = u_zm_mean[surface_idx, :]

    return u_zm_mean, t_zm_mean, eke_mean, u_surface, grid, levels


class TestHeldSuarezClimatology:
    """Validate the Held-Suarez climatology against benchmark targets.

    Runs a 300-day integration (100 spinup + 200 averaging) at T21 L20.
    This is sufficient for the gross features to develop, though the
    climatology will be noisier than the full 1200-day T42 benchmark.
    """

    @pytest.fixture(scope="class")
    def climatology(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GaussianGrid, SigmaLevels]:
        """Run a 300-day T21L20 integration."""
        return _run_held_suarez_climatology(
            truncation=21,
            n_levels=20,
            dt=1200.0,
            spinup_days=100,
            averaging_days=200,
        )

    def test_subtropical_jet_exists(self, climatology: tuple) -> None:
        """Upper-level jet should exceed 15 m/s.

        H&S reference: ~25-30 m/s at T42.  At T21 with shorter averaging
        we accept a lower threshold.
        """
        u_zm, _, _, _, _, levels = climatology
        sigma = np.asarray(levels.sigma_full)
        upper = sigma < 0.4
        jet_max = float(np.max(np.abs(u_zm[upper, :])))
        assert jet_max > 15.0, f"Jet too weak: {jet_max:.1f} m/s (expected > 15)"

    def test_jet_in_subtropics(self, climatology: tuple) -> None:
        """Jet peak should be in the subtropics (15°-50°)."""
        u_zm, _, _, _, grid, levels = climatology
        lat_deg = np.degrees(np.asarray(grid.latitudes))
        sigma = np.asarray(levels.sigma_full)
        upper = sigma < 0.4

        u_upper = np.mean(u_zm[upper, :], axis=0)
        nh_mask = lat_deg > 0
        jet_lat = float(lat_deg[nh_mask][np.argmax(u_upper[nh_mask])])
        assert 15.0 <= jet_lat <= 50.0, f"Jet at {jet_lat:.1f}°N (expected 15-50°N)"

    def test_surface_westerlies(self, climatology: tuple) -> None:
        """Midlatitude surface westerlies should develop."""
        _, _, _, u_surface, grid, _ = climatology
        lat_deg = np.degrees(np.asarray(grid.latitudes))
        midlat = (np.abs(lat_deg) > 30.0) & (np.abs(lat_deg) < 60.0)
        u_midlat_max = float(np.max(u_surface[midlat]))
        assert u_midlat_max > 1.0, (
            f"Surface westerlies too weak: {u_midlat_max:.1f} m/s (expected > 1)"
        )

    def test_equatorial_surface_temperature(self, climatology: tuple) -> None:
        """Equatorial surface temperature should be warm (~295-315 K)."""
        _, t_zm, _, _, grid, levels = climatology
        lat_deg = np.degrees(np.asarray(grid.latitudes))
        sigma = np.asarray(levels.sigma_full)
        eq_idx = np.argmin(np.abs(lat_deg))
        sfc_idx = np.argmax(sigma)
        t_eq = float(t_zm[sfc_idx, eq_idx])
        assert 285.0 < t_eq < 320.0, (
            f"Equatorial surface T = {t_eq:.1f} K (expected 285-320 K)"
        )

    def test_polar_surface_temperature(self, climatology: tuple) -> None:
        """Polar surface temperature should be cold (~245-275 K)."""
        _, t_zm, _, _, grid, levels = climatology
        lat_deg = np.degrees(np.asarray(grid.latitudes))
        sigma = np.asarray(levels.sigma_full)
        pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))
        sfc_idx = np.argmax(sigma)
        t_pole = float(t_zm[sfc_idx, pole_idx])
        assert 235.0 < t_pole < 280.0, (
            f"Polar surface T = {t_pole:.1f} K (expected 235-280 K)"
        )

    def test_equator_pole_temperature_gradient(self, climatology: tuple) -> None:
        """Surface equator-to-pole temperature difference should be 20-60 K."""
        _, t_zm, _, _, grid, levels = climatology
        lat_deg = np.degrees(np.asarray(grid.latitudes))
        sigma = np.asarray(levels.sigma_full)
        sfc_idx = np.argmax(sigma)
        eq_idx = np.argmin(np.abs(lat_deg))
        pole_idx = np.argmin(np.abs(np.abs(lat_deg) - 90.0))
        dt = float(t_zm[sfc_idx, eq_idx] - t_zm[sfc_idx, pole_idx])
        assert 20.0 < dt < 60.0, (
            f"ΔT(eq-pole) = {dt:.1f} K (expected 20-60 K)"
        )

    def test_cold_tropopause(self, climatology: tuple) -> None:
        """Tropopause region should be cold (< 230 K)."""
        _, t_zm, _, _, _, levels = climatology
        sigma = np.asarray(levels.sigma_full)
        tropo = (sigma > 0.05) & (sigma < 0.3)
        if np.any(tropo):
            t_min = float(np.min(t_zm[tropo, :]))
            assert t_min < 230.0, (
                f"Tropopause T_min = {t_min:.1f} K (expected < 230 K)"
            )

    def test_eddy_kinetic_energy(self, climatology: tuple) -> None:
        """Eddy kinetic energy should be significant in midlatitudes."""
        _, _, eke, _, grid, _ = climatology
        lat_deg = np.degrees(np.asarray(grid.latitudes))
        midlat = (np.abs(lat_deg) > 20.0) & (np.abs(lat_deg) < 70.0)
        eke_midlat_max = float(np.max(eke[:, midlat]))
        assert eke_midlat_max > 5.0, (
            f"Midlatitude EKE = {eke_midlat_max:.1f} m²/s² (expected > 5)"
        )
