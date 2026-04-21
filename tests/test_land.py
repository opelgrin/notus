"""Tests for the bucket land surface model (Phase 8C)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.physics.physics_suite import PhysicsSuite, PhysicsSuiteConfig
from notus.physics.radiation import ByrneRadiation
from notus.physics.solar import EARTH_ORBIT
from notus.physics.surface import (
    BucketLandConfig,
    LandState,
    OceanState,
    PrescribedSST,
    SlabOceanConfig,
    SurfaceState,
    beta_function,
    compute_net_land_flux,
    compute_sst,
    diagnose_precipitation,
    init_land_state,
    land_flux_derivative,
    step_bucket_hydrology,
    step_land_implicit,
)
from notus.physics.surface_types import (
    flat_continent_surface,
    moisture_dependent_albedo,
)
from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.timestepping.spinup import spinup_prescribed_sst
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# BucketLandConfig
# ---------------------------------------------------------------------------


class TestBucketLandConfig:
    """Tests for BucketLandConfig."""

    def test_w_crit_property(self) -> None:
        cfg = BucketLandConfig(bucket_capacity=0.2, bucket_critical_fraction=0.5)
        assert cfg.w_crit == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# LandState
# ---------------------------------------------------------------------------


class TestLandState:
    """Tests for LandState dataclass."""

    def test_replace(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        new = land.replace(soil_temperature=jnp.full((4, 8), 300.0))
        assert float(jnp.mean(new.soil_temperature)) == pytest.approx(300.0)
        assert float(jnp.mean(new.bucket_depth)) == pytest.approx(0.1)

    def test_jax_pytree(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        leaves, treedef = jax.tree_util.tree_flatten(land)
        assert len(leaves) == 2
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.soil_temperature, land.soil_temperature)
        assert jnp.allclose(restored.bucket_depth, land.bucket_depth)

    def test_jit_compatible(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )

        @jax.jit
        def f(s: LandState) -> jnp.ndarray:
            return jnp.mean(s.soil_temperature)

        assert float(f(land)) == pytest.approx(280.0)


# ---------------------------------------------------------------------------
# SurfaceState
# ---------------------------------------------------------------------------


class TestSurfaceState:
    """Tests for SurfaceState container."""

    def test_ocean_only(self) -> None:
        ocean = OceanState(surface_temperature=jnp.full(4, 290.0))
        sfc = SurfaceState(ocean=ocean)
        assert sfc.land is None
        assert float(jnp.mean(sfc.ocean.surface_temperature)) == pytest.approx(290.0)

    def test_ocean_plus_land(self) -> None:
        ocean = OceanState(surface_temperature=jnp.full(4, 290.0))
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        sfc = SurfaceState(ocean=ocean, land=land)
        assert sfc.land is not None

    def test_jax_pytree(self) -> None:
        ocean = OceanState(surface_temperature=jnp.full(4, 290.0))
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        sfc = SurfaceState(ocean=ocean, land=land)
        leaves, treedef = jax.tree_util.tree_flatten(sfc)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.ocean.surface_temperature, 290.0)
        assert restored.land is not None
        assert jnp.allclose(restored.land.soil_temperature, 280.0)

    def test_replace(self) -> None:
        ocean = OceanState(surface_temperature=jnp.full(4, 290.0))
        sfc = SurfaceState(ocean=ocean)
        new_ocean = OceanState(surface_temperature=jnp.full(4, 300.0))
        sfc2 = sfc.replace(ocean=new_ocean)
        assert float(jnp.mean(sfc2.ocean.surface_temperature)) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# init_land_state
# ---------------------------------------------------------------------------


class TestInitLandState:
    """Tests for init_land_state factory."""

    def test_shape_from_1d_sst(self) -> None:
        land_frac = jnp.ones((4, 8))
        sst = jnp.full(4, 290.0)
        land = init_land_state(land_frac, sst, BucketLandConfig())
        assert land.soil_temperature.shape == (4, 8)
        assert land.bucket_depth.shape == (4, 8)

    def test_soil_temp_from_sst(self) -> None:
        land_frac = jnp.ones((4, 8))
        sst = jnp.full(4, 285.0)
        land = init_land_state(land_frac, sst, BucketLandConfig())
        assert jnp.allclose(land.soil_temperature, 285.0)


# ---------------------------------------------------------------------------
# Beta function
# ---------------------------------------------------------------------------


class TestBetaFunction:
    """Tests for evaporation availability factor."""

    def test_zero_bucket(self) -> None:
        w = jnp.array(0.0)
        assert float(beta_function(w, 0.1)) == pytest.approx(0.0)

    def test_at_critical(self) -> None:
        w = jnp.array(0.1)
        assert float(beta_function(w, 0.1)) == pytest.approx(1.0)

    def test_above_critical(self) -> None:
        w = jnp.array(0.2)
        assert float(beta_function(w, 0.1)) == pytest.approx(1.0)

    def test_half_critical(self) -> None:
        w = jnp.array(0.05)
        assert float(beta_function(w, 0.1)) == pytest.approx(0.5)

    def test_array(self) -> None:
        w = jnp.array([0.0, 0.05, 0.1, 0.15])
        beta = beta_function(w, 0.1)
        expected = jnp.array([0.0, 0.5, 1.0, 1.0])
        assert jnp.allclose(beta, expected)


# ---------------------------------------------------------------------------
# Net land flux
# ---------------------------------------------------------------------------


class TestComputeNetLandFlux:
    """Tests for land surface energy balance."""

    def _make_args(self, t_land: float = 290.0, beta_val: float = 1.0) -> dict:
        n_lat, n_lon = 4, 8
        # sw_down_surface = insolation * exp(-sw_tau_0) = 300 * exp(-0.22)
        sw_down = 300.0 * jnp.exp(-0.22)
        return {
            "land_temperature": jnp.full((n_lat, n_lon), t_land),
            "t_air": jnp.full((n_lat, n_lon), 285.0),
            "q_air": jnp.full((n_lat, n_lon), 0.005),
            "wind_speed": jnp.full((n_lat, n_lon), 5.0),
            "surface_pressure": jnp.full((n_lat, n_lon), 1.0e5),
            "sw_down_surface": jnp.full((n_lat, n_lon), sw_down),
            "lw_down": jnp.full((n_lat, n_lon), 300.0),
            "beta": jnp.full((n_lat, n_lon), beta_val),
            "gravity": 9.80616,
            "gas_constant": 287.04,
            "specific_heat_cp": 1004.64,
            "epsilon": 0.622,
            "latent_heat": 2.5e6,
            "drag_coefficient": 0.001,
            "surface_albedo": 0.25,
        }

    def test_returns_tuple(self) -> None:
        args = self._make_args()
        result = compute_net_land_flux(**args)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_flux_finite(self) -> None:
        net_flux, evap = compute_net_land_flux(**self._make_args())
        assert jnp.all(jnp.isfinite(net_flux))
        assert jnp.all(jnp.isfinite(evap))

    def test_evap_positive(self) -> None:
        """Evaporation is positive (upward) when surface is warmer."""
        _, evap = compute_net_land_flux(**self._make_args(t_land=300.0))
        assert jnp.all(evap >= 0.0)

    def test_beta_zero_kills_evap(self) -> None:
        """Zero beta means no evaporation."""
        _, evap = compute_net_land_flux(**self._make_args(beta_val=0.0))
        assert jnp.allclose(evap, 0.0)

    def test_beta_reduces_evap(self) -> None:
        """Partial beta reduces evaporation vs full beta."""
        _, evap_full = compute_net_land_flux(**self._make_args(beta_val=1.0))
        _, evap_half = compute_net_land_flux(**self._make_args(beta_val=0.5))
        # evap_half should be roughly half of evap_full (since beta scales linearly)
        ratio = float(jnp.mean(evap_half)) / float(jnp.mean(evap_full))
        assert 0.4 < ratio < 0.6


# ---------------------------------------------------------------------------
# Land flux derivative
# ---------------------------------------------------------------------------


class TestLandFluxDerivative:
    """Tests for dF/dT_land."""

    def test_always_negative(self) -> None:
        n_lat, n_lon = 4, 8
        dflux = land_flux_derivative(
            jnp.full((n_lat, n_lon), 290.0),
            jnp.full((n_lat, n_lon), 5.0),
            jnp.full((n_lat, n_lon), 1.0e5),
            jnp.full((n_lat, n_lon), 1.0),
            gas_constant=287.04,
            specific_heat_cp=1004.64,
            epsilon=0.622,
            latent_heat=2.5e6,
            drag_coefficient=0.001,
        )
        assert jnp.all(dflux < 0.0)

    def test_beta_zero_reduces_derivative(self) -> None:
        """With beta=0, latent component vanishes → derivative is less negative."""
        n_lat, n_lon = 4, 8
        args = {
            "land_temperature": jnp.full((n_lat, n_lon), 290.0),
            "wind_speed": jnp.full((n_lat, n_lon), 5.0),
            "surface_pressure": jnp.full((n_lat, n_lon), 1.0e5),
            "gas_constant": 287.04,
            "specific_heat_cp": 1004.64,
            "epsilon": 0.622,
            "latent_heat": 2.5e6,
            "drag_coefficient": 0.001,
        }
        dflux_wet = land_flux_derivative(
            **args,
            beta=jnp.full((n_lat, n_lon), 1.0),
        )
        dflux_dry = land_flux_derivative(
            **args,
            beta=jnp.full((n_lat, n_lon), 0.0),
        )
        # dry derivative (no latent) should be less negative
        assert float(jnp.mean(dflux_dry)) > float(jnp.mean(dflux_wet))


# ---------------------------------------------------------------------------
# step_land_implicit
# ---------------------------------------------------------------------------


class TestStepLandImplicit:
    """Tests for implicit soil temperature update."""

    def test_positive_flux_warms(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        net_flux = jnp.full((4, 8), 100.0)  # 100 W/m^2 warming
        dflux_dt = jnp.full((4, 8), -5.0)
        new = step_land_implicit(land, net_flux, dflux_dt, 1.0e6, 900.0)
        assert jnp.all(new.soil_temperature > land.soil_temperature)

    def test_negative_flux_cools(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 300.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        net_flux = jnp.full((4, 8), -100.0)
        dflux_dt = jnp.full((4, 8), -5.0)
        new = step_land_implicit(land, net_flux, dflux_dt, 1.0e6, 900.0)
        assert jnp.all(new.soil_temperature < land.soil_temperature)

    def test_bucket_unchanged(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        new = step_land_implicit(
            land,
            jnp.full((4, 8), 100.0),
            jnp.full((4, 8), -5.0),
            1.0e6,
            900.0,
        )
        assert jnp.allclose(new.bucket_depth, land.bucket_depth)

    def test_implicit_stable_large_dt(self) -> None:
        """Implicit scheme should be stable even with very large dt."""
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        net_flux = jnp.full((4, 8), 500.0)
        dflux_dt = jnp.full((4, 8), -10.0)
        new = step_land_implicit(land, net_flux, dflux_dt, 1.0e6, 86400.0)
        assert jnp.all(jnp.isfinite(new.soil_temperature))
        # Should warm but not blow up
        assert jnp.all(new.soil_temperature < 400.0)


# ---------------------------------------------------------------------------
# step_bucket_hydrology
# ---------------------------------------------------------------------------


class TestStepBucketHydrology:
    """Tests for bucket P-E-R update."""

    def test_precipitation_fills_bucket(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.05),
        )
        precip = jnp.full((4, 8), 1.0e-4)  # 0.1 mm/s
        evap = jnp.zeros((4, 8))
        new = step_bucket_hydrology(land, precip, evap, 0.15, 3600.0)
        assert jnp.all(new.bucket_depth > land.bucket_depth)

    def test_evaporation_drains_bucket(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.1),
        )
        precip = jnp.zeros((4, 8))
        evap = jnp.full((4, 8), 1.0e-4)
        new = step_bucket_hydrology(land, precip, evap, 0.15, 3600.0)
        assert jnp.all(new.bucket_depth < land.bucket_depth)

    def test_bucket_cannot_go_negative(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.001),
        )
        precip = jnp.zeros((4, 8))
        evap = jnp.full((4, 8), 1.0)  # very large evaporation
        new = step_bucket_hydrology(land, precip, evap, 0.15, 3600.0)
        assert jnp.all(new.bucket_depth >= 0.0)

    def test_bucket_cannot_exceed_capacity(self) -> None:
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.14),
        )
        precip = jnp.full((4, 8), 1.0)  # very large precipitation
        evap = jnp.zeros((4, 8))
        new = step_bucket_hydrology(land, precip, evap, 0.15, 3600.0)
        assert jnp.all(new.bucket_depth <= 0.15)

    def test_conservation_no_overflow(self) -> None:
        """Without overflow, P-E is fully captured."""
        land = LandState(
            soil_temperature=jnp.full((4, 8), 280.0),
            bucket_depth=jnp.full((4, 8), 0.05),
        )
        precip = jnp.full((4, 8), 1.0e-5)
        evap = jnp.full((4, 8), 0.5e-5)
        dt = 3600.0
        new = step_bucket_hydrology(land, precip, evap, 0.15, dt)
        dw = float(jnp.mean(new.bucket_depth - land.bucket_depth))
        expected = dt * (1.0e-5 - 0.5e-5) / 1000.0  # P-E converted to m
        assert dw == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Moisture-dependent albedo
# ---------------------------------------------------------------------------


class TestMoistureDependentAlbedo:
    """Tests for Frierson moisture-dependent albedo."""

    def test_dry_bucket_gives_dry_albedo(self) -> None:
        land_frac = jnp.ones((4, 8))
        bucket = jnp.zeros((4, 8))
        alpha = moisture_dependent_albedo(bucket, 0.15, 0.35, 0.20, land_frac, 0.06)
        assert float(jnp.mean(alpha)) == pytest.approx(0.35)

    def test_full_bucket_gives_wet_albedo(self) -> None:
        land_frac = jnp.ones((4, 8))
        bucket = jnp.full((4, 8), 0.15)
        alpha = moisture_dependent_albedo(bucket, 0.15, 0.35, 0.20, land_frac, 0.06)
        assert float(jnp.mean(alpha)) == pytest.approx(0.20)

    def test_ocean_returns_ocean_albedo(self) -> None:
        land_frac = jnp.zeros((4, 8))
        bucket = jnp.full((4, 8), 0.1)
        alpha = moisture_dependent_albedo(bucket, 0.15, 0.35, 0.20, land_frac, 0.06)
        assert float(jnp.mean(alpha)) == pytest.approx(0.06)

    def test_blended(self) -> None:
        """50% land fraction blends ocean and land albedo."""
        land_frac = jnp.full((4, 8), 0.5)
        bucket = jnp.full((4, 8), 0.15)  # full
        alpha = moisture_dependent_albedo(bucket, 0.15, 0.35, 0.20, land_frac, 0.06)
        expected = 0.5 * 0.06 + 0.5 * 0.20
        assert float(jnp.mean(alpha)) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Precipitation diagnostic
# ---------------------------------------------------------------------------


class TestDiagnosePrecipitation:
    """Tests for the precipitation diagnostic."""

    def test_dry_atmosphere_zero_precip(self) -> None:
        """Dry atmosphere produces no precipitation."""
        n_lev, n_lat, n_lon = 5, 4, 8
        t_grid = jnp.full((n_lev, n_lat, n_lon), 250.0)
        q_grid = jnp.full((n_lev, n_lat, n_lon), 1.0e-6)
        ps = jnp.full((n_lat, n_lon), 1.0e5)
        sigma = jnp.linspace(0.1, 0.9, n_lev)
        dsigma = jnp.full(n_lev, 0.8 / n_lev)
        pressure = sigma[:, None, None] * ps[None, :, :]

        precip = diagnose_precipitation(
            t_grid,
            q_grid,
            pressure,
            dsigma,
            ps,
            gravity=9.80616,
            epsilon=0.622,
            latent_heat=2.5e6,
            specific_heat_cp=1004.64,
            gas_constant=287.04,
        )
        assert jnp.all(precip >= 0.0)
        # Very dry atmosphere should have near-zero precipitation
        assert float(jnp.max(precip)) < 1.0e-3

    def test_non_negative(self) -> None:
        """Precipitation is always non-negative."""
        n_lev, n_lat, n_lon = 5, 4, 8
        t_grid = jnp.full((n_lev, n_lat, n_lon), 280.0)
        q_grid = jnp.full((n_lev, n_lat, n_lon), 0.01)
        ps = jnp.full((n_lat, n_lon), 1.0e5)
        sigma = jnp.linspace(0.1, 0.9, n_lev)
        dsigma = jnp.full(n_lev, 0.8 / n_lev)
        pressure = sigma[:, None, None] * ps[None, :, :]

        precip = diagnose_precipitation(
            t_grid,
            q_grid,
            pressure,
            dsigma,
            ps,
            gravity=9.80616,
            epsilon=0.622,
            latent_heat=2.5e6,
            specific_heat_cp=1004.64,
            gas_constant=287.04,
        )
        assert jnp.all(precip >= 0.0)


# ---------------------------------------------------------------------------
# Integration tests: coupled land-ocean
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLandOceanIntegration:
    """Integration tests for the coupled land-ocean stepper."""

    @pytest.fixture(scope="class")
    def setup(self) -> dict:
        """Shared spinup state for integration tests."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid, EARTH.radius)
        levels = standard_sigma_levels(20)
        state, ref, sphi = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
            initial_rh=0.7,
            seed=42,
        )
        config = PhysicsSuiteConfig(
            radiation=ByrneRadiation(sw_tau_0=0.22),
            orbital=EARTH_ORBIT,
        )
        forcing_spinup = PhysicsSuite(transform, EARTH, levels, config=config)
        result = spinup_prescribed_sst(
            state,
            forcing_spinup,
            transform,
            EARTH,
            levels,
            ref,
            sphi,
            dt=900.0,
            spinup_days=50,
            averaging_days=50,
            verbose=False,
        )
        return {
            "grid": grid,
            "transform": transform,
            "levels": levels,
            "config": config,
            "ref": ref,
            "sphi": sphi,
            "result": result,
        }

    def _run_coupled_land(
        self,
        setup: dict,
        n_days: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run coupled integration with land, return (T_grid, SST, T_land, bucket)."""
        s = setup
        forcing = PhysicsSuite(s["transform"], EARTH, s["levels"], config=s["config"])
        sfc_props = flat_continent_surface(s["grid"].latitudes, s["grid"].longitudes)
        sst_init = compute_sst(PrescribedSST(), s["grid"].latitudes)
        ocean = OceanState(surface_temperature=sst_init)
        land_cfg = BucketLandConfig()
        land = init_land_state(sfc_props.land_fraction, sst_init, land_cfg)
        surface = SurfaceState(ocean=ocean, land=land)

        init_fn, step_fn = build_coupled_pe_stepper(
            transform=s["transform"],
            planet=EARTH,
            levels=s["levels"],
            reference_temperature=s["ref"],
            surface_geopotential=s["sphi"],
            dt=900.0,
            forcing=forcing,
            ocean_config=SlabOceanConfig(mixed_layer_depth=50.0),
            q_flux=s["result"].q_flux,
            surface_properties=sfc_props,
            land_config=land_cfg,
            spectral_filter=exponential_filter(s["transform"].arrays, 900.0),
        )
        steps_per_day = 96
        day_of_year = jnp.float64(0.0)
        forcing.prescribed_sst = surface.ocean.surface_temperature
        prev, curr, surface, _diags = init_fn(s["result"].state, surface, day_of_year)

        for day in range(1, n_days + 1):
            day_of_year = jnp.float64(day)
            forcing.prescribed_sst = surface.ocean.surface_temperature

            def scan_body(carry, _, day_of_year=day_of_year):
                p, c, sfc = carry
                p, c, sfc, _diags = step_fn(p, c, sfc, day_of_year)
                return (p, c, sfc), None

            (prev, curr, surface), _ = jax.lax.scan(
                scan_body,
                (prev, curr, surface),
                None,
                length=steps_per_day,
            )

        t_grid = np.asarray(
            jax.vmap(s["transform"].spectral_to_grid)(curr.temperature),
        )
        sst = np.asarray(surface.ocean.surface_temperature)
        assert surface.land is not None
        t_land = np.asarray(surface.land.soil_temperature)
        bucket = np.asarray(surface.land.bucket_depth)
        return t_grid, sst, t_land, bucket

    def test_land_ocean_stable_10_days(self, setup: dict) -> None:
        """Coupled land-ocean run stays finite for 10 days."""
        t_grid, sst, t_land, bucket = self._run_coupled_land(setup)
        assert np.all(np.isfinite(t_grid))
        assert np.all(np.isfinite(sst))
        assert np.all(np.isfinite(t_land))
        assert np.all(np.isfinite(bucket))

    def test_sst_physical_range(self, setup: dict) -> None:
        """SST stays in a physically plausible range."""
        _, sst, _, _ = self._run_coupled_land(setup)
        assert np.min(sst) > 200.0
        assert np.max(sst) < 350.0

    def test_land_temp_physical_range(self, setup: dict) -> None:
        """Soil temperature stays in a physically plausible range."""
        _, _, t_land, _ = self._run_coupled_land(setup)
        assert np.min(t_land) > 150.0
        assert np.max(t_land) < 400.0

    def test_bucket_in_range(self, setup: dict) -> None:
        """Bucket depth stays in [0, W_max]."""
        _, _, _, bucket = self._run_coupled_land(setup)
        assert np.min(bucket) >= 0.0
        assert np.max(bucket) <= 0.15
