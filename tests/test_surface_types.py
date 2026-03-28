"""Tests for surface type classification and properties."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import moist_aquaplanet_initial_state
from notus.operators import exponential_filter
from notus.physics.simple_physics import SimplePhysics, SimplePhysicsConfig
from notus.physics.solar import EARTH_ORBIT
from notus.physics.surface import OceanState, PrescribedSST, SlabOceanConfig, compute_sst
from notus.physics.surface_types import (
    LAND_ALBEDO,
    LAND_Z0_MOMENTUM,
    OCEAN_ALBEDO,
    OCEAN_Z0_MOMENTUM,
    SurfaceProperties,
    aquaplanet_surface,
    flat_continent_surface,
)
from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.timestepping.spinup import spinup_prescribed_sst
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# SurfaceProperties dataclass
# ---------------------------------------------------------------------------


class TestSurfaceProperties:
    """Tests for the SurfaceProperties dataclass."""

    def test_shape_properties(self) -> None:
        props = aquaplanet_surface(16, 32)
        assert props.n_lat == 16
        assert props.n_lon == 32

    def test_replace(self) -> None:
        props = aquaplanet_surface(4, 8)
        new_albedo = jnp.full((4, 8), 0.5)
        replaced = props.replace(albedo=new_albedo)
        assert jnp.allclose(replaced.albedo, 0.5)
        assert jnp.allclose(replaced.land_fraction, 0.0)  # unchanged

    def test_jax_pytree(self) -> None:
        """SurfaceProperties works as a JAX pytree."""
        props = aquaplanet_surface(4, 8)
        leaves, treedef = jax.tree_util.tree_flatten(props)
        assert len(leaves) == 4
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.albedo, props.albedo)

    def test_jit_compatible(self) -> None:
        """SurfaceProperties can be passed through jax.jit."""
        props = aquaplanet_surface(4, 8)

        @jax.jit
        def f(p: SurfaceProperties) -> jnp.ndarray:
            return jnp.mean(p.albedo)

        result = f(props)
        assert float(result) == pytest.approx(OCEAN_ALBEDO)


# ---------------------------------------------------------------------------
# Aquaplanet surface
# ---------------------------------------------------------------------------


class TestAquaplanetSurface:
    """Tests for the aquaplanet (all-ocean) surface."""

    def test_all_ocean(self) -> None:
        props = aquaplanet_surface(16, 32)
        assert jnp.all(props.land_fraction == 0.0)

    def test_uniform_albedo(self) -> None:
        props = aquaplanet_surface(16, 32)
        assert jnp.allclose(props.albedo, OCEAN_ALBEDO)

    def test_uniform_roughness(self) -> None:
        props = aquaplanet_surface(16, 32)
        assert jnp.allclose(props.z0_momentum, OCEAN_Z0_MOMENTUM)

    def test_custom_albedo(self) -> None:
        props = aquaplanet_surface(4, 8, ocean_albedo=0.1)
        assert jnp.allclose(props.albedo, 0.1)

    def test_shape(self) -> None:
        props = aquaplanet_surface(16, 32)
        assert props.land_fraction.shape == (16, 32)
        assert props.albedo.shape == (16, 32)
        assert props.z0_momentum.shape == (16, 32)
        assert props.z0_heat.shape == (16, 32)


# ---------------------------------------------------------------------------
# Flat continent surface
# ---------------------------------------------------------------------------


class TestFlatContinentSurface:
    """Tests for the rectangular continent surface."""

    @pytest.fixture()
    def latitudes(self) -> np.ndarray:
        return np.radians(np.linspace(-87.5, 87.5, 16))

    @pytest.fixture()
    def longitudes(self) -> np.ndarray:
        return np.radians(np.linspace(0, 348.75, 32))

    def test_has_land(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        props = flat_continent_surface(latitudes, longitudes)
        assert float(jnp.max(props.land_fraction)) == 1.0

    def test_has_ocean(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        props = flat_continent_surface(latitudes, longitudes)
        assert float(jnp.min(props.land_fraction)) == 0.0

    def test_land_fraction_binary(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """Land fraction is 0 or 1 (no fractional cells)."""
        props = flat_continent_surface(latitudes, longitudes)
        unique = jnp.unique(props.land_fraction)
        assert len(unique) == 2
        assert 0.0 in unique
        assert 1.0 in unique

    def test_land_albedo_higher(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """Land points have higher albedo than ocean points."""
        props = flat_continent_surface(latitudes, longitudes)
        land_mask = props.land_fraction > 0.5
        ocean_mask = ~land_mask
        assert float(jnp.mean(props.albedo[land_mask])) > float(jnp.mean(props.albedo[ocean_mask]))

    def test_land_rougher(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """Land points have larger roughness than ocean points."""
        props = flat_continent_surface(latitudes, longitudes)
        land_mask = props.land_fraction > 0.5
        ocean_mask = ~land_mask
        assert float(jnp.mean(props.z0_momentum[land_mask])) > float(jnp.mean(props.z0_momentum[ocean_mask]))

    def test_default_covers_half_longitudes(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """Default continent (0-180 lon) covers roughly half the longitudes."""
        props = flat_continent_surface(latitudes, longitudes)
        frac = float(jnp.mean(props.land_fraction))
        assert 0.3 < frac < 0.7

    def test_custom_bounds(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """Custom lat/lon bounds produce land only in the specified region."""
        props = flat_continent_surface(
            latitudes, longitudes,
            lat_south=-30.0, lat_north=30.0,
            lon_west=90.0, lon_east=180.0,
        )
        # Check that poles have no land
        pole_land = float(props.land_fraction[0, 0]) + float(props.land_fraction[-1, 0])
        assert pole_land == 0.0
        # Check that some equatorial points have land
        eq_idx = len(latitudes) // 2
        assert float(jnp.max(props.land_fraction[eq_idx, :])) == 1.0

    def test_all_ocean_when_no_land(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """If bounds exclude all grid points, result is all ocean."""
        props = flat_continent_surface(
            latitudes, longitudes,
            lat_south=89.0, lat_north=90.0,
            lon_west=359.0, lon_east=360.0,
        )
        assert float(jnp.sum(props.land_fraction)) == 0.0

    def test_custom_parameters(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        """Custom albedo and roughness are applied correctly."""
        props = flat_continent_surface(
            latitudes, longitudes,
            ocean_albedo=0.05, land_albedo=0.40,
            ocean_z0_momentum=2e-4, land_z0_momentum=0.2,
        )
        land_mask = props.land_fraction > 0.5
        ocean_mask = ~land_mask
        if jnp.any(land_mask):
            assert float(props.albedo[land_mask][0]) == pytest.approx(0.40)
        if jnp.any(ocean_mask):
            assert float(props.albedo[ocean_mask][0]) == pytest.approx(0.05)

    def test_albedo_in_range(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        props = flat_continent_surface(latitudes, longitudes)
        assert float(jnp.min(props.albedo)) >= 0.0
        assert float(jnp.max(props.albedo)) <= 1.0

    def test_roughness_positive(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        props = flat_continent_surface(latitudes, longitudes)
        assert jnp.all(props.z0_momentum > 0)
        assert jnp.all(props.z0_heat > 0)


# ---------------------------------------------------------------------------
# Integration test: coupled slab ocean with SurfaceProperties
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSurfacePropertiesIntegration:
    """Integration tests: coupled slab ocean with spatially varying surface."""

    @pytest.fixture(scope="class")
    def setup(self) -> dict:
        """Shared spinup state for integration tests (expensive, run once)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid, EARTH.radius)
        levels = standard_sigma_levels(20)
        state, ref, sphi = moist_aquaplanet_initial_state(
            transform, EARTH, levels, initial_rh=0.7, seed=42,
        )
        config = SimplePhysicsConfig(
            radiation_scheme="byrne", sw_tau_0=0.22, orbital=EARTH_ORBIT,
        )
        forcing_spinup = SimplePhysics(transform, EARTH, levels, config=config)
        result = spinup_prescribed_sst(
            state, forcing_spinup, transform, EARTH, levels, ref, sphi,
            dt=900.0, spinup_days=50, averaging_days=50, verbose=False,
        )
        return {
            "grid": grid, "transform": transform, "levels": levels,
            "config": config, "ref": ref, "sphi": sphi, "result": result,
        }

    def _run_coupled(
        self, setup: dict, surface_props: SurfaceProperties, n_days: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run coupled integration, return (temperature_grid, sst)."""
        s = setup
        forcing = SimplePhysics(s["transform"], EARTH, s["levels"], config=s["config"])
        ocean = OceanState(
            surface_temperature=compute_sst(PrescribedSST(), s["grid"].latitudes),
        )
        init_fn, step_fn = build_coupled_pe_stepper(
            transform=s["transform"], planet=EARTH, levels=s["levels"],
            reference_temperature=s["ref"], surface_geopotential=s["sphi"],
            dt=900.0, forcing=forcing,
            ocean_config=SlabOceanConfig(mixed_layer_depth=50.0),
            q_flux=s["result"].q_flux, surface_properties=surface_props,
            spectral_filter=exponential_filter(s["transform"].arrays, 900.0),
        )
        steps_per_day = 96
        forcing.day_of_year = jnp.float64(0.0)
        forcing.sst = ocean.surface_temperature
        prev, curr, ocean = init_fn(s["result"].state, ocean)
        for day in range(1, n_days + 1):
            forcing.day_of_year = jnp.float64(day)
            forcing.sst = ocean.surface_temperature

            def scan_body(carry, _):
                p, c, o = carry
                p, c, o = step_fn(p, c, o)
                return (p, c, o), None

            (prev, curr, ocean), _ = jax.lax.scan(
                scan_body, (prev, curr, ocean), None, length=steps_per_day,
            )
        t_grid = np.asarray(
            jax.vmap(s["transform"].spectral_to_grid)(curr.temperature),
        )
        sst = np.asarray(ocean.surface_temperature)
        return t_grid, sst

    def test_aquaplanet_stable_10_days(self, setup: dict) -> None:
        """Aquaplanet surface: 10-day coupled run stays finite."""
        sfc = aquaplanet_surface(setup["grid"].n_lat, setup["grid"].n_lon)
        t_grid, sst = self._run_coupled(setup, sfc)
        assert np.all(np.isfinite(t_grid))
        assert np.all(np.isfinite(sst))

    def test_continent_stable_10_days(self, setup: dict) -> None:
        """Flat continent surface: 10-day coupled run stays finite."""
        sfc = flat_continent_surface(
            setup["grid"].latitudes, setup["grid"].longitudes,
        )
        t_grid, sst = self._run_coupled(setup, sfc)
        assert np.all(np.isfinite(t_grid))
        assert np.all(np.isfinite(sst))

    def test_continent_cooler_than_aquaplanet(self, setup: dict) -> None:
        """Higher land albedo makes the continent run cooler."""
        sfc_aqua = aquaplanet_surface(setup["grid"].n_lat, setup["grid"].n_lon)
        sfc_cont = flat_continent_surface(
            setup["grid"].latitudes, setup["grid"].longitudes,
        )
        _, sst_aqua = self._run_coupled(setup, sfc_aqua)
        _, sst_cont = self._run_coupled(setup, sfc_cont)
        # Continent reflects more SW → cooler SST on average
        assert np.mean(sst_cont) < np.mean(sst_aqua)

    def test_sst_physical_range(self, setup: dict) -> None:
        """SST stays in a physically plausible range after 10 days."""
        sfc = flat_continent_surface(
            setup["grid"].latitudes, setup["grid"].longitudes,
        )
        _, sst = self._run_coupled(setup, sfc)
        assert np.min(sst) > 200.0  # not frozen solid
        assert np.max(sst) < 350.0  # not boiling
