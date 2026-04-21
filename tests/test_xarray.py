"""Tests for xarray conversion utilities.

Validates:
1. Round-trip state → dataset → state on Gaussian grid
2. Spectral evaluation on regular lat/lon grid
3. Bilinear regridding from regular to Gaussian grid
4. Wind ↔ vorticity/divergence conversion
5. Surface state round-tripping
6. Static field round-tripping (orography, reference_temperature, q_flux, surface_properties)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from notus.constants import EARTH
from notus.grid import GaussianGrid
from notus.initial_conditions import held_suarez_initial_state
from notus.physics.surface import LandState, OceanState, SeaIceState, SurfaceState
from notus.physics.surface_types import SurfaceProperties
from notus.topography import gaussian_mountain
from notus.transforms import SpectralTransform
from notus.vertical.sigma import standard_sigma_levels
from notus.xarray import (
    DatasetContents,
    _regrid_to_gaussian,
    dataset_to_state,
    from_regular_latlon,
    state_to_dataset,
    to_regular_latlon,
)


jax.config.update("jax_enable_x64", True)

TRUNCATION = 21
N_LEVELS = 10


def _setup() -> tuple[SpectralTransform, GaussianGrid, object, object, object, object]:
    """Create a T21 Held-Suarez initial state for testing."""
    grid = GaussianGrid(truncation=TRUNCATION)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = standard_sigma_levels(N_LEVELS)
    state, ref_temps, surface_phi = held_suarez_initial_state(
        transform,
        EARTH,
        levels,
    )
    return transform, grid, levels, state, ref_temps, surface_phi


# =====================================================================
# state_to_dataset
# =====================================================================


class TestStateToDataset:
    """Verify state_to_dataset produces well-formed datasets."""

    def test_coordinate_shapes(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        assert ds.sizes["lat"] == grid.n_lat
        assert ds.sizes["lon"] == grid.n_lon
        assert ds.sizes["sigma"] == N_LEVELS

    def test_expected_variables(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        for var in ["u", "v", "temperature", "surface_pressure"]:
            assert var in ds, f"Missing variable: {var}"
        assert "specific_humidity" not in ds

    def test_temperature_reasonable(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        t = ds["temperature"].values
        assert np.all(t > 150.0), "Temperature below 150 K"
        assert np.all(t < 400.0), "Temperature above 400 K"

    def test_surface_pressure_reasonable(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        ps = ds["surface_pressure"].values
        np.testing.assert_allclose(np.mean(ps), 1e5, rtol=0.01)

    def test_with_surface_state(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ocean = OceanState(surface_temperature=jnp.full(grid.n_lat, 300.0))
        surface = SurfaceState(ocean=ocean)
        ds = state_to_dataset(state, transform, EARTH, levels, surface=surface)
        assert "sst" in ds
        np.testing.assert_allclose(ds["sst"].values, 300.0)

    def test_with_land_state(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ocean = OceanState(
            surface_temperature=jnp.full((grid.n_lat, grid.n_lon), 300.0),
        )
        land = LandState(
            soil_temperature=jnp.full((grid.n_lat, grid.n_lon), 280.0),
            bucket_depth=jnp.full((grid.n_lat, grid.n_lon), 0.1),
        )
        surface = SurfaceState(ocean=ocean, land=land)
        ds = state_to_dataset(state, transform, EARTH, levels, surface=surface)
        assert "sst" in ds
        assert "soil_temperature" in ds
        assert "bucket_depth" in ds

    def test_with_sea_ice_state(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ocean = OceanState(surface_temperature=jnp.full(grid.n_lat, 271.35))
        ice = SeaIceState(
            ice_thickness=jnp.full(grid.n_lat, 0.8),
            ice_fraction=jnp.full(grid.n_lat, 0.8),
        )
        surface = SurfaceState(ocean=ocean, ice=ice)
        ds = state_to_dataset(state, transform, EARTH, levels, surface=surface)
        assert "ice_thickness" in ds
        assert "ice_fraction" in ds
        np.testing.assert_allclose(ds["ice_thickness"].values, 0.8)
        np.testing.assert_allclose(ds["ice_fraction"].values, 0.8)

    def test_dataset_attributes(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        assert ds.attrs["grid_type"] == "gaussian"
        assert ds.attrs["truncation"] == TRUNCATION


# =====================================================================
# Round-trip: state → dataset → state (Gaussian grid)
# =====================================================================


class TestGaussianRoundTrip:
    """Verify that dataset_to_state inverts state_to_dataset."""

    def test_temperature_roundtrip(self) -> None:
        """Temperature should survive the round trip exactly."""
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        contents = dataset_to_state(ds, transform, EARTH)

        np.testing.assert_allclose(
            np.asarray(contents.state.temperature),
            np.asarray(state.temperature),
            atol=1e-8,
        )

    def test_surface_pressure_roundtrip(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        contents = dataset_to_state(ds, transform, EARTH)

        np.testing.assert_allclose(
            np.asarray(contents.state.log_surface_pressure),
            np.asarray(state.log_surface_pressure),
            atol=1e-8,
        )

    def test_vordiv_roundtrip(self) -> None:
        """Vorticity/divergence round-trip via u,v → spectral curl/div."""
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        contents = dataset_to_state(ds, transform, EARTH)

        np.testing.assert_allclose(
            np.asarray(contents.state.vorticity),
            np.asarray(state.vorticity),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(contents.state.divergence),
            np.asarray(state.divergence),
            atol=1e-6,
        )

    def test_surface_state_roundtrip(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ocean = OceanState(surface_temperature=jnp.full(grid.n_lat, 300.0))
        land = LandState(
            soil_temperature=jnp.full((grid.n_lat, grid.n_lon), 280.0),
            bucket_depth=jnp.full((grid.n_lat, grid.n_lon), 0.1),
        )
        surface = SurfaceState(ocean=ocean, land=land)
        ds = state_to_dataset(state, transform, EARTH, levels, surface=surface)
        contents = dataset_to_state(ds, transform, EARTH)

        assert contents.surface is not None
        assert contents.surface.land is not None
        np.testing.assert_allclose(
            np.asarray(contents.surface.ocean.surface_temperature),
            np.asarray(surface.ocean.surface_temperature),
        )
        np.testing.assert_allclose(
            np.asarray(contents.surface.land.soil_temperature),
            np.asarray(surface.land.soil_temperature),
        )

    def test_surface_state_roundtrip_with_ice(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ocean = OceanState(surface_temperature=jnp.full(grid.n_lat, 271.35))
        ice = SeaIceState(
            ice_thickness=jnp.full(grid.n_lat, 1.2),
            ice_fraction=jnp.full(grid.n_lat, 1.0),
        )
        surface = SurfaceState(ocean=ocean, ice=ice)
        ds = state_to_dataset(state, transform, EARTH, levels, surface=surface)
        contents = dataset_to_state(ds, transform, EARTH)

        assert contents.surface is not None
        assert contents.surface.ice is not None
        np.testing.assert_allclose(
            np.asarray(contents.surface.ice.ice_thickness),
            np.asarray(surface.ice.ice_thickness),
        )
        np.testing.assert_allclose(
            np.asarray(contents.surface.ice.ice_fraction),
            np.asarray(surface.ice.ice_fraction),
        )

    def test_returns_dataset_contents(self) -> None:
        """dataset_to_state should return a DatasetContents instance."""
        transform, _, levels, state, _, _ = _setup()
        ds = state_to_dataset(state, transform, EARTH, levels)
        contents = dataset_to_state(ds, transform, EARTH)
        assert isinstance(contents, DatasetContents)
        assert contents.surface is None
        assert contents.surface_geopotential is None
        assert contents.reference_temperature is None
        assert contents.q_flux is None
        assert contents.surface_properties is None


# =====================================================================
# to_regular_latlon
# =====================================================================


class TestToRegularLatlon:
    """Verify spectral evaluation on a regular lat/lon grid."""

    def test_coordinate_shapes(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        n_lat, n_lon = 64, 128
        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=n_lat,
            n_lon=n_lon,
        )
        assert ds.sizes["lat"] == n_lat
        assert ds.sizes["lon"] == n_lon
        assert ds.sizes["sigma"] == N_LEVELS

    def test_regular_grid_attributes(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = to_regular_latlon(state, transform, EARTH, levels)
        assert ds.attrs["grid_type"] == "regular"
        assert ds.attrs["truncation"] == TRUNCATION

    def test_temperature_matches_gaussian(self) -> None:
        """Regular-grid temperature should match Gaussian at comparable res."""
        transform, _, levels, state, _, _ = _setup()
        ds_gauss = state_to_dataset(state, transform, EARTH, levels)
        ds_reg = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=90,
            n_lon=180,
        )
        t_mean_gauss = float(np.mean(ds_gauss["temperature"].values))
        t_mean_reg = float(np.mean(ds_reg["temperature"].values))
        np.testing.assert_allclose(t_mean_gauss, t_mean_reg, rtol=0.01)

    def test_surface_pressure_global_mean(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds_reg = to_regular_latlon(state, transform, EARTH, levels)
        ps_mean = float(np.mean(ds_reg["surface_pressure"].values))
        np.testing.assert_allclose(ps_mean, 1e5, rtol=0.01)

    def test_n_lon_too_small_raises(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        import pytest

        with pytest.raises(ValueError, match="too small"):
            to_regular_latlon(
                state,
                transform,
                EARTH,
                levels,
                n_lat=64,
                n_lon=20,
            )

    def test_surface_regridded(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        ocean = OceanState(surface_temperature=jnp.full(grid.n_lat, 300.0))
        surface = SurfaceState(ocean=ocean)
        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=64,
            n_lon=128,
            surface=surface,
        )
        assert "sst" in ds
        np.testing.assert_allclose(ds["sst"].values, 300.0, atol=0.5)


# =====================================================================
# Bilinear regridding
# =====================================================================


class TestBilinearRegrid:
    """Unit tests for the _regrid_to_gaussian helper."""

    def test_constant_field(self) -> None:
        lat_in = np.linspace(-90, 90, 73)
        lon_in = np.linspace(0, 360, 144, endpoint=False)
        field = np.full((73, 144), 42.0)
        grid = GaussianGrid(truncation=TRUNCATION)
        lat_out = np.degrees(np.asarray(grid.latitudes))
        lon_out = np.degrees(np.asarray(grid.longitudes))
        result = _regrid_to_gaussian(field, lat_in, lon_in, lat_out, lon_out)
        np.testing.assert_allclose(result, 42.0, atol=1e-12)

    def test_linear_latitude_field(self) -> None:
        lat_in = np.linspace(-90, 90, 181)
        lon_in = np.linspace(0, 360, 360, endpoint=False)
        field = np.broadcast_to(lat_in[:, None], (181, 360)).copy()
        grid = GaussianGrid(truncation=TRUNCATION)
        lat_out = np.degrees(np.asarray(grid.latitudes))
        lon_out = np.degrees(np.asarray(grid.longitudes))
        result = _regrid_to_gaussian(field, lat_in, lon_in, lat_out, lon_out)
        expected = np.broadcast_to(lat_out[:, None], result.shape)
        np.testing.assert_allclose(result, expected, atol=0.5)

    def test_longitude_wrapping(self) -> None:
        lat_in = np.linspace(-90, 90, 37)
        lon_in = np.linspace(0, 360, 72, endpoint=False)
        lon_rad = np.deg2rad(lon_in)
        field = np.broadcast_to(np.cos(lon_rad)[None, :], (37, 72)).copy()
        lat_out = np.array([0.0])
        lon_out = np.array([357.5])
        result = _regrid_to_gaussian(field, lat_in, lon_in, lat_out, lon_out)
        expected = np.cos(np.deg2rad(357.5))
        np.testing.assert_allclose(result[0, 0], expected, atol=0.01)

    def test_descending_latitude(self) -> None:
        lat_in = np.linspace(90, -90, 73)
        lon_in = np.linspace(0, 360, 144, endpoint=False)
        field = np.full((73, 144), 7.0)
        lat_out = np.array([45.0, 0.0, -45.0])
        lon_out = np.array([0.0, 180.0])
        result = _regrid_to_gaussian(field, lat_in, lon_in, lat_out, lon_out)
        np.testing.assert_allclose(result, 7.0, atol=1e-12)


# =====================================================================
# from_regular_latlon
# =====================================================================


class TestFromRegularLatlon:
    """Verify regular → Gaussian → spectral pipeline."""

    def test_round_trip_via_regular(self) -> None:
        """state → regular → state should approximately recover original."""
        transform, _, levels, state, _, _ = _setup()
        ds_reg = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=90,
            n_lon=180,
        )
        contents = from_regular_latlon(ds_reg, transform, EARTH)

        t_orig = jax.vmap(transform.spectral_to_grid)(state.temperature)
        t_rt = jax.vmap(transform.spectral_to_grid)(contents.state.temperature)
        np.testing.assert_allclose(np.asarray(t_rt), np.asarray(t_orig), rtol=0.02)

    def test_missing_variable_raises(self) -> None:
        import pytest

        ds = xr.Dataset(
            {"u": (["sigma", "lat", "lon"], np.zeros((3, 10, 20)))},
            coords={
                "sigma": np.linspace(0.05, 0.95, 3),
                "lat": np.linspace(-90, 90, 10),
                "lon": np.linspace(0, 360, 20, endpoint=False),
            },
        )
        transform, _, _, _, _, _ = _setup()
        with pytest.raises(KeyError, match="Missing required variable"):
            from_regular_latlon(ds, transform, EARTH)

    def test_negative_longitudes(self) -> None:
        transform, _, levels, state, _, _ = _setup()
        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=64,
            n_lon=128,
        )
        new_lon = (ds["lon"].values + 180) % 360 - 180
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
        contents = from_regular_latlon(ds, transform, EARTH)
        t_grid = jax.vmap(transform.spectral_to_grid)(contents.state.temperature)
        assert not np.any(np.isnan(np.asarray(t_grid)))

    def test_returns_dataset_contents(self) -> None:
        """from_regular_latlon should return a DatasetContents instance."""
        transform, _, levels, state, _, _ = _setup()
        ds_reg = to_regular_latlon(state, transform, EARTH, levels)
        contents = from_regular_latlon(ds_reg, transform, EARTH)
        assert isinstance(contents, DatasetContents)


# =====================================================================
# Humidity round-trip
# =====================================================================


class TestHumidityRoundTrip:
    """Test that humidity field survives conversion round-trips."""

    def test_gaussian_roundtrip_with_humidity(self) -> None:
        transform, _grid, levels, state, _, _ = _setup()
        q_spec = jnp.zeros_like(state.temperature) + 0.001
        state = state.replace(humidity=q_spec)

        ds = state_to_dataset(state, transform, EARTH, levels)
        assert "specific_humidity" in ds

        contents = dataset_to_state(ds, transform, EARTH)
        assert contents.state.has_humidity
        np.testing.assert_allclose(
            np.asarray(contents.state.humidity),
            np.asarray(state.humidity),
            atol=1e-8,
        )


# =====================================================================
# Static fields: orography, reference_temperature, q_flux, surface_properties
# =====================================================================


class TestOrographyRoundTrip:
    """Test surface geopotential round-tripping."""

    def test_gaussian_roundtrip(self) -> None:
        """Orography should survive Gaussian grid round-trip."""
        transform, _, levels, state, _, _ = _setup()
        surf_phi = gaussian_mountain(transform, EARTH)

        ds = state_to_dataset(
            state,
            transform,
            EARTH,
            levels,
            surface_geopotential=surf_phi,
        )
        assert "orography" in ds
        orog = ds["orography"].values
        assert orog.shape == (transform.grid.n_lat, transform.grid.n_lon)
        # Peak should be near 2000 m (default gaussian_mountain height)
        assert np.max(orog) > 1500.0

        contents = dataset_to_state(ds, transform, EARTH)
        assert contents.surface_geopotential is not None
        np.testing.assert_allclose(
            np.asarray(contents.surface_geopotential),
            np.asarray(surf_phi),
            atol=1e-6,
        )

    def test_regular_latlon_includes_orography(self) -> None:
        """Orography should appear in regular lat/lon output."""
        transform, _, levels, state, _, _ = _setup()
        surf_phi = gaussian_mountain(transform, EARTH)

        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=64,
            n_lon=128,
            surface_geopotential=surf_phi,
        )
        assert "orography" in ds
        assert ds["orography"].shape == (64, 128)
        assert np.max(ds["orography"].values) > 1500.0

    def test_from_regular_with_orography(self) -> None:
        """Orography in regular dataset should be extracted."""
        transform, _, levels, state, _, _ = _setup()
        surf_phi = gaussian_mountain(transform, EARTH)

        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=90,
            n_lon=180,
            surface_geopotential=surf_phi,
        )
        contents = from_regular_latlon(ds, transform, EARTH)
        assert contents.surface_geopotential is not None
        # Approximate: interpolation + spectral roundtrip loses precision
        orog_orig = transform.spectral_to_grid(surf_phi) / EARTH.gravity
        orog_rt = transform.spectral_to_grid(contents.surface_geopotential) / EARTH.gravity
        np.testing.assert_allclose(
            np.asarray(orog_rt),
            np.asarray(orog_orig),
            atol=50.0,
        )


class TestReferenceTemperatureRoundTrip:
    """Test reference temperature profile round-tripping."""

    def test_gaussian_roundtrip(self) -> None:
        transform, _, levels, state, ref_temps, _ = _setup()
        ref_temps = np.asarray(ref_temps)

        ds = state_to_dataset(
            state,
            transform,
            EARTH,
            levels,
            reference_temperature=ref_temps,
        )
        assert "reference_temperature" in ds
        assert ds["reference_temperature"].dims == ("sigma",)

        contents = dataset_to_state(ds, transform, EARTH)
        assert contents.reference_temperature is not None
        np.testing.assert_allclose(
            contents.reference_temperature,
            ref_temps,
        )

    def test_regular_latlon_includes_ref_temp(self) -> None:
        """reference_temperature is 1D, should pass through to regular grid."""
        transform, _, levels, state, ref_temps, _ = _setup()
        ref_temps = np.asarray(ref_temps)

        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            reference_temperature=ref_temps,
        )
        assert "reference_temperature" in ds
        np.testing.assert_allclose(
            ds["reference_temperature"].values,
            ref_temps,
        )


class TestQFluxRoundTrip:
    """Test q_flux round-tripping."""

    def test_gaussian_roundtrip(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        q_flux = jnp.linspace(-30.0, 30.0, grid.n_lat)

        ds = state_to_dataset(
            state,
            transform,
            EARTH,
            levels,
            q_flux=q_flux,
        )
        assert "q_flux" in ds
        assert ds["q_flux"].dims == ("lat",)
        np.testing.assert_allclose(
            ds["q_flux"].values,
            np.asarray(q_flux),
        )

        contents = dataset_to_state(ds, transform, EARTH)
        assert contents.q_flux is not None
        np.testing.assert_allclose(
            np.asarray(contents.q_flux),
            np.asarray(q_flux),
        )


class TestSurfacePropertiesRoundTrip:
    """Test SurfaceProperties round-tripping."""

    def test_gaussian_roundtrip(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        props = SurfaceProperties(
            land_fraction=jnp.zeros((grid.n_lat, grid.n_lon)),
            albedo=jnp.full((grid.n_lat, grid.n_lon), 0.06),
            z0_momentum=jnp.full((grid.n_lat, grid.n_lon), 1e-4),
            z0_heat=jnp.full((grid.n_lat, grid.n_lon), 1e-5),
        )

        ds = state_to_dataset(
            state,
            transform,
            EARTH,
            levels,
            surface_properties=props,
        )
        for var in ["land_fraction", "albedo", "z0_momentum", "z0_heat"]:
            assert var in ds, f"Missing variable: {var}"

        contents = dataset_to_state(ds, transform, EARTH)
        assert contents.surface_properties is not None
        np.testing.assert_allclose(
            np.asarray(contents.surface_properties.albedo),
            np.asarray(props.albedo),
        )
        np.testing.assert_allclose(
            np.asarray(contents.surface_properties.land_fraction),
            np.asarray(props.land_fraction),
        )

    def test_regular_latlon_with_properties(self) -> None:
        transform, grid, levels, state, _, _ = _setup()
        props = SurfaceProperties(
            land_fraction=jnp.zeros((grid.n_lat, grid.n_lon)),
            albedo=jnp.full((grid.n_lat, grid.n_lon), 0.06),
            z0_momentum=jnp.full((grid.n_lat, grid.n_lon), 1e-4),
            z0_heat=jnp.full((grid.n_lat, grid.n_lon), 1e-5),
        )

        ds = to_regular_latlon(
            state,
            transform,
            EARTH,
            levels,
            n_lat=64,
            n_lon=128,
            surface_properties=props,
        )
        assert "land_fraction" in ds
        assert "albedo" in ds
        # Uniform albedo should be close to 0.06 after spectral interpolation
        np.testing.assert_allclose(ds["albedo"].values, 0.06, atol=0.01)


class TestAllStaticFieldsTogether:
    """Test passing all optional fields at once."""

    def test_full_round_trip(self) -> None:
        transform, grid, levels, state, ref_temps, _ = _setup()
        ref_temps = np.asarray(ref_temps)
        surf_phi = gaussian_mountain(transform, EARTH)
        q_flux = jnp.linspace(-20.0, 20.0, grid.n_lat)
        ocean = OceanState(surface_temperature=jnp.full(grid.n_lat, 300.0))
        surface = SurfaceState(ocean=ocean)
        props = SurfaceProperties(
            land_fraction=jnp.zeros((grid.n_lat, grid.n_lon)),
            albedo=jnp.full((grid.n_lat, grid.n_lon), 0.06),
            z0_momentum=jnp.full((grid.n_lat, grid.n_lon), 1e-4),
            z0_heat=jnp.full((grid.n_lat, grid.n_lon), 1e-5),
        )

        ds = state_to_dataset(
            state,
            transform,
            EARTH,
            levels,
            surface=surface,
            surface_geopotential=surf_phi,
            reference_temperature=ref_temps,
            q_flux=q_flux,
            surface_properties=props,
        )

        # Verify all variables present
        expected_vars = [
            "u",
            "v",
            "temperature",
            "surface_pressure",
            "sst",
            "orography",
            "reference_temperature",
            "q_flux",
            "land_fraction",
            "albedo",
            "z0_momentum",
            "z0_heat",
        ]
        for var in expected_vars:
            assert var in ds, f"Missing variable: {var}"

        # Round-trip
        contents = dataset_to_state(ds, transform, EARTH)
        assert contents.surface is not None
        assert contents.surface_geopotential is not None
        assert contents.reference_temperature is not None
        assert contents.q_flux is not None
        assert contents.surface_properties is not None
