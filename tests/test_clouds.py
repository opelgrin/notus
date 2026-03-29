"""Tests for diagnostic cloud scheme."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)

from notus.constants import EARTH
from notus.physics.clouds import CloudDiagnostic, diagnose_clouds
from notus.physics.radiation import (
    speedy_longwave_heating,
    speedy_shortwave_heating,
)
from notus.vertical.sigma import standard_sigma_levels


@pytest.fixture
def levels():
    return standard_sigma_levels(20)


class TestCloudDiagnostic:
    """Basic tests for cloud diagnosis output."""

    def test_shape_and_finite(self, levels) -> None:
        """Cloud fields should have correct shape and be finite."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        rh = jnp.full((n_levels, n_lat, n_lon), 0.5)
        q = jnp.full((n_levels, n_lat, n_lon), 0.005)
        t = jnp.full((n_levels, n_lat, n_lon), 260.0)
        phi = EARTH.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t)
        precip = jnp.zeros((n_lat, n_lon))
        conv_mask = jnp.zeros_like(t, dtype=bool)

        cloud = diagnose_clouds(
            rh,
            q,
            t,
            phi,
            precip,
            conv_mask,
            gravity=EARTH.gravity,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        assert cloud.cloud_cover.shape == (n_lat, n_lon)
        assert cloud.cloud_top.shape == (n_lat, n_lon)
        assert cloud.stratiform_cover.shape == (n_lat, n_lon)
        assert cloud.cloud_humidity.shape == (n_lat, n_lon)
        assert jnp.all(jnp.isfinite(cloud.cloud_cover))
        assert jnp.all(jnp.isfinite(cloud.stratiform_cover))

    def test_cloud_cover_bounded(self, levels) -> None:
        """Cloud cover should be in [0, 1]."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        rh = jnp.full((n_levels, n_lat, n_lon), 0.8)
        q = jnp.full((n_levels, n_lat, n_lon), 0.005)
        t = jnp.full((n_levels, n_lat, n_lon), 260.0)
        phi = EARTH.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t)
        precip = jnp.full((n_lat, n_lon), 5e-5)  # moderate precip
        conv_mask = jnp.zeros_like(t, dtype=bool)

        cloud = diagnose_clouds(
            rh,
            q,
            t,
            phi,
            precip,
            conv_mask,
            gravity=EARTH.gravity,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        assert jnp.all(cloud.cloud_cover >= 0.0)
        assert jnp.all(cloud.cloud_cover <= 1.0)
        assert jnp.all(cloud.stratiform_cover >= 0.0)
        assert jnp.all(cloud.stratiform_cover <= 1.0)


class TestCloudCoverPhysics:
    """Test cloud cover responds to physical inputs."""

    def test_dry_gives_no_clouds(self, levels) -> None:
        """Very low RH and no precipitation should give near-zero cloud cover."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        rh = jnp.full((n_levels, n_lat, n_lon), 0.1)
        q = jnp.full((n_levels, n_lat, n_lon), 0.0001)  # below qacl
        t = jnp.full((n_levels, n_lat, n_lon), 260.0)
        phi = EARTH.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t)
        precip = jnp.zeros((n_lat, n_lon))
        conv_mask = jnp.zeros_like(t, dtype=bool)

        cloud = diagnose_clouds(
            rh,
            q,
            t,
            phi,
            precip,
            conv_mask,
            gravity=EARTH.gravity,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        np.testing.assert_allclose(cloud.cloud_cover, 0.0, atol=1e-10)

    def test_high_rh_gives_clouds(self, levels) -> None:
        """High RH in the troposphere should produce nonzero cloud cover."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        rh = jnp.full((n_levels, n_lat, n_lon), 0.85)
        q = jnp.full((n_levels, n_lat, n_lon), 0.005)  # above qacl
        t = jnp.full((n_levels, n_lat, n_lon), 260.0)
        phi = EARTH.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t)
        precip = jnp.zeros((n_lat, n_lon))
        conv_mask = jnp.zeros_like(t, dtype=bool)

        cloud = diagnose_clouds(
            rh,
            q,
            t,
            phi,
            precip,
            conv_mask,
            gravity=EARTH.gravity,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        assert jnp.all(cloud.cloud_cover > 0.0)

    def test_precipitation_adds_clouds(self, levels) -> None:
        """Precipitation should increase cloud cover."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        rh = jnp.full((n_levels, n_lat, n_lon), 0.2)  # low RH
        q = jnp.full((n_levels, n_lat, n_lon), 0.0001)
        t = jnp.full((n_levels, n_lat, n_lon), 260.0)
        phi = EARTH.gravity * levels.sigma_full[:, None, None] * jnp.ones_like(t)
        conv_mask = jnp.zeros_like(t, dtype=bool)

        cloud_dry = diagnose_clouds(
            rh,
            q,
            t,
            phi,
            jnp.zeros((n_lat, n_lon)),
            conv_mask,
            gravity=EARTH.gravity,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        cloud_wet = diagnose_clouds(
            rh,
            q,
            t,
            phi,
            jnp.full((n_lat, n_lon), 1e-4),
            conv_mask,
            gravity=EARTH.gravity,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        assert jnp.all(cloud_wet.cloud_cover >= cloud_dry.cloud_cover)


class TestCloudRadiationCoupling:
    """Test that clouds modify SPEEDY radiation as expected."""

    def test_clouds_reduce_sw_at_surface(self, levels) -> None:
        """Clouds should reduce SW reaching the surface."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        q = jnp.full((n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)
        insol = jnp.full(n_lat, 340.0)

        _, sw_clear = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )

        cloud = CloudDiagnostic(
            cloud_cover=jnp.full((n_lat, n_lon), 0.5),
            cloud_top=jnp.full((n_lat, n_lon), 5, dtype=jnp.int32),
            stratiform_cover=jnp.full((n_lat, n_lon), 0.2),
            cloud_humidity=jnp.full((n_lat, n_lon), 0.005),
        )
        _, sw_cloudy = speedy_shortwave_heating(
            levels.dsigma,
            levels.sigma_full,
            q,
            ps,
            1e5,
            insol,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            cloud=cloud,
        )

        # Clouds should reduce SW at surface
        assert jnp.all(sw_cloudy < sw_clear)

    def test_clouds_increase_lw_down(self, levels) -> None:
        """Clouds should increase downward LW at the surface (greenhouse)."""
        n_lat, n_lon = 4, 8
        n_levels = levels.n_levels
        t = jnp.full((n_levels, n_lat, n_lon), 260.0)
        t_s = jnp.full(n_lat, 280.0)
        q = jnp.full((n_levels, n_lat, n_lon), 0.005)
        ps = jnp.full((n_lat, n_lon), 1e5)

        _, lw_clear, _ = speedy_longwave_heating(
            t,
            t_s,
            q,
            levels.dsigma,
            ps,
            1e5,
            EARTH.gravity,
            EARTH.specific_heat_cp,
        )

        cloud = CloudDiagnostic(
            cloud_cover=jnp.full((n_lat, n_lon), 0.5),
            cloud_top=jnp.full((n_lat, n_lon), 5, dtype=jnp.int32),
            stratiform_cover=jnp.full((n_lat, n_lon), 0.2),
            cloud_humidity=jnp.full((n_lat, n_lon), 0.005),
        )
        _, lw_cloudy, _ = speedy_longwave_heating(
            t,
            t_s,
            q,
            levels.dsigma,
            ps,
            1e5,
            EARTH.gravity,
            EARTH.specific_heat_cp,
            cloud=cloud,
        )

        # Clouds should increase LW down at surface
        assert jnp.all(lw_cloudy > lw_clear)
