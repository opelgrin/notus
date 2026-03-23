"""Tests for the Gaussian grid."""

from __future__ import annotations

import math

import jax.numpy as jnp

from notus import GaussianGrid


class TestGaussianGridConstruction:
    def test_t21_default_sizes(self):
        grid = GaussianGrid(truncation=21)
        assert grid.truncation == 21
        assert grid.n_lat == 32
        assert grid.n_lon == 64

    def test_t42_default_sizes(self):
        grid = GaussianGrid(truncation=42)
        assert grid.n_lat == 64
        assert grid.n_lon == 128

    def test_custom_sizes(self):
        grid = GaussianGrid(truncation=21, n_lat=48, n_lon=96)
        assert grid.n_lat == 48
        assert grid.n_lon == 96

    def test_odd_n_lat_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="n_lat must be even"):
            GaussianGrid(truncation=21, n_lat=33)


class TestGaussianLatitudes:
    def test_latitudes_symmetric(self, t21_grid: GaussianGrid):
        """Gaussian latitudes should be symmetric about the equator."""
        n = t21_grid.n_lat
        north = t21_grid.latitudes[: n // 2]
        south = t21_grid.latitudes[n // 2 :][::-1]
        assert jnp.allclose(north, -south, atol=1e-14)

    def test_latitudes_range(self, t21_grid: GaussianGrid):
        """Latitudes should be in (-π/2, π/2), not touching the poles."""
        assert jnp.all(t21_grid.latitudes > -math.pi / 2)
        assert jnp.all(t21_grid.latitudes < math.pi / 2)
        # North-to-south ordering
        assert float(t21_grid.latitudes[0]) > 0
        assert float(t21_grid.latitudes[-1]) < 0

    def test_sin_lat_consistency(self, t21_grid: GaussianGrid):
        assert jnp.allclose(t21_grid.sin_lat, jnp.sin(t21_grid.latitudes), atol=1e-14)

    def test_cos_lat_positive(self, t21_grid: GaussianGrid):
        """cos(lat) should be strictly positive (no poles)."""
        assert jnp.all(t21_grid.cos_lat > 0)


class TestQuadratureWeights:
    def test_weights_sum_to_two(self, t21_grid: GaussianGrid):
        """Gauss-Legendre weights integrate 1 over [-1,1] → sum = 2."""
        assert jnp.isclose(jnp.sum(t21_grid.lat_weights), 2.0, atol=1e-14)

    def test_weights_positive(self, t21_grid: GaussianGrid):
        assert jnp.all(t21_grid.lat_weights > 0)


class TestLongitudes:
    def test_longitudes_range(self, t21_grid: GaussianGrid):
        """Longitudes should be in [0, 2π)."""
        assert jnp.isclose(t21_grid.longitudes[0], 0.0, atol=1e-15)
        assert float(t21_grid.longitudes[-1]) < 2 * math.pi

    def test_uniform_spacing(self, t21_grid: GaussianGrid):
        dlon = jnp.diff(t21_grid.longitudes)
        assert jnp.allclose(dlon, dlon[0], atol=1e-14)


class TestSpectralIndexing:
    def test_index_00(self, t21_grid: GaussianGrid):
        assert t21_grid.spectral_index(0, 0) == 0

    def test_index_monotonic(self, t21_grid: GaussianGrid):
        """Indices should increase with m and n."""
        t = t21_grid.truncation
        prev = -1
        for m in range(t + 1):
            for n in range(m, t + 1):
                idx = t21_grid.spectral_index(m, n)
                assert idx > prev
                prev = idx

    def test_total_count(self, t21_grid: GaussianGrid):
        t = t21_grid.truncation
        assert t21_grid.n_spectral_coeffs == (t + 1) * (t + 2) // 2
