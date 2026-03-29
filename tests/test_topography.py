"""Tests for idealized topography generators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus import (
    EARTH,
    SpectralTransform,
    gaussian_mountain,
    sinusoidal_mountains,
    zonal_ridge,
)


jax.config.update("jax_enable_x64", True)


class TestGaussianMountain:
    def test_peak_height(self, t42_transform: SpectralTransform):
        """Roundtrip spectral → grid recovers approximately the peak height."""
        h0 = 3000.0
        center_lat = np.pi / 4
        center_lon = np.pi

        phi_s = gaussian_mountain(
            t42_transform,
            EARTH,
            height=h0,
            center_lat=center_lat,
            center_lon=center_lon,
            half_width=np.pi / 6,
        )
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s)) / EARTH.gravity
        peak = float(z_grid.max())
        # Spectral truncation smooths the peak, but should be within 20%
        assert peak == pytest.approx(h0, rel=0.20)

    def test_peak_location(self, t42_transform: SpectralTransform):
        """Peak should be near the specified centre."""
        center_lat = np.pi / 6
        center_lon = np.pi

        phi_s = gaussian_mountain(
            t42_transform,
            EARTH,
            height=2000.0,
            center_lat=center_lat,
            center_lon=center_lon,
        )
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s)) / EARTH.gravity

        grid = t42_transform.grid
        lat = np.asarray(grid.latitudes)
        lon = np.asarray(grid.longitudes)
        peak_idx = np.unravel_index(z_grid.argmax(), z_grid.shape)
        assert abs(lat[peak_idx[0]] - center_lat) < np.pi / 20
        assert abs(lon[peak_idx[1]] - center_lon) < np.pi / 20

    def test_zero_height_gives_zero(self, t21_transform: SpectralTransform):
        """Zero height mountain should produce zero coefficients."""
        phi_s = gaussian_mountain(t21_transform, EARTH, height=0.0)
        assert jnp.allclose(phi_s, 0.0, atol=1e-30)

    def test_positive_everywhere(self, t21_transform: SpectralTransform):
        """A wide Gaussian should be non-negative on the grid (before Gibbs)."""
        phi_s = gaussian_mountain(
            t21_transform,
            EARTH,
            height=2000.0,
            half_width=np.pi / 4,
        )
        z_grid = np.asarray(t21_transform.spectral_to_grid(phi_s)) / EARTH.gravity
        # A wide mountain at T21 will have mild Gibbs ringing, but the
        # minimum should be close to zero
        assert z_grid.min() > -200.0

    def test_spectral_shape(self, t21_transform: SpectralTransform):
        """Output has the correct spectral shape."""
        phi_s = gaussian_mountain(t21_transform, EARTH)
        assert phi_s.shape == (t21_transform.grid.n_spectral_coeffs,)
        assert phi_s.dtype == jnp.complex128


class TestZonalRidge:
    def test_zonally_symmetric(self, t42_transform: SpectralTransform):
        """A zonal ridge should have no zonal variation."""
        phi_s = zonal_ridge(t42_transform, EARTH, height=2000.0)
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s))
        # Check that each latitude band is nearly constant
        zonal_std = np.std(z_grid, axis=1)
        assert np.all(zonal_std < 1e-6)

    def test_peak_latitude(self, t42_transform: SpectralTransform):
        """Peak should be near the specified centre latitude."""
        center_lat = -np.pi / 6  # Southern hemisphere
        phi_s = zonal_ridge(
            t42_transform,
            EARTH,
            height=2000.0,
            center_lat=center_lat,
            half_width=np.pi / 9,
        )
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s)) / EARTH.gravity
        zonal_mean = z_grid.mean(axis=1)
        lat = np.asarray(t42_transform.grid.latitudes)
        peak_lat = lat[np.argmax(zonal_mean)]
        assert abs(peak_lat - center_lat) < np.pi / 20

    def test_spectral_only_m0(self, t21_transform: SpectralTransform):
        """Zonal symmetry means only m=0 modes should be nonzero."""
        phi_s = zonal_ridge(t21_transform, EARTH, height=2000.0)
        grid = t21_transform.grid
        for m in range(1, grid.truncation + 1):
            for n in range(m, grid.truncation + 1):
                idx = grid.spectral_index(m, n)
                assert abs(phi_s[idx]) < 1e-10


class TestSinusoidalMountains:
    def test_wavenumber_2_pattern(self, t42_transform: SpectralTransform):
        """Wavenumber-2 pattern should have two peaks along the equator."""
        phi_s = sinusoidal_mountains(
            t42_transform,
            EARTH,
            height=2000.0,
            zonal_wavenumber=2,
        )
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s)) / EARTH.gravity

        # Find the latitude closest to the equator
        lat = np.asarray(t42_transform.grid.latitudes)
        eq_idx = np.argmin(np.abs(lat))
        equatorial_profile = z_grid[eq_idx, :]

        # FFT of equatorial profile: dominant mode should be wavenumber 2
        spectrum = np.abs(np.fft.rfft(equatorial_profile))
        dominant_k = np.argmax(spectrum[1:]) + 1  # skip DC
        assert dominant_k == 2

    def test_zero_at_poles(self, t42_transform: SpectralTransform):
        """cos²(φ) vanishes at the poles, so topography should too."""
        phi_s = sinusoidal_mountains(
            t42_transform,
            EARTH,
            height=2000.0,
            zonal_wavenumber=2,
        )
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s)) / EARTH.gravity
        # North and south pole rows
        assert np.all(np.abs(z_grid[0, :]) < 50.0)
        assert np.all(np.abs(z_grid[-1, :]) < 50.0)

    def test_global_mean_near_zero(self, t42_transform: SpectralTransform):
        """cos(k·λ) integrates to zero zonally for k≥1, so global mean ≈ 0."""
        phi_s = sinusoidal_mountains(
            t42_transform,
            EARTH,
            height=2000.0,
            zonal_wavenumber=2,
        )
        # The (0,0) spectral coefficient is proportional to the global mean
        assert abs(phi_s[0]) < 1e-6

    def test_wavenumber_1(self, t42_transform: SpectralTransform):
        """Wavenumber-1 should have a single peak along the equator."""
        phi_s = sinusoidal_mountains(
            t42_transform,
            EARTH,
            height=2000.0,
            zonal_wavenumber=1,
        )
        z_grid = np.asarray(t42_transform.spectral_to_grid(phi_s)) / EARTH.gravity
        lat = np.asarray(t42_transform.grid.latitudes)
        eq_idx = np.argmin(np.abs(lat))
        equatorial_profile = z_grid[eq_idx, :]

        # FFT of equatorial profile: dominant mode should be wavenumber 1
        spectrum = np.abs(np.fft.rfft(equatorial_profile))
        dominant_k = np.argmax(spectrum[1:]) + 1
        assert dominant_k == 1
