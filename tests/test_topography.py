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
    smooth_orography,
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


class TestSmoothOrography:
    def test_lanczos_preserves_global_mean(self, t42_transform: SpectralTransform):
        """Lanczos σ(n=0) = 1, so the global mean is unchanged."""
        phi_s = gaussian_mountain(t42_transform, EARTH, height=3000.0)
        arrays = t42_transform.arrays
        smoothed = smooth_orography(phi_s, arrays, method="lanczos")
        # (0,0) coefficient = global mean
        assert jnp.isclose(smoothed[0], phi_s[0], rtol=1e-14)

    def test_lanczos_damps_high_wavenumbers(self, t42_transform: SpectralTransform):
        """High-n coefficients should be smaller after Lanczos smoothing."""
        phi_s = gaussian_mountain(t42_transform, EARTH, height=3000.0, half_width=np.pi / 18)
        arrays = t42_transform.arrays
        smoothed = smooth_orography(phi_s, arrays, method="lanczos")

        # Energy in the upper half of the spectrum should decrease
        grid = t42_transform.grid
        t = grid.truncation
        high_n_mask = np.asarray(arrays.n_index) > t // 2
        raw_energy = float(jnp.sum(jnp.abs(phi_s[high_n_mask]) ** 2))
        smooth_energy = float(jnp.sum(jnp.abs(smoothed[high_n_mask]) ** 2))
        assert smooth_energy < raw_energy

    def test_lanczos_reduces_gibbs_ringing(self, t21_transform: SpectralTransform):
        """Smoothing a narrow mountain should reduce negative undershoots."""
        phi_s = gaussian_mountain(t21_transform, EARTH, height=4000.0, half_width=np.pi / 18)
        arrays = t21_transform.arrays
        z_raw = np.asarray(t21_transform.spectral_to_grid(phi_s)) / EARTH.gravity
        smoothed = smooth_orography(phi_s, arrays, method="lanczos")
        z_smooth = np.asarray(t21_transform.spectral_to_grid(smoothed)) / EARTH.gravity
        assert z_smooth.min() > z_raw.min()

    def test_exponential_preserves_global_mean(self, t42_transform: SpectralTransform):
        """Exponential filter with k=0 → exp(0) = 1, so global mean is unchanged."""
        phi_s = gaussian_mountain(t42_transform, EARTH, height=3000.0)
        arrays = t42_transform.arrays
        smoothed = smooth_orography(phi_s, arrays, method="exponential")
        assert jnp.isclose(smoothed[0], phi_s[0], rtol=1e-14)

    def test_exponential_damps_high_wavenumbers(self, t42_transform: SpectralTransform):
        """Exponential filter should damp high-n modes."""
        phi_s = gaussian_mountain(t42_transform, EARTH, height=3000.0, half_width=np.pi / 18)
        arrays = t42_transform.arrays
        smoothed = smooth_orography(phi_s, arrays, method="exponential")

        grid = t42_transform.grid
        t = grid.truncation
        high_n_mask = np.asarray(arrays.n_index) > t // 2
        raw_energy = float(jnp.sum(jnp.abs(phi_s[high_n_mask]) ** 2))
        smooth_energy = float(jnp.sum(jnp.abs(smoothed[high_n_mask]) ** 2))
        assert smooth_energy < raw_energy

    def test_higher_order_damps_more(self, t42_transform: SpectralTransform):
        """Higher order should produce more damping at high wavenumbers."""
        phi_s = gaussian_mountain(t42_transform, EARTH, height=3000.0, half_width=np.pi / 18)
        arrays = t42_transform.arrays
        smooth1 = smooth_orography(phi_s, arrays, method="lanczos", order=1)
        smooth2 = smooth_orography(phi_s, arrays, method="lanczos", order=2)

        grid = t42_transform.grid
        t = grid.truncation
        high_n_mask = np.asarray(arrays.n_index) > t // 2
        energy1 = float(jnp.sum(jnp.abs(smooth1[high_n_mask]) ** 2))
        energy2 = float(jnp.sum(jnp.abs(smooth2[high_n_mask]) ** 2))
        assert energy2 < energy1

    def test_zero_input_gives_zero(self, t21_transform: SpectralTransform):
        """Smoothing zero topography returns zero."""
        arrays = t21_transform.arrays
        n_spec = t21_transform.grid.n_spectral_coeffs
        phi_s = jnp.zeros(n_spec, dtype=jnp.complex128)
        smoothed = smooth_orography(phi_s, arrays, method="lanczos")
        assert jnp.allclose(smoothed, 0.0, atol=1e-30)

    def test_invalid_method_raises(self, t21_transform: SpectralTransform):
        """Unknown method should raise ValueError."""
        arrays = t21_transform.arrays
        n_spec = t21_transform.grid.n_spectral_coeffs
        phi_s = jnp.zeros(n_spec, dtype=jnp.complex128)
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smooth_orography(phi_s, arrays, method="bogus")
