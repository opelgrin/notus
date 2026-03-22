"""Tests for the spherical harmonic transforms.

The gold standard test is the round-trip: forward then inverse (or vice versa)
should recover the original field to machine precision for fields within
the truncation bandwidth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gcm import SpectralTransform


# Enable float64 for precision in transforms.
jax.config.update("jax_enable_x64", True)


class TestRoundTrip:
    """Forward → inverse should recover the original grid-point field."""

    def test_constant_field(self, t21_transform: SpectralTransform):
        grid = t21_transform.grid
        field = jnp.ones((grid.n_lat, grid.n_lon)) * 42.0
        coeffs = t21_transform.grid_to_spectral(field)
        recovered = t21_transform.spectral_to_grid(coeffs)
        assert jnp.allclose(recovered, field, atol=1e-10)

    def test_spherical_harmonic_y11(self, t21_transform: SpectralTransform):
        """Y₁¹ ∝ cos(φ)·cos(λ) should survive the round trip.

        Unlike bare cos(λ), this IS a band-limited spherical harmonic.
        P̄₁¹(μ) = √(3/2) · cos(φ), so Y₁¹ = √(3/2) · cos(φ) · e^{iλ}
        and Re[Y₁¹] = √(3/2) · cos(φ) · cos(λ).
        """
        grid = t21_transform.grid
        lon = grid.longitudes[None, :]  # (1, n_lon)
        cos_lat = grid.cos_lat[:, None]  # (n_lat, 1)
        field = jnp.sqrt(3.0 / 2.0) * cos_lat * jnp.cos(lon)

        coeffs = t21_transform.grid_to_spectral(field)
        recovered = t21_transform.spectral_to_grid(coeffs)
        assert jnp.allclose(recovered, field, atol=1e-10)

    def test_spherical_harmonic_y20(self, t21_transform: SpectralTransform):
        """Y₂⁰ ∝ P̄₂⁰(sin φ) should be exactly representable.

        P̄₂⁰(x) = √(5/4) · (3x² - 1)  (fully normalized)
        """
        grid = t21_transform.grid
        x = grid.sin_lat
        p20 = jnp.sqrt(5.0 / 4.0) * (3.0 * x**2 - 1.0)
        # Y₂⁰ is purely zonal: constant in longitude
        field = p20[:, None] * jnp.ones((1, grid.n_lon))

        coeffs = t21_transform.grid_to_spectral(field)
        recovered = t21_transform.spectral_to_grid(coeffs)
        assert jnp.allclose(recovered, field, atol=1e-10)

    def test_random_field(self, t21_transform: SpectralTransform):
        """A random bandlimited field should survive the round trip."""
        grid = t21_transform.grid
        key = jax.random.PRNGKey(42)

        # Generate random spectral coefficients within truncation
        n_spec = grid.n_spectral_coeffs
        real = jax.random.normal(key, (n_spec,))
        key, subkey = jax.random.split(key)
        imag = jax.random.normal(subkey, (n_spec,))
        coeffs = real + 1j * imag

        # Enforce reality condition: m=0 coefficients must be real
        t = grid.truncation
        for n in range(t + 1):
            idx = grid.spectral_index(0, n)
            coeffs = coeffs.at[idx].set(coeffs[idx].real)

        # Round trip: spectral → grid → spectral
        field = t21_transform.spectral_to_grid(coeffs)
        recovered_coeffs = t21_transform.grid_to_spectral(field)
        assert jnp.allclose(recovered_coeffs, coeffs, atol=1e-8)


class TestSpectralToGridProperties:
    def test_output_is_real(self, t21_transform: SpectralTransform):
        """Inverse transform of valid coefficients should produce real field."""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        # Set a few coefficients (m=0 must be real)
        coeffs = coeffs.at[0].set(1.0)  # global mean
        field = t21_transform.spectral_to_grid(coeffs)
        assert jnp.allclose(jnp.imag(field), 0.0, atol=1e-14)

    def test_global_mean(self, t21_transform: SpectralTransform):
        """The n=0,m=0 coefficient should control the global mean."""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        # P̄₀⁰ = 1 everywhere, so setting f̂₀⁰ = c gives f(λ,φ) = c
        coeffs = coeffs.at[0].set(7.0 + 0j)
        field = t21_transform.spectral_to_grid(coeffs)
        assert jnp.allclose(field, 7.0, atol=1e-10)
