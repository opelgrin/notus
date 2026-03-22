"""Tests for spectral operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gcm import SpectralTransform, inverse_laplacian, laplacian


jax.config.update("jax_enable_x64", True)


class TestLaplacian:
    def test_eigenvalue_y20(self, t21_transform: SpectralTransform, earth):
        """∇²Y₂⁰ = -n(n+1)/a² · Y₂⁰ = -6/a² · Y₂⁰"""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_20 = grid.spectral_index(0, 2)
        coeffs = coeffs.at[idx_20].set(1.0 + 0j)

        result = laplacian(coeffs, grid.truncation, earth.radius)

        expected_eigenvalue = -2 * 3 / earth.radius**2
        assert jnp.isclose(result[idx_20], expected_eigenvalue, rtol=1e-12)
        # All other coefficients should be zero
        assert jnp.isclose(jnp.abs(result.at[idx_20].set(0.0)).max(), 0.0, atol=1e-30)

    def test_laplacian_of_constant_is_zero(self, t21_transform: SpectralTransform, earth):
        """∇²(constant) = 0."""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        coeffs = coeffs.at[0].set(5.0 + 0j)  # n=0 mode only

        result = laplacian(coeffs, grid.truncation, earth.radius)
        assert jnp.allclose(result, 0.0, atol=1e-30)


class TestInverseLaplacian:
    def test_roundtrip(self, t21_transform: SpectralTransform, earth):
        """∇²(∇⁻²f) = f for fields with zero global mean."""
        grid = t21_transform.grid
        key = jax.random.PRNGKey(123)
        n_spec = grid.n_spectral_coeffs
        coeffs = jax.random.normal(key, (n_spec,)) + 0j
        # Zero out global mean (n=0 mode)
        coeffs = coeffs.at[0].set(0.0)
        # Make m=0 real
        t = grid.truncation
        for n in range(t + 1):
            idx = grid.spectral_index(0, n)
            coeffs = coeffs.at[idx].set(coeffs[idx].real)

        inv = inverse_laplacian(coeffs, grid.truncation, earth.radius)
        recovered = laplacian(inv, grid.truncation, earth.radius)
        assert jnp.allclose(recovered, coeffs, atol=1e-8)
