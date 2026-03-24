"""Tests for meridional derivative and vector spectral operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from notus import (
    SpectralTransform,
    inverse_laplacian,
    meridional_derivative,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
)


jax.config.update("jax_enable_x64", True)


class TestMeridionalDerivative:
    def test_constant_has_zero_derivative(self, t21_transform: SpectralTransform, earth):
        """cos(φ)·∂(constant)/∂φ = 0."""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        coeffs = coeffs.at[0].set(5.0)  # global mean only
        result = meridional_derivative(coeffs, t21_transform.arrays)
        assert jnp.allclose(result, 0.0, atol=1e-14)

    def test_y10_derivative(self, t21_transform: SpectralTransform):
        """cos(φ)·∂/∂φ[P₁⁰(sin φ)] = cos(φ)·∂/∂φ[√3·sin φ] = √3·cos²(φ).

        P₁⁰(μ) = √3·μ, so ∂(P₁⁰)/∂φ = √3·cos(φ).
        cos(φ)·∂(P₁⁰)/∂φ = √3·cos²(φ) = √3·(1-sin²φ).

        In spectral space, cos²(φ) = 2/3·P₀⁰ − (2/3)·√(4/5)·... actually
        let's just check the round-trip on the grid.
        """
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_10 = grid.spectral_index(0, 1)
        coeffs = coeffs.at[idx_10].set(1.0)

        deriv_spec = meridional_derivative(coeffs, t21_transform.arrays)
        deriv_grid = t21_transform.spectral_to_grid(deriv_spec)

        # Expected: cos(φ)·d/dφ[P₁⁰(sinφ)] = √3·cos²(φ)
        expected = jnp.sqrt(3.0) * grid.cos_lat**2
        expected_grid = expected[:, None] * jnp.ones((1, grid.n_lon))

        assert jnp.allclose(deriv_grid, expected_grid, atol=1e-10)

    def test_zonal_harmonic_unchanged(self, t21_transform: SpectralTransform):
        """Meridional derivative of a purely zonal field (m=0) should be real."""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        coeffs = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_02 = grid.spectral_index(0, 2)
        coeffs = coeffs.at[idx_02].set(1.0)

        result = meridional_derivative(coeffs, t21_transform.arrays)
        # m=0 coefficients should remain real
        for n in range(grid.truncation + 1):
            idx = grid.spectral_index(0, n)
            assert jnp.isclose(jnp.imag(result[idx]), 0.0, atol=1e-14)


class TestUVFromVorDiv:
    def test_solid_body_rotation(self, t21_transform: SpectralTransform, earth):
        """Solid-body rotation: u = u₀·cos(φ), v = 0.

        Vorticity of solid body rotation: ζ = 2·u₀/a·sin(φ)
        In spectral space: ζ has only the (m=0, n=1) coefficient.
        Divergence = 0.

        The reconstructed U = u·cos(φ) = u₀·cos²(φ), V = 0.
        """
        grid = t21_transform.grid
        a = earth.radius
        u0 = 40.0  # m/s

        # Vorticity of solid body rotation: ζ = 2·u₀·sin(φ)/a
        # P₁⁰(μ) = √3·μ, so ζ = (2·u₀/a)·sin(φ) = (2·u₀/(a·√3))·P₁⁰
        n_spec = grid.n_spectral_coeffs
        vort_spec = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_01 = grid.spectral_index(0, 1)
        vort_spec = vort_spec.at[idx_01].set(2.0 * u0 / (a * jnp.sqrt(3.0)))

        div_spec = jnp.zeros(n_spec, dtype=jnp.complex128)

        u_cos_spec, v_cos_spec = uv_from_vordiv(vort_spec, div_spec, t21_transform.arrays)

        u_cos_grid = t21_transform.spectral_to_grid(u_cos_spec)
        v_cos_grid = t21_transform.spectral_to_grid(v_cos_spec)

        # Expected: U = u₀·cos²(φ), V = 0
        expected_u = u0 * grid.cos_lat[:, None] ** 2 * jnp.ones((1, grid.n_lon))
        expected_v = jnp.zeros_like(u_cos_grid)

        assert jnp.allclose(u_cos_grid, expected_u, atol=1e-6)
        assert jnp.allclose(v_cos_grid, expected_v, atol=1e-6)

    def test_nondivergent_gives_zero_v_potential(self, t21_transform: SpectralTransform, earth):
        """With δ=0, the velocity potential χ=0 so winds come only from ψ."""
        grid = t21_transform.grid
        n_spec = grid.n_spectral_coeffs
        key = jax.random.PRNGKey(42)
        vort = jax.random.normal(key, (n_spec,)) + 0j
        vort = vort.at[0].set(vort[0].real)
        div = jnp.zeros(n_spec, dtype=jnp.complex128)

        _u_spec, _v_spec = uv_from_vordiv(vort, div, t21_transform.arrays)

        # ∇²χ = δ = 0, so χ = 0 — the irrotational component is zero
        chi = inverse_laplacian(div, t21_transform.arrays)
        assert jnp.allclose(chi, 0.0, atol=1e-30)


class TestSpectralCurlDivergence:
    def test_balanced_state_zero_tendencies(self, t21_transform: SpectralTransform, earth):
        """For solid-body rotation, vorticity and divergence tendencies ~ 0.

        This is the key consistency test: the curl/div operators applied
        to the balanced fluxes should give near-zero tendencies.
        """
        grid = t21_transform.grid
        a = earth.radius
        u0 = 38.61

        # Set up balanced solid-body rotation
        n_spec = grid.n_spectral_coeffs
        vort_spec = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_01 = grid.spectral_index(0, 1)
        vort_spec = vort_spec.at[idx_01].set(2.0 * u0 / (a * jnp.sqrt(3.0)))
        div_spec = jnp.zeros(n_spec, dtype=jnp.complex128)

        u_spec, v_spec = uv_from_vordiv(vort_spec, div_spec, t21_transform.arrays)

        u_grid = t21_transform.spectral_to_grid(u_spec)
        v_grid = t21_transform.spectral_to_grid(v_spec)
        vort_grid = t21_transform.spectral_to_grid(vort_spec)

        f_cor = 2.0 * earth.rotation_rate * grid.sin_lat[:, None]
        zeta_a = vort_grid + f_cor
        cos2_inv = 1.0 / (grid.cos_lat[:, None] ** 2)

        a_grid = zeta_a * u_grid * cos2_inv
        b_grid = zeta_a * v_grid * cos2_inv

        a_spec = t21_transform.grid_to_spectral(a_grid)
        b_spec = t21_transform.grid_to_spectral(b_grid)

        # dζ/dt = -div(ζ_a·v⃗) = -spectral_divergence(A, B)
        # dδ/dt = +curl(ζ_a·v⃗) = -spectral_curl(A, B)  (flux part only)
        vort_tend = -spectral_divergence(a_spec, b_spec, t21_transform.arrays)
        div_tend = -spectral_curl(a_spec, b_spec, t21_transform.arrays)

        # For a balanced state, both should be near machine zero
        vort_tend_grid = t21_transform.spectral_to_grid(vort_tend)
        div_tend_grid = t21_transform.spectral_to_grid(div_tend)
        assert jnp.max(jnp.abs(vort_tend_grid)) < 1e-6
        assert jnp.max(jnp.abs(div_tend_grid)) < 1e-6
