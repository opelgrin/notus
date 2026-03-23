"""Tests for vertical operators (hydrostatic geopotential)."""

import jax
import jax.numpy as jnp
import numpy as np

from notus.sigma import uniform_sigma_levels
from notus.vertical import geopotential, geopotential_weights, sigma_ratios


jax.config.update("jax_enable_x64", True)

# Earth-like gas constant for tests
R = 287.04


class TestSigmaRatios:
    """Tests for the sigma ratio (α) computation."""

    def test_uniform_4_levels(self) -> None:
        """Verify α values for uniform 4-level case against hand computation.

        Uniform 4 levels: interfaces [0, 0.25, 0.5, 0.75, 1.0]
        Centers (midpoints): [0.125, 0.375, 0.625, 0.875]

        α[0] = ln(0.375 / 0.125) / 2 = ln(3) / 2
        α[1] = ln(0.625 / 0.375) / 2 = ln(5/3) / 2
        α[2] = ln(0.875 / 0.625) / 2 = ln(7/5) / 2
        α[3] = -ln(0.875)
        """
        levels = uniform_sigma_levels(4)
        alpha = sigma_ratios(levels)

        assert alpha.shape == (4,)
        np.testing.assert_allclose(alpha[0], np.log(3.0) / 2, rtol=1e-14)
        np.testing.assert_allclose(alpha[1], np.log(5.0 / 3.0) / 2, rtol=1e-14)
        np.testing.assert_allclose(alpha[2], np.log(7.0 / 5.0) / 2, rtol=1e-14)
        np.testing.assert_allclose(alpha[3], -np.log(0.875), rtol=1e-14)

    def test_all_positive(self) -> None:
        """All α values should be positive."""
        levels = uniform_sigma_levels(20)
        alpha = sigma_ratios(levels)
        assert np.all(alpha > 0)


class TestGeopotentialWeights:
    """Tests for the G matrix construction."""

    def test_shape(self) -> None:
        levels = uniform_sigma_levels(5)
        g = geopotential_weights(levels, R)
        assert g.shape == (5, 5)

    def test_structure_element_by_element(self) -> None:
        """Verify G matrix entries: zero below diagonal, α[j] on diagonal,
        α[j]+α[j-1] above diagonal."""
        levels = uniform_sigma_levels(6)
        g = geopotential_weights(levels, R)
        alpha = sigma_ratios(levels)
        n = levels.n_levels

        for i in range(n):
            for j in range(n):
                if i > j:
                    expected = 0.0
                elif i == j:
                    expected = R * alpha[j]
                else:
                    expected = R * (alpha[j] + alpha[j - 1])
                np.testing.assert_allclose(
                    g[i, j], expected, rtol=1e-14,
                    err_msg=f"Mismatch at G[{i},{j}]",
                )

    def test_upper_triangular(self) -> None:
        """G should be zero below the diagonal."""
        levels = uniform_sigma_levels(10)
        g = geopotential_weights(levels, R)
        for i in range(levels.n_levels):
            for j in range(i):
                np.testing.assert_equal(
                    g[i, j], 0.0, err_msg=f"G[{i},{j}] should be zero"
                )


class TestGeopotential:
    """Tests for the full geopotential computation."""

    def test_isothermal_atmosphere(self) -> None:
        """Isothermal atmosphere on flat surface.

        Continuous solution: Φ(σ) = -R·T·ln(σ)
        The discrete approximation should converge toward this with more levels.
        Test with many levels for good accuracy.
        """
        n_levels = 100
        levels = uniform_sigma_levels(n_levels)
        t_const = 250.0

        # Temperature array: constant at all levels, single spectral coeff
        temperature = jnp.full((n_levels, 1), t_const)
        surface_phi = jnp.zeros(1)

        phi = geopotential(temperature, surface_phi, levels, R)

        # Continuous solution at each full level
        expected = -R * t_const * jnp.log(levels.sigma_full)

        # Should be close with 100 levels (discretization error ~ 1/L²)
        np.testing.assert_allclose(phi[:, 0], expected, rtol=1e-3)

    def test_isothermal_near_exact(self) -> None:
        """Durran discretization is exact for isothermal T to machine precision."""
        t_const = 250.0
        levels = uniform_sigma_levels(5)
        temperature = jnp.full((5, 1), t_const)
        surface_phi = jnp.zeros(1)
        phi = geopotential(temperature, surface_phi, levels, R)
        expected = -R * t_const * jnp.log(levels.sigma_full)
        np.testing.assert_allclose(phi[:, 0], expected, rtol=1e-13)

    def test_surface_geopotential_offset(self) -> None:
        """Non-zero surface geopotential should shift all levels by a constant."""
        n_levels = 20
        levels = uniform_sigma_levels(n_levels)
        temperature = jnp.full((n_levels, 1), 250.0)

        phi_flat = geopotential(temperature, jnp.zeros(1), levels, R)

        # Mountain with Φ_s = g * 2000m
        phi_s = jnp.array([9.80616 * 2000.0])
        phi_mountain = geopotential(temperature, phi_s, levels, R)

        # Difference should be exactly Φ_s at every level
        diff = phi_mountain - phi_flat
        np.testing.assert_allclose(diff[:, 0], phi_s[0], rtol=1e-14)

    def test_geopotential_increases_upward(self) -> None:
        """For positive T, geopotential should increase from surface to top."""
        n_levels = 20
        levels = uniform_sigma_levels(n_levels)
        # Realistic temperature profile decreasing with height
        t_profile = jnp.linspace(300.0, 200.0, n_levels)
        temperature = t_profile[:, None] * jnp.ones((1, 3))
        surface_phi = jnp.zeros(3)

        phi = geopotential(temperature, surface_phi, levels, R)

        # Φ[0] (top) > Φ[1] > ... > Φ[L-1] (bottom) for real part
        for k in range(n_levels - 1):
            assert float(phi[k, 0].real) > float(phi[k + 1, 0].real), (
                f"Φ[{k}] should be > Φ[{k + 1}]"
            )

    def test_shape(self) -> None:
        """Output shape should match (n_levels, n_spectral)."""
        n_levels = 10
        n_spectral = 42
        levels = uniform_sigma_levels(n_levels)
        temperature = jnp.ones((n_levels, n_spectral))
        surface_phi = jnp.zeros(n_spectral)

        phi = geopotential(temperature, surface_phi, levels, R)
        assert phi.shape == (n_levels, n_spectral)

    def test_single_level(self) -> None:
        """Single level: Φ = Φ_s + R·T·α[0]."""
        levels = uniform_sigma_levels(1)
        alpha = sigma_ratios(levels)
        t_val = 300.0
        temperature = jnp.array([[t_val]])
        surface_phi = jnp.array([1000.0])

        phi = geopotential(temperature, surface_phi, levels, R)

        expected = 1000.0 + R * t_val * alpha[0]
        np.testing.assert_allclose(phi[0, 0], expected, rtol=1e-14)

    def test_zero_temperature(self) -> None:
        """Zero temperature should give Φ = Φ_s at every level."""
        n_levels = 10
        levels = uniform_sigma_levels(n_levels)
        temperature = jnp.zeros((n_levels, 5))
        surface_phi = jnp.full(5, 42.0)

        phi = geopotential(temperature, surface_phi, levels, R)

        for k in range(n_levels):
            np.testing.assert_allclose(phi[k], 42.0, rtol=1e-14)
