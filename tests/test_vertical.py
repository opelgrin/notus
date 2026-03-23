"""Tests for vertical operators (hydrostatic geopotential)."""

import jax
import jax.numpy as jnp
import numpy as np

from notus.sigma import uniform_sigma_levels
from notus.vertical import (
    geopotential,
    geopotential_weights,
    sigma_dot,
    sigma_ratios,
    surface_pressure_tendency,
)


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
                    g[i, j],
                    expected,
                    rtol=1e-14,
                    err_msg=f"Mismatch at G[{i},{j}]",
                )

    def test_upper_triangular(self) -> None:
        """G should be zero below the diagonal."""
        levels = uniform_sigma_levels(10)
        g = geopotential_weights(levels, R)
        for i in range(levels.n_levels):
            for j in range(i):
                np.testing.assert_equal(g[i, j], 0.0, err_msg=f"G[{i},{j}] should be zero")


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


class TestSurfacePressureTendency:
    """Tests for the surface pressure tendency computation."""

    def test_uniform_divergence(self) -> None:
        """Uniform D at all levels → tendency = -D (since Σ Δσ = 1)."""
        n_levels = 10
        levels = uniform_sigma_levels(n_levels)
        d_val = 3.5
        column_div = jnp.full((n_levels, 4), d_val)

        tend = surface_pressure_tendency(column_div, levels)

        np.testing.assert_allclose(tend, -d_val, rtol=1e-14)

    def test_zero_divergence(self) -> None:
        """Zero divergence → zero tendency."""
        levels = uniform_sigma_levels(5)
        column_div = jnp.zeros((5, 3))

        tend = surface_pressure_tendency(column_div, levels)

        np.testing.assert_allclose(tend, 0.0, atol=1e-15)

    def test_single_nonzero_level(self) -> None:
        """Divergence only at level k → tendency = -D_k · Δσ_k."""
        levels = uniform_sigma_levels(4)
        column_div = jnp.zeros((4, 1))
        column_div = column_div.at[2, 0].set(8.0)

        tend = surface_pressure_tendency(column_div, levels)

        expected = -8.0 * float(levels.dsigma[2])
        np.testing.assert_allclose(tend[0], expected, rtol=1e-14)

    def test_shape(self) -> None:
        """(n_levels, n_spectral) → (n_spectral,)."""
        levels = uniform_sigma_levels(8)
        column_div = jnp.ones((8, 42))

        tend = surface_pressure_tendency(column_div, levels)

        assert tend.shape == (42,)

    def test_single_level(self) -> None:
        """Single level: tendency = -D (Δσ = 1)."""
        levels = uniform_sigma_levels(1)
        column_div = jnp.array([[5.0]])

        tend = surface_pressure_tendency(column_div, levels)

        np.testing.assert_allclose(tend[0], -5.0, rtol=1e-14)


class TestSigmaDot:
    """Tests for the sigma-dot (vertical velocity) computation."""

    def test_boundary_conditions(self) -> None:
        """σ̇ = 0 at top and surface for any input."""
        levels = uniform_sigma_levels(10)
        # Random-ish divergence profile
        column_div = jnp.array([1, -2, 3, -1, 2, -3, 1, 0, 2, -1])[:, None]

        sd = sigma_dot(column_div, levels)

        np.testing.assert_equal(float(sd[0, 0]), 0.0)
        np.testing.assert_equal(float(sd[-1, 0]), 0.0)

    def test_uniform_divergence_gives_zero(self) -> None:
        """Uniform D with uniform levels → σ̇ = 0 everywhere.

        Proof: C_k = D·k/L, C_L = D, σ_{k+1/2} = k/L
        → σ̇ = (k/L)·D - D·k/L = 0.
        """
        n_levels = 8
        levels = uniform_sigma_levels(n_levels)
        column_div = jnp.full((n_levels, 3), 2.5)

        sd = sigma_dot(column_div, levels)

        np.testing.assert_allclose(sd, 0.0, atol=1e-14)

    def test_hand_computed_3_levels(self) -> None:
        """3 uniform levels, D = [3, 0, 0].

        Δσ = 1/3, interfaces [0, 1/3, 2/3, 1]
        C_1 = 3·(1/3) = 1, C_2 = 1, C_3 = 1 = C_L
        σ̇[0] = 0                         (top)
        σ̇[1] = (1/3)·1 - 1 = -2/3
        σ̇[2] = (2/3)·1 - 1 = -1/3
        σ̇[3] = 0                         (surface)
        """
        levels = uniform_sigma_levels(3)
        column_div = jnp.array([[3.0], [0.0], [0.0]])

        sd = sigma_dot(column_div, levels)

        expected = jnp.array([[0.0], [-2.0 / 3.0], [-1.0 / 3.0], [0.0]])
        np.testing.assert_allclose(sd, expected, atol=1e-14)

    def test_shape(self) -> None:
        """(n_levels, ...) → (n_levels + 1, ...)."""
        levels = uniform_sigma_levels(6)
        column_div = jnp.ones((6, 10))

        sd = sigma_dot(column_div, levels)

        assert sd.shape == (7, 10)

    def test_single_level(self) -> None:
        """Single level: σ̇ = [0, 0] (only boundaries)."""
        levels = uniform_sigma_levels(1)
        column_div = jnp.array([[7.0]])

        sd = sigma_dot(column_div, levels)

        assert sd.shape == (2, 1)
        np.testing.assert_allclose(sd, 0.0, atol=1e-15)

    def test_consistency_with_surface_tendency(self) -> None:
        """σ̇ at surface = 0 is consistent with the surface pressure tendency.

        The formula σ̇_L = σ_L · C_L - C_L = 0 holds because σ_L = 1.
        Verify that sigma_dot[-1] = 0 even when the surface pressure
        tendency is large.
        """
        levels = uniform_sigma_levels(5)
        column_div = jnp.array([10.0, -5.0, 3.0, -8.0, 20.0])[:, None]

        tend = surface_pressure_tendency(column_div, levels)
        sd = sigma_dot(column_div, levels)

        # Surface pressure tendency should be nonzero
        assert float(jnp.abs(tend[0])) > 0
        # But σ̇ at surface is exactly 0
        np.testing.assert_allclose(float(sd[-1, 0]), 0.0, atol=1e-13)
