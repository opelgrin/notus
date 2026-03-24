"""Tests for conservation diagnostics.

Validates:
1. Spherical and sigma integration against analytic results
2. Conservation properties of the J-W steady state under integration
3. Conservation properties of the perturbed J-W baroclinic wave
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import EARTH
from notus.diagnostics import (
    compute_conservation_diagnostics,
    sigma_integral,
    spherical_integral,
)
from notus.grid import GaussianGrid
from notus.initial_conditions import (
    jablonowski_williamson_perturbation,
    jablonowski_williamson_steady_state,
)
from notus.operators import exponential_filter
from notus.sigma import uniform_sigma_levels
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform


jax.config.update("jax_enable_x64", True)


# =====================================================================
# Integration unit tests
# =====================================================================


class TestSphericalIntegral:
    """Verify spherical_integral against analytic results."""

    def test_constant_field(self) -> None:
        """Integral of 1 over the sphere should equal 4π (unit sphere area)."""
        grid = GaussianGrid(truncation=21)
        field = jnp.ones((grid.n_lat, grid.n_lon))
        result = float(spherical_integral(field, grid))
        np.testing.assert_allclose(result, 4.0 * np.pi, rtol=1e-14)

    def test_cos_squared_lat(self) -> None:
        """Integral of cos²φ over the sphere should equal 8π/3."""
        grid = GaussianGrid(truncation=21)
        cos2 = grid.cos_lat[:, None] ** 2 * jnp.ones(grid.n_lon)[None, :]
        result = float(spherical_integral(cos2, grid))
        np.testing.assert_allclose(result, 8.0 * np.pi / 3.0, rtol=1e-14)

    def test_spherical_harmonic_orthogonality(self) -> None:
        """Integral of Y_2^0 over the sphere should be zero (orthogonal to Y_0^0)."""
        grid = GaussianGrid(truncation=21)
        # P_2(sinφ) = (3sin²φ - 1)/2
        p2 = 0.5 * (3.0 * grid.sin_lat**2 - 1.0)
        field = p2[:, None] * jnp.ones(grid.n_lon)[None, :]
        result = float(spherical_integral(field, grid))
        np.testing.assert_allclose(result, 0.0, atol=1e-13)

    def test_batched_integral(self) -> None:
        """Integration should work with leading batch dimensions."""
        grid = GaussianGrid(truncation=21)
        field = jnp.ones((3, grid.n_lat, grid.n_lon))
        result = spherical_integral(field, grid)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 4.0 * np.pi, rtol=1e-14)


class TestSigmaIntegral:
    """Verify sigma_integral against analytic results."""

    def test_constant_field(self) -> None:
        """Integral of 1 from σ=0 to σ=1 should be 1."""
        levels = uniform_sigma_levels(10)
        field = jnp.ones((10, 4, 8))
        result = sigma_integral(field, levels)
        np.testing.assert_allclose(result, 1.0, rtol=1e-14)

    def test_linear_field(self) -> None:
        """Integral of σ from 0 to 1 should be 0.5."""
        levels = uniform_sigma_levels(20)
        # f(σ) = σ at full levels
        sigma_vals = jnp.array(levels.sigma_full)
        field = sigma_vals[:, None, None] * jnp.ones((1, 4, 8))
        result = sigma_integral(field, levels)
        np.testing.assert_allclose(result, 0.5, rtol=1e-3)

    def test_1d_field(self) -> None:
        """Should work with 1D input (n_levels,)."""
        levels = uniform_sigma_levels(5)
        field = jnp.ones(5)
        result = float(sigma_integral(field, levels))
        np.testing.assert_allclose(result, 1.0, rtol=1e-14)


class TestSigmaIntegralAdditional:
    """Additional sigma integration tests against analytic solutions."""

    def test_quadratic_field(self) -> None:
        """Integral of σ² from 0 to 1 should be 1/3."""
        levels = uniform_sigma_levels(20)
        sigma_vals = jnp.array(levels.sigma_full)
        field = sigma_vals**2
        result = float(sigma_integral(field, levels))
        np.testing.assert_allclose(result, 1.0 / 3.0, rtol=1e-3)

    def test_integral_converges_with_resolution(self) -> None:
        """Sigma integral of σ³ should converge to 1/4 with more levels."""
        # σ³ is nonlinear, so midpoint rule has finite error
        errors = []
        for n in [5, 10, 20, 40]:
            levels = uniform_sigma_levels(n)
            sigma_vals = jnp.array(levels.sigma_full)
            result = float(sigma_integral(sigma_vals**3, levels))
            errors.append(abs(result - 0.25))
        # Error should decrease with resolution
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"Error should decrease: L{[5, 10, 20, 40][i]}={errors[i]:.2e} "
                f">= L{[5, 10, 20, 40][i + 1]}={errors[i + 1]:.2e}"
            )


# =====================================================================
# Conservation diagnostics on the J-W state
# =====================================================================


class TestConservationDiagnostics:
    """Verify conservation properties of the dynamical core."""

    def test_diagnostics_have_physical_values(self) -> None:
        """Mass, energy, and angular momentum should be in reasonable ranges."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(20)
        state, _, surface_phi = jablonowski_williamson_steady_state(
            transform, EARTH, levels
        )

        diag = compute_conservation_diagnostics(
            state, transform, EARTH, levels, surface_phi
        )

        # Total mass of Earth's atmosphere ≈ 5.15e18 kg
        assert 4e18 < diag.mass < 6e18, (
            f"Mass {diag.mass:.3e} kg not in expected range"
        )

        # Total energy should be dominated by internal energy (cp*T*mass)
        # Rough: cp * 250K * 5e18 kg ≈ 1.3e27 J
        assert diag.total_energy > 0
        assert diag.internal_energy > diag.kinetic_energy

        # Angular momentum should be positive and substantial
        assert diag.angular_momentum > 1e27

    def test_steady_state_mass_conserved(self) -> None:
        """Mass should be conserved to machine precision for the steady state."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(20)
        dt = 600.0
        n_steps = 36  # 6 hours

        state, ref_temps, surface_phi = jablonowski_williamson_steady_state(
            transform, EARTH, levels
        )
        filt = exponential_filter(grid.truncation, dt)
        init_fn, step_fn = build_pe_stepper(
            transform=transform, planet=EARTH, levels=levels,
            reference_temperature=ref_temps,
            surface_geopotential=surface_phi,
            dt=dt, spectral_filter=filt,
        )

        diag0 = compute_conservation_diagnostics(
            state, transform, EARTH, levels, surface_phi
        )

        prev, curr = init_fn(state)
        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        diag1 = compute_conservation_diagnostics(
            curr, transform, EARTH, levels, surface_phi
        )

        np.testing.assert_allclose(
            diag1.mass, diag0.mass, rtol=1e-8,
            err_msg="Mass should be conserved for steady state"
        )

    def test_perturbed_state_approximate_conservation(self) -> None:
        """Energy and angular momentum should be approximately conserved.

        For the adiabatic, frictionless dynamics the conservation is
        not exact (spectral truncation, time discretization, filters)
        but should hold to within a few percent over 1 day.
        """
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        levels = uniform_sigma_levels(20)
        dt = 600.0
        n_steps = 144  # 1 day

        state, ref_temps, surface_phi = jablonowski_williamson_steady_state(
            transform, EARTH, levels
        )
        pert = jablonowski_williamson_perturbation(transform, EARTH, levels)
        perturbed = jax.tree.map(jnp.add, state, pert)

        filt = exponential_filter(grid.truncation, dt)
        init_fn, step_fn = build_pe_stepper(
            transform=transform, planet=EARTH, levels=levels,
            reference_temperature=ref_temps,
            surface_geopotential=surface_phi,
            dt=dt, spectral_filter=filt,
        )

        diag0 = compute_conservation_diagnostics(
            perturbed, transform, EARTH, levels, surface_phi
        )

        prev, curr = init_fn(perturbed)
        for _ in range(n_steps):
            prev, curr = step_fn(prev, curr)

        diag1 = compute_conservation_diagnostics(
            curr, transform, EARTH, levels, surface_phi
        )

        # Mass: should be conserved to ~1e-6 or better
        np.testing.assert_allclose(
            diag1.mass, diag0.mass, rtol=1e-5,
            err_msg="Mass should be well conserved over 1 day"
        )

        # Total energy: conserved to within ~1%
        np.testing.assert_allclose(
            diag1.total_energy, diag0.total_energy, rtol=0.01,
            err_msg="Total energy should be approximately conserved over 1 day"
        )

        # Angular momentum: conserved to within ~1%
        np.testing.assert_allclose(
            diag1.angular_momentum, diag0.angular_momentum, rtol=0.01,
            err_msg="Angular momentum should be approximately conserved over 1 day"
        )
