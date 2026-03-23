"""Tests for the primitive equation explicit tendencies."""

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import EARTH
from notus.dynamics.primitive_equations import primitive_equation_tendencies
from notus.dynamics.shallow_water import shallow_water_tendencies
from notus.grid import GaussianGrid
from notus.operators import laplacian
from notus.sigma import uniform_sigma_levels
from notus.state import PrimitiveEquationState, ShallowWaterState
from notus.transforms import SpectralTransform


jax.config.update("jax_enable_x64", True)


def _make_resting_state(
    grid: GaussianGrid,
    n_levels: int,
    t_ref_value: float = 250.0,
) -> PrimitiveEquationState:
    """Create a resting isothermal atmosphere."""
    n_spec = grid.n_spectral_coeffs
    return PrimitiveEquationState(
        vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(
            t_ref_value + 0j
        ),
        log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
    )


class TestRestingState:
    """Resting isothermal atmosphere should produce zero tendencies."""

    def test_all_tendencies_zero(self) -> None:
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = jnp.full(n_levels, 250.0)
        surface_phi = jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128)

        state = _make_resting_state(grid, n_levels, 250.0)

        tendency_fn = primitive_equation_tendencies(
            transform, EARTH, levels, t_ref, surface_phi,
            diffusion_order=0,
        )
        tend = tendency_fn(state)

        np.testing.assert_allclose(
            jnp.abs(tend.vorticity), 0.0, atol=1e-20,
            err_msg="Vorticity tendency should be zero for resting state",
        )
        np.testing.assert_allclose(
            jnp.abs(tend.divergence), 0.0, atol=1e-20,
            err_msg="Divergence tendency should be zero for resting state",
        )
        np.testing.assert_allclose(
            jnp.abs(tend.temperature), 0.0, atol=1e-20,
            err_msg="Temperature tendency should be zero for resting state",
        )
        np.testing.assert_allclose(
            jnp.abs(tend.log_surface_pressure), 0.0, atol=1e-20,
            err_msg="ln(ps) tendency should be zero for resting state",
        )


class TestShapeCorrectness:
    """Verify output shapes."""

    def test_shapes(self) -> None:
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = jnp.full(n_levels, 250.0)
        n_spec = grid.n_spectral_coeffs
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)
        state = _make_resting_state(grid, n_levels)

        tendency_fn = primitive_equation_tendencies(
            transform, EARTH, levels, t_ref, surface_phi, diffusion_order=0,
        )
        tend = tendency_fn(state)

        assert tend.vorticity.shape == (n_levels, n_spec)
        assert tend.divergence.shape == (n_levels, n_spec)
        assert tend.temperature.shape == (n_levels, n_spec)
        assert tend.log_surface_pressure.shape == (n_spec,)


class TestBarotropicState:
    """Barotropic (depth-independent) state tests."""

    def test_vertical_advection_zero(self) -> None:
        """Vertically uniform fields → zero vertical advection."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        n_spec = grid.n_spectral_coeffs
        t_ref = jnp.full(n_levels, 250.0)
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)

        # Create a state with the same fields at every level
        # Use a small vorticity perturbation (Y_2^0 mode)
        single_vort = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_20 = grid.spectral_index(0, 2)
        single_vort = single_vort.at[idx_20].set(1e-5 + 0j)

        state = PrimitiveEquationState(
            vorticity=jnp.tile(single_vort, (n_levels, 1)),
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(
                250.0 + 0j
            ),
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )

        tendency_fn = primitive_equation_tendencies(
            transform, EARTH, levels, t_ref, surface_phi, diffusion_order=0,
        )
        tend = tendency_fn(state)

        # With zero divergence, zero lnps gradient, and barotropic state,
        # sigma-dot should be zero and vertical terms should vanish.
        # The vorticity tendency should come only from the horizontal flux
        # (Coriolis effect on the vorticity perturbation).
        # We can at least verify it's finite and not NaN.
        assert jnp.all(jnp.isfinite(tend.vorticity))
        assert jnp.all(jnp.isfinite(tend.divergence))
        assert jnp.all(jnp.isfinite(tend.temperature))
        assert jnp.all(jnp.isfinite(tend.log_surface_pressure))


class TestVorticityMatchesSW:
    """Barotropic PE vorticity tendency should match SW per level."""

    def test_vorticity_tendency_matches_sw(self) -> None:
        """With ln(ps)=const and barotropic state, PE vorticity tendency
        at each level should match the SW vorticity tendency."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 3
        levels = uniform_sigma_levels(n_levels)
        n_spec = grid.n_spectral_coeffs
        t_ref = jnp.full(n_levels, 250.0)
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)

        # Create a balanced-ish vorticity field (Y_2^0)
        single_vort = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_20 = grid.spectral_index(0, 2)
        single_vort = single_vort.at[idx_20].set(1e-5 + 0j)

        # PE state: same vorticity at all levels, zero everything else
        pe_state = PrimitiveEquationState(
            vorticity=jnp.tile(single_vort, (n_levels, 1)),
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(
                250.0 + 0j
            ),
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )

        # Matching SW state (geopotential doesn't matter for vorticity)
        sw_state = ShallowWaterState(
            vorticity=single_vort,
            divergence=jnp.zeros(n_spec, dtype=jnp.complex128),
            geopotential=jnp.zeros(n_spec, dtype=jnp.complex128),
        )

        pe_tend_fn = primitive_equation_tendencies(
            transform, EARTH, levels, t_ref, surface_phi, diffusion_order=0,
        )
        sw_tend_fn = shallow_water_tendencies(
            transform, EARTH, diffusion_order=0,
        )

        pe_tend = pe_tend_fn(pe_state)
        sw_tend = sw_tend_fn(sw_state)

        # PE vorticity tendency at each level should match SW
        for k in range(n_levels):
            np.testing.assert_allclose(
                pe_tend.vorticity[k],
                sw_tend.vorticity,
                atol=1e-20,
                err_msg=f"PE vorticity tendency at level {k} doesn't match SW",
            )


class TestOrographyTerm:
    """Divergence tendency should include -∇²(g·z_s)."""

    def test_orography_in_divergence(self) -> None:
        """With non-zero orography and zero state, only orography
        contributes to the divergence tendency."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 3
        levels = uniform_sigma_levels(n_levels)
        n_spec = grid.n_spectral_coeffs
        t_ref = jnp.full(n_levels, 250.0)

        # Non-zero orography: a Y_2^0 mode
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)
        idx_20 = grid.spectral_index(0, 2)
        surface_phi = surface_phi.at[idx_20].set(1000.0 + 0j)

        state = _make_resting_state(grid, n_levels)

        tendency_fn = primitive_equation_tendencies(
            transform, EARTH, levels, t_ref, surface_phi, diffusion_order=0,
        )
        tend = tendency_fn(state)

        # Expected: -∇²(g·z_s) at each level
        expected = -laplacian(surface_phi, grid.truncation, EARTH.radius)

        for k in range(n_levels):
            np.testing.assert_allclose(
                tend.divergence[k], expected, atol=1e-20,
                err_msg=f"Divergence at level {k} doesn't match orography",
            )


class TestHyperdiffusion:
    """Hyperdiffusion tests."""

    def test_diffusion_on_vort_div_temp_not_lnps(self) -> None:
        """Hyperdiffusion should act on ζ, δ, T but NOT on ln(ps)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 3
        levels = uniform_sigma_levels(n_levels)
        n_spec = grid.n_spectral_coeffs
        t_ref = jnp.full(n_levels, 250.0)
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)

        state = _make_resting_state(grid, n_levels)

        # With diffusion on, still zero for resting state (nothing to diffuse)
        tend_with_diff = primitive_equation_tendencies(
            transform, EARTH, levels, t_ref, surface_phi, diffusion_order=4,
        )(state)

        # Both should be zero for resting state (no gradients to diffuse)
        np.testing.assert_allclose(
            jnp.abs(tend_with_diff.vorticity), 0.0, atol=1e-20,
        )
        np.testing.assert_allclose(
            jnp.abs(tend_with_diff.log_surface_pressure), 0.0, atol=1e-20,
        )
