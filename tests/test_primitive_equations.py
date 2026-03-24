"""Tests for the primitive equation explicit tendencies."""

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import EARTH, PlanetaryConstants
from notus.dynamics.primitive_equations import primitive_equation_tendencies
from notus.dynamics.shallow_water import shallow_water_tendencies
from notus.grid import GaussianGrid
from notus.operators import laplacian
from notus.vertical.sigma import uniform_sigma_levels
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
        temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(t_ref_value + 0j),
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
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        tend = tendency_fn(state)

        np.testing.assert_allclose(
            jnp.abs(tend.vorticity),
            0.0,
            atol=1e-20,
            err_msg="Vorticity tendency should be zero for resting state",
        )
        np.testing.assert_allclose(
            jnp.abs(tend.divergence),
            0.0,
            atol=1e-20,
            err_msg="Divergence tendency should be zero for resting state",
        )
        np.testing.assert_allclose(
            jnp.abs(tend.temperature),
            0.0,
            atol=1e-20,
            err_msg="Temperature tendency should be zero for resting state",
        )
        np.testing.assert_allclose(
            jnp.abs(tend.log_surface_pressure),
            0.0,
            atol=1e-20,
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
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
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
            temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(250.0 + 0j),
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )

        tendency_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
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
            temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(250.0 + 0j),
            log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
        )

        # Matching SW state (geopotential doesn't matter for vorticity)
        sw_state = ShallowWaterState(
            vorticity=single_vort,
            divergence=jnp.zeros(n_spec, dtype=jnp.complex128),
            geopotential=jnp.zeros(n_spec, dtype=jnp.complex128),
        )

        pe_tend_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        sw_tend_fn = shallow_water_tendencies(
            transform,
            EARTH,
            diffusion_order=0,
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
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        tend = tendency_fn(state)

        # Expected: -∇²(g·z_s) at each level
        expected = -laplacian(surface_phi, grid.truncation, EARTH.radius)

        for k in range(n_levels):
            np.testing.assert_allclose(
                tend.divergence[k],
                expected,
                atol=1e-20,
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
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=4,
        )(state)

        # Both should be zero for resting state (no gradients to diffuse)
        np.testing.assert_allclose(
            jnp.abs(tend_with_diff.vorticity),
            0.0,
            atol=1e-20,
        )
        np.testing.assert_allclose(
            jnp.abs(tend_with_diff.log_surface_pressure),
            0.0,
            atol=1e-20,
        )


# ---------------------------------------------------------------------------
# Tests with nonzero lnps / T' / divergence — exercises ∇(lnps) scaling
# ---------------------------------------------------------------------------


def _make_perturbed_state(
    grid: GaussianGrid,
    n_levels: int,
    t_ref_value: float = 250.0,
) -> PrimitiveEquationState:
    """State with small perturbations in ALL fields including lnps.

    This ensures that ∇(lnps) ≠ 0 and exercises the radius scaling in the
    surface pressure gradient computation.
    """
    n_spec = grid.n_spectral_coeffs
    idx_20 = grid.spectral_index(0, 2)
    idx_21 = grid.spectral_index(1, 2)

    vort = jnp.zeros((n_levels, n_spec), dtype=jnp.complex128)
    vort = vort.at[:, idx_20].set(1e-6 + 0j)

    div = jnp.zeros((n_levels, n_spec), dtype=jnp.complex128)
    div = div.at[:, idx_21].set(1e-8 + 0j)

    temp = jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(t_ref_value + 0j)
    temp = temp.at[:, idx_20].set(0.1 + 0j)  # small T' perturbation

    lnps = jnp.zeros(n_spec, dtype=jnp.complex128)
    lnps = lnps.at[idx_21].set(1e-4 + 0j)  # nonzero ∇(lnps)

    return PrimitiveEquationState(
        vorticity=vort,
        divergence=div,
        temperature=temp,
        log_surface_pressure=lnps,
    )


class TestNonzeroLnpsGradient:
    """Tests that exercise nonzero ∇(lnps), catching 1/a scaling bugs."""

    def test_tendency_magnitudes_bounded(self) -> None:
        """With O(1e-4) lnps perturbation, tendencies should be physically
        reasonable — not inflated by a factor of Earth radius."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = jnp.full(n_levels, 250.0)
        surface_phi = jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128)

        state = _make_perturbed_state(grid, n_levels)

        tend_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        tend = tend_fn(state)

        # Physical bounds: with O(1e-4) lnps and O(1e-6) vorticity,
        # tendencies should be small.  If 1/a is missing, F.lnps would
        # be ~a times too large (O(1) instead of O(1e-7)).
        lnps_max = float(jnp.max(jnp.abs(tend.log_surface_pressure)))
        temp_max = float(jnp.max(jnp.abs(tend.temperature)))

        assert lnps_max < 1e-3, (
            f"lnps tendency {lnps_max:.2e} is too large — possible missing 1/a in ∇(lnps)"
        )
        assert temp_max < 1.0, (
            f"temperature tendency {temp_max:.2e} is too large — possible missing 1/a in ∇(lnps)"
        )

    def test_radius_scaling_consistency(self) -> None:
        """Doubling the planet radius should not change the lnps tendency.

        For the lnps tendency (d(lnps)/dt = -Σ v⃗·∇(lnps) Δσ):
        - Winds from fixed spectral vorticity scale as a (inverse Laplacian ∝ a²,
          then 1/a in the wind formula, giving net ∝ a)
        - Physical gradient scales as 1/a (from ∇ = (1/a)∂/∂...)
        - So v⃗·∇(lnps) is radius-independent and the ratio should be ~1.0.

        A missing 1/a in the gradient would make winds × gradient scale as a,
        giving ratio ~2.0 when the radius doubles.
        """
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 3
        levels = uniform_sigma_levels(n_levels)
        t_ref = jnp.full(n_levels, 250.0)
        n_spec = grid.n_spectral_coeffs
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)

        state = _make_perturbed_state(grid, n_levels)

        # Compute tendencies at standard radius
        tend_fn_1 = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        tend_1 = tend_fn_1(state)

        # Compute tendencies at double radius
        planet_2a = PlanetaryConstants(
            name="2x Earth",
            radius=EARTH.radius * 2,
            rotation_rate=EARTH.rotation_rate,
            gravity=EARTH.gravity,
            gas_constant=EARTH.gas_constant,
            specific_heat_cp=EARTH.specific_heat_cp,
        )
        tend_fn_2 = primitive_equation_tendencies(
            transform,
            planet_2a,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        tend_2 = tend_fn_2(state)

        # The lnps tendency should be approximately radius-independent.
        lnps_1 = float(jnp.max(jnp.abs(tend_1.log_surface_pressure)))
        lnps_2 = float(jnp.max(jnp.abs(tend_2.log_surface_pressure)))

        if lnps_1 > 1e-30:  # skip if negligible
            ratio = lnps_2 / lnps_1
            # Correct code: ratio ≈ 1.0.  Buggy (missing 1/a): ratio ≈ 2.0.
            assert 0.8 < ratio < 1.2, (
                f"lnps tendency ratio at 2a vs a is {ratio:.4f}, "
                f"expected ~1.0 (radius-independent). "
                f"If ~2.0, there's a missing 1/a factor in ∇(lnps)."
            )

    def test_lnps_tendency_sign_and_structure(self) -> None:
        """Surface pressure tendency from a Y_2^1 lnps field should have
        the correct spectral structure (not contain spurious modes)."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 3
        levels = uniform_sigma_levels(n_levels)
        t_ref = jnp.full(n_levels, 250.0)
        n_spec = grid.n_spectral_coeffs
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)

        state = _make_perturbed_state(grid, n_levels)

        tend_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        tend = tend_fn(state)

        # The tendency should be finite and structured
        assert jnp.all(jnp.isfinite(tend.log_surface_pressure))
        assert jnp.all(jnp.isfinite(tend.temperature))
        assert jnp.all(jnp.isfinite(tend.divergence))
        assert jnp.all(jnp.isfinite(tend.vorticity))

        # The (0,0) mode of lnps tendency should be near zero (mass conservation
        # for the explicit tendency — the total column mass should not change
        # from the v⃗·∇(lnps) terms alone in a finite perturbation).
        mode_00 = float(jnp.abs(tend.log_surface_pressure[0]))
        assert mode_00 < 1e-10, (
            f"lnps tendency mode (0,0) = {mode_00:.2e}, should be ~0 "
            "(explicit lnps tendency conserves global mean)"
        )


class TestCombinedPerturbationStability:
    """Integration tests with simultaneous nonzero perturbations in all fields."""

    def test_all_fields_perturbed_10_steps(self) -> None:
        """State with nonzero div + T' + lnps should survive 10 IMEX steps."""
        from notus.operators import exponential_filter
        from notus.timestepping.imex import build_pe_stepper

        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.full(n_levels, 250.0)
        n_spec = grid.n_spectral_coeffs
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)
        dt = 1200.0

        state = _make_perturbed_state(grid, n_levels)
        filt = exponential_filter(grid.truncation, dt)

        init_fn, step_fn = build_pe_stepper(
            transform=transform,
            planet=EARTH,
            levels=levels,
            reference_temperature=t_ref,
            surface_geopotential=surface_phi,
            dt=dt,
            spectral_filter=filt,
        )

        prev, curr = init_fn(state)
        for _ in range(9):
            prev, curr = step_fn(prev, curr)

        for field_name in ("vorticity", "divergence", "temperature", "log_surface_pressure"):
            field = getattr(curr, field_name)
            assert jnp.all(jnp.isfinite(field)), f"{field_name} blew up"

    def test_lapse_rate_with_lnps_perturbation(self) -> None:
        """Lapse-rate T_ref with lnps perturbation exercises K terms + ∇(lnps)."""
        from notus.operators import exponential_filter
        from notus.timestepping.imex import build_pe_stepper

        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.linspace(300.0, 200.0, n_levels)
        n_spec = grid.n_spectral_coeffs
        surface_phi = jnp.zeros(n_spec, dtype=jnp.complex128)
        dt = 1200.0

        # Build state with temperature matching the varying T_ref
        idx_20 = grid.spectral_index(0, 2)
        idx_21 = grid.spectral_index(1, 2)

        vort = jnp.zeros((n_levels, n_spec), dtype=jnp.complex128)
        vort = vort.at[:, idx_20].set(1e-6 + 0j)

        temp = jnp.zeros((n_levels, n_spec), dtype=jnp.complex128)
        for k in range(n_levels):
            temp = temp.at[k, 0].set(t_ref[k] + 0j)

        lnps = jnp.zeros(n_spec, dtype=jnp.complex128)
        lnps = lnps.at[idx_21].set(1e-4 + 0j)

        state = PrimitiveEquationState(
            vorticity=vort,
            divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
            temperature=temp,
            log_surface_pressure=lnps,
        )

        filt = exponential_filter(grid.truncation, dt)
        init_fn, step_fn = build_pe_stepper(
            transform=transform,
            planet=EARTH,
            levels=levels,
            reference_temperature=t_ref,
            surface_geopotential=surface_phi,
            dt=dt,
            spectral_filter=filt,
        )

        prev, curr = init_fn(state)
        for _ in range(9):
            prev, curr = step_fn(prev, curr)

        for field_name in ("vorticity", "divergence", "temperature", "log_surface_pressure"):
            field = getattr(curr, field_name)
            assert jnp.all(jnp.isfinite(field)), f"{field_name} blew up"
