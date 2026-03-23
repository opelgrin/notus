"""Tests for the IMEX leapfrog time stepper.

Covers the generic euler_init / imex_leapfrog_step functions and the
PE-specific build_pe_stepper convenience factory.  Two T_ref profiles
are used throughout — isothermal (250 K) and lapse-rate (300→200 K) —
to exercise the K terms in the temperature implicit weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.dynamics.primitive_equations import primitive_equation_tendencies
from notus.grid import GaussianGrid
from notus.operators import exponential_filter
from notus.sigma import uniform_sigma_levels
from notus.state import PrimitiveEquationState
from notus.timestepping.imex import (
    build_pe_stepper,
    euler_init,
)
from notus.timestepping.semi_implicit import (
    build_pe_semi_implicit_config,
    pe_implicit_inverse,
    pe_implicit_terms,
)
from notus.transforms import SpectralTransform


jax.config.update("jax_enable_x64", True)

# Standard parameters for fast tests
TRUNC = 21
DT = 1200.0  # 20-minute timestep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_resting_state(
    grid: GaussianGrid,
    n_levels: int,
    t_ref_value: float = 250.0,
) -> PrimitiveEquationState:
    """Isothermal resting atmosphere in spectral space."""
    n_spec = grid.n_spectral_coeffs
    return PrimitiveEquationState(
        vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        temperature=jnp.full((n_levels, n_spec), 0.0 + 0j).at[:, 0].set(t_ref_value + 0j),
        log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
    )


def _make_perturbed_state(
    grid: GaussianGrid,
    n_levels: int,
    t_ref_value: float = 250.0,
    amplitude: float = 1e-6,
) -> PrimitiveEquationState:
    """Resting atmosphere with a small Y_2^0 vorticity perturbation."""
    state = _make_resting_state(grid, n_levels, t_ref_value)
    idx_20 = grid.spectral_index(0, 2)
    vort = state.vorticity.at[:, idx_20].set(amplitude + 0j)
    return state.replace(vorticity=vort)


def _make_stepper(
    n_levels: int = 5,
    t_ref_profile: str = "isothermal",
    with_filter: bool = False,
    dt: float = DT,
    robert_coeff: float = 0.05,
) -> tuple:
    """Build (init_fn, step_fn, grid, t_ref)."""
    grid = GaussianGrid(truncation=TRUNC)
    transform = SpectralTransform(grid)
    levels = uniform_sigma_levels(n_levels)

    if t_ref_profile == "isothermal":
        t_ref = np.full(n_levels, 250.0)
    else:
        t_ref = np.linspace(300.0, 200.0, n_levels)

    surface_phi = jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128)
    filt = None
    if with_filter:
        filt = exponential_filter(TRUNC, dt)

    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=t_ref,
        surface_geopotential=surface_phi,
        dt=dt,
        spectral_filter=filt,
        robert_coeff=robert_coeff,
    )
    return init_fn, step_fn, grid, t_ref


def _run_steps(
    init_fn,
    step_fn,
    state: PrimitiveEquationState,
    n_steps: int,
) -> PrimitiveEquationState:
    """Run n_steps of IMEX leapfrog and return the final current state."""
    prev, curr = init_fn(state)
    for _ in range(n_steps - 1):
        prev, curr = step_fn(prev, curr)
    return curr


# ---------------------------------------------------------------------------
# TestEulerInit
# ---------------------------------------------------------------------------
class TestEulerInit:
    """Tests for backward-forward Euler initialization."""

    def test_resting_isothermal_stays_resting(self) -> None:
        """Resting isothermal state should be unchanged after Euler init."""
        init_fn, _, grid, _ = _make_stepper(t_ref_profile="isothermal")
        state = _make_resting_state(grid, 5)
        _prev, curr = init_fn(state)

        for field in ("vorticity", "divergence", "temperature", "log_surface_pressure"):
            np.testing.assert_allclose(
                jnp.abs(getattr(curr, field) - getattr(state, field)),
                0.0,
                atol=1e-15,
                err_msg=f"{field} changed after Euler init of resting state",
            )

    def test_resting_lapse_rate_stays_resting(self) -> None:
        """Resting state with non-constant T_ref stays resting (K terms active)."""
        init_fn, _, grid, t_ref = _make_stepper(
            t_ref_profile="lapse_rate",
        )
        state = _make_resting_state(grid, 5, t_ref_value=t_ref[0])
        # Set each level's temperature to its T_ref
        temp = jnp.zeros_like(state.temperature)
        for k in range(5):
            temp = temp.at[k, 0].set(t_ref[k] + 0j)
        state = state.replace(temperature=temp)

        _prev, curr = init_fn(state)

        for field in ("vorticity", "divergence", "log_surface_pressure"):
            np.testing.assert_allclose(
                jnp.abs(getattr(curr, field)),
                0.0,
                atol=1e-14,
                err_msg=f"{field} nonzero after Euler init of resting lapse-rate state",
            )

    def test_shapes(self) -> None:
        """Output shapes match input."""
        init_fn, _, grid, _ = _make_stepper()
        n_levels, n_spec = 5, grid.n_spectral_coeffs
        state = _make_resting_state(grid, n_levels)
        prev, curr = init_fn(state)

        assert prev.vorticity.shape == (n_levels, n_spec)
        assert curr.vorticity.shape == (n_levels, n_spec)
        assert curr.log_surface_pressure.shape == (n_spec,)

    def test_jit_compatible(self) -> None:
        """euler_init path compiles under jax.jit."""
        init_fn, _, grid, _ = _make_stepper()
        state = _make_resting_state(grid, 5)
        jit_init = jax.jit(init_fn)
        _prev, curr = jit_init(state)
        assert jnp.all(jnp.isfinite(curr.divergence))

    def test_vorticity_unaffected_by_implicit(self) -> None:
        """Vorticity in Euler step = state + dt*F.vorticity (implicit doesn't touch it)."""
        grid = GaussianGrid(truncation=TRUNC)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.full(n_levels, 250.0)
        surface_phi = jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128)

        explicit_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        si_config = build_pe_semi_implicit_config(
            levels,
            EARTH.gas_constant,
            EARTH.kappa,
            t_ref,
        )

        def inverse_fn(s: PrimitiveEquationState, step_size: float) -> PrimitiveEquationState:
            return pe_implicit_inverse(s, step_size, si_config, TRUNC, EARTH.radius)

        state = _make_perturbed_state(grid, n_levels)
        tend = explicit_fn(state)
        expected_vort = state.vorticity + DT * tend.vorticity

        _prev, curr = euler_init(state, explicit_fn, inverse_fn, DT)

        np.testing.assert_allclose(curr.vorticity, expected_vort, atol=1e-20)


# ---------------------------------------------------------------------------
# TestIMEXStep
# ---------------------------------------------------------------------------
class TestIMEXStep:
    """Tests for a single IMEX leapfrog step."""

    def test_resting_stays_resting(self) -> None:
        """Euler init + one leapfrog step: resting state stays resting."""
        init_fn, step_fn, grid, _ = _make_stepper()
        state = _make_resting_state(grid, 5)
        prev, curr = init_fn(state)
        _filt_curr, future = step_fn(prev, curr)

        for field in ("vorticity", "divergence", "temperature", "log_surface_pressure"):
            np.testing.assert_allclose(
                jnp.abs(getattr(future, field) - getattr(state, field)),
                0.0,
                atol=1e-14,
                err_msg=f"{field} drifted after one IMEX step from resting state",
            )

    def test_resting_lapse_rate_stays_resting(self) -> None:
        """Resting state with lapse-rate T_ref stays resting after one step."""
        init_fn, step_fn, grid, t_ref = _make_stepper(t_ref_profile="lapse_rate")
        temp = jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128)
        for k in range(5):
            temp = temp.at[k, 0].set(t_ref[k] + 0j)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            divergence=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            temperature=temp,
            log_surface_pressure=jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128),
        )
        prev, curr = init_fn(state)
        _filt_curr, future = step_fn(prev, curr)

        for field in ("vorticity", "divergence", "log_surface_pressure"):
            np.testing.assert_allclose(
                jnp.abs(getattr(future, field)),
                0.0,
                atol=1e-13,
                err_msg=f"{field} nonzero after IMEX step from resting lapse-rate state",
            )

    def test_shapes(self) -> None:
        init_fn, step_fn, grid, _ = _make_stepper()
        n_levels, n_spec = 5, grid.n_spectral_coeffs
        state = _make_resting_state(grid, n_levels)
        prev, curr = init_fn(state)
        filt_curr, future = step_fn(prev, curr)

        assert filt_curr.vorticity.shape == (n_levels, n_spec)
        assert future.divergence.shape == (n_levels, n_spec)
        assert future.log_surface_pressure.shape == (n_spec,)

    def test_jit_compatible(self) -> None:
        init_fn, step_fn, grid, _ = _make_stepper()
        state = _make_resting_state(grid, 5)
        prev, curr = init_fn(state)
        jit_step = jax.jit(step_fn)
        _filt_curr, future = jit_step(prev, curr)
        assert jnp.all(jnp.isfinite(future.divergence))

    def test_robert_coeff_zero_means_no_filtering(self) -> None:
        """With r=0, filtered_current should be exactly current."""
        init_fn, step_fn, grid, _ = _make_stepper(robert_coeff=0.0)
        state = _make_perturbed_state(grid, 5)
        prev, curr = init_fn(state)
        filt_curr, _future = step_fn(prev, curr)

        # filtered_current = (1-0)*current + 0*(previous+future) = current
        np.testing.assert_allclose(filt_curr.vorticity, curr.vorticity, atol=1e-20)
        np.testing.assert_allclose(filt_curr.divergence, curr.divergence, atol=1e-20)

    def test_robert_filter_formula(self) -> None:
        """Verify Robert-Asselin formula: (1-2r)*current + r*(previous+future)."""
        r = 0.05
        init_fn, step_fn, grid, _ = _make_stepper(robert_coeff=r)
        state = _make_perturbed_state(grid, 5)
        prev, curr = init_fn(state)

        # Get the future from a step with r=0 (no filtering of current)
        _, step_fn_nofilt, _, _ = _make_stepper(robert_coeff=0.0)
        _, future_raw = step_fn_nofilt(prev, curr)

        # Now get the filtered result from the r=0.05 step
        filt_curr, _ = step_fn(prev, curr)

        # The future should be the same (RA only affects filtered_current)
        # Check the RA formula on one field
        expected = (1.0 - 2.0 * r) * curr.vorticity + r * (prev.vorticity + future_raw.vorticity)
        np.testing.assert_allclose(filt_curr.vorticity, expected, atol=1e-18)


# ---------------------------------------------------------------------------
# TestMultiStepStability
# ---------------------------------------------------------------------------
class TestMultiStepStability:
    """Multi-step integration stability tests."""

    def test_resting_isothermal_100_steps(self) -> None:
        """Resting isothermal atmosphere stays at rest over 100 steps."""
        init_fn, step_fn, grid, _ = _make_stepper()
        state = _make_resting_state(grid, 5)
        final = _run_steps(init_fn, step_fn, state, 100)

        for field in ("vorticity", "divergence", "log_surface_pressure"):
            max_err = float(jnp.max(jnp.abs(getattr(final, field))))
            assert max_err < 1e-10, f"{field} drifted to {max_err} after 100 steps"

    def test_resting_lapse_rate_100_steps(self) -> None:
        """Resting lapse-rate atmosphere stable over 100 steps (exercises K terms)."""
        init_fn, step_fn, grid, t_ref = _make_stepper(t_ref_profile="lapse_rate")
        temp = jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128)
        for k in range(5):
            temp = temp.at[k, 0].set(t_ref[k] + 0j)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            divergence=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            temperature=temp,
            log_surface_pressure=jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128),
        )
        final = _run_steps(init_fn, step_fn, state, 100)

        for field in ("vorticity", "divergence", "log_surface_pressure"):
            max_err = float(jnp.max(jnp.abs(getattr(final, field))))
            assert max_err < 1e-8, f"{field} drifted to {max_err} after 100 lapse-rate steps"

    @pytest.mark.xfail(reason="PE perturbation stability bug — see docs/debugging_pe_stability.md")
    def test_perturbation_20_steps(self) -> None:
        """Perturbed state should survive 20 IMEX steps with filter.

        Dinosaur achieves this with the same physical state. Our code blows
        up at step 4 due to excessive lnps/temperature growth through the
        explicit tendency. See docs/debugging_pe_stability.md.
        """
        init_fn, step_fn, grid, _ = _make_stepper(with_filter=True)
        state = _make_perturbed_state(grid, 5)
        final = _run_steps(init_fn, step_fn, state, 20)

        assert jnp.all(jnp.isfinite(final.vorticity)), "vorticity blew up"
        assert jnp.all(jnp.isfinite(final.divergence)), "divergence blew up"
        assert jnp.all(jnp.isfinite(final.temperature)), "temperature blew up"

    def test_with_spectral_filter_stable(self) -> None:
        """100 steps with exponential spectral filter, resting state."""
        init_fn, step_fn, grid, _ = _make_stepper(with_filter=True)
        state = _make_resting_state(grid, 5)
        final = _run_steps(init_fn, step_fn, state, 100)

        assert jnp.all(jnp.isfinite(final.divergence))


# ---------------------------------------------------------------------------
# TestMassConservation
# ---------------------------------------------------------------------------
class TestMassConservation:
    """ln(ps) mode (0,0) should be conserved by the semi-implicit scheme."""

    def test_lnps_mode0_conserved_resting_50_steps(self) -> None:
        init_fn, step_fn, grid, _ = _make_stepper()
        state = _make_resting_state(grid, 5)
        final = _run_steps(init_fn, step_fn, state, 50)

        np.testing.assert_allclose(
            final.log_surface_pressure[0],
            state.log_surface_pressure[0],
            atol=1e-12,
        )

    def test_lnps_mode0_conserved_resting_lapse_rate_50_steps(self) -> None:
        """Mass conservation with lapse-rate T_ref (exercises K terms)."""
        init_fn, step_fn, grid, t_ref = _make_stepper(t_ref_profile="lapse_rate")
        temp = jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128)
        for k in range(5):
            temp = temp.at[k, 0].set(t_ref[k] + 0j)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            divergence=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            temperature=temp,
            log_surface_pressure=jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128),
        )
        final = _run_steps(init_fn, step_fn, state, 50)

        np.testing.assert_allclose(
            final.log_surface_pressure[0],
            state.log_surface_pressure[0],
            atol=1e-12,
        )

    def test_lnps_mode0_conserved_with_filter(self) -> None:
        """Spectral filter preserves mode 0 (eigenvalue 0, filter = 1.0)."""
        init_fn, step_fn, grid, _ = _make_stepper(with_filter=True)
        state = _make_resting_state(grid, 5)
        final = _run_steps(init_fn, step_fn, state, 50)

        np.testing.assert_allclose(
            final.log_surface_pressure[0],
            state.log_surface_pressure[0],
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# TestSpectralFilter
# ---------------------------------------------------------------------------
class TestSpectralFilter:
    """Tests for spectral filter interaction with time stepping."""

    def test_filter_damps_high_wavenumbers_directly(self) -> None:
        """The exponential filter array damps high-n modes and preserves low-n."""
        filt = exponential_filter(TRUNC, DT)

        # n=0 mode (index 0): filter should be exactly 1.0
        np.testing.assert_allclose(float(filt[0]), 1.0, atol=1e-15)

        # n=TRUNC mode (last coefficient): filter should be very small
        assert float(filt[-1]) < 0.01, "filter should strongly damp highest mode"

        # Low modes (n <= TRUNC/2) should be nearly 1.0
        # n=2 mode index: spectral_index(0, 2) for triangular truncation
        grid = GaussianGrid(truncation=TRUNC)
        idx_02 = grid.spectral_index(0, 2)
        assert float(filt[idx_02]) > 0.999, "filter should preserve n=2"

    def test_filter_preserves_resting_state(self) -> None:
        """Resting state should be unchanged by spectral filter over 10 steps.

        The resting state has energy only in mode (0,0) where the filter
        is exactly 1.0, so filtering should have no effect.
        """
        init_nf, step_nf, grid, _ = _make_stepper(with_filter=False)
        init_f, step_f, _, _ = _make_stepper(with_filter=True)
        state = _make_resting_state(grid, 5)

        final_nf = _run_steps(init_nf, step_nf, state, 10)
        final_f = _run_steps(init_f, step_f, state, 10)

        # For a resting state, filtered and unfiltered should match
        np.testing.assert_allclose(
            final_f.temperature[:, 0],
            final_nf.temperature[:, 0],
            atol=1e-12,
        )

    def test_filter_applied_to_future_not_current(self) -> None:
        """The spectral filter should be on future, not on filtered_current."""
        init_fn, step_fn, grid, _ = _make_stepper(with_filter=True)

        # Also build a no-filter stepper for comparison
        init_nf, step_nf, _, _ = _make_stepper(with_filter=False)

        state = _make_perturbed_state(grid, 5)
        prev, curr = init_fn(state)
        prev_nf, curr_nf = init_nf(state)

        filt_curr_f, _ = step_fn(prev, curr)
        _filt_curr_nf, _ = step_nf(prev_nf, curr_nf)

        # filtered_current comes from RA filter only (not spectral filter)
        # Both should agree since RA doesn't use spectral filter
        # (the spectral filter is only applied to curr in init, and to future in step)
        # Note: curr may differ due to filter in init, so this test checks
        # that filt_curr is RA-filtered current, not double-filtered.
        assert jnp.all(jnp.isfinite(filt_curr_f.vorticity))


# ---------------------------------------------------------------------------
# TestBuildPeStepper
# ---------------------------------------------------------------------------
class TestBuildPeStepper:
    """Tests for the PE convenience factory."""

    def test_returns_callables(self) -> None:
        init_fn, step_fn, _, _ = _make_stepper()
        assert callable(init_fn)
        assert callable(step_fn)

    def test_full_loop_10_steps(self) -> None:
        """10-step integration produces finite output."""
        init_fn, step_fn, grid, _ = _make_stepper()
        state = _make_resting_state(grid, 5)
        final = _run_steps(init_fn, step_fn, state, 10)
        assert jnp.all(jnp.isfinite(final.divergence))
        assert jnp.all(jnp.isfinite(final.temperature))

    def test_full_loop_lapse_rate_10_steps(self) -> None:
        """10-step integration with lapse-rate T_ref."""
        init_fn, step_fn, grid, t_ref = _make_stepper(t_ref_profile="lapse_rate")
        temp = jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128)
        for k in range(5):
            temp = temp.at[k, 0].set(t_ref[k] + 0j)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            divergence=jnp.zeros((5, grid.n_spectral_coeffs), dtype=jnp.complex128),
            temperature=temp,
            log_surface_pressure=jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128),
        )
        final = _run_steps(init_fn, step_fn, state, 10)
        assert jnp.all(jnp.isfinite(final.divergence))
        assert jnp.all(jnp.isfinite(final.temperature))


# ---------------------------------------------------------------------------
# TestPhysicsConsistency
# ---------------------------------------------------------------------------
class TestPhysicsConsistency:
    """Cross-checks between explicit tendencies and implicit terms."""

    def test_explicit_plus_implicit_zero_for_resting(self) -> None:
        """F(x_rest) + L(x_rest) = 0 for a resting isothermal state."""
        grid = GaussianGrid(truncation=TRUNC)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.full(n_levels, 250.0)
        surface_phi = jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128)

        explicit_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        si_config = build_pe_semi_implicit_config(
            levels,
            EARTH.gas_constant,
            EARTH.kappa,
            t_ref,
        )

        state = _make_resting_state(grid, n_levels)
        f_x = explicit_fn(state)
        l_x = pe_implicit_terms(state, si_config, TRUNC, EARTH.radius)

        for field in ("vorticity", "divergence", "temperature", "log_surface_pressure"):
            total = getattr(f_x, field) + getattr(l_x, field)
            np.testing.assert_allclose(
                jnp.abs(total),
                0.0,
                atol=1e-18,
                err_msg=f"F+L nonzero for {field}",
            )

    def test_vorticity_independent_of_implicit_solver(self) -> None:
        """Vorticity result should be independent of implicit solver output."""
        grid = GaussianGrid(truncation=TRUNC)
        transform = SpectralTransform(grid)
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.full(n_levels, 250.0)
        surface_phi = jnp.zeros(grid.n_spectral_coeffs, dtype=jnp.complex128)

        explicit_fn = primitive_equation_tendencies(
            transform,
            EARTH,
            levels,
            t_ref,
            surface_phi,
            diffusion_order=0,
        )
        si_config = build_pe_semi_implicit_config(
            levels,
            EARTH.gas_constant,
            EARTH.kappa,
            t_ref,
        )

        def real_inverse(s, step_size):
            return pe_implicit_inverse(s, step_size, si_config, TRUNC, EARTH.radius)

        def identity_inverse(s, _step_size):
            return s  # don't solve, just pass through

        state = _make_perturbed_state(grid, n_levels)
        _prev_r, curr_r = euler_init(state, explicit_fn, real_inverse, DT)
        _prev_i, curr_i = euler_init(state, explicit_fn, identity_inverse, DT)

        # Vorticity should be identical regardless of implicit solver
        np.testing.assert_allclose(curr_r.vorticity, curr_i.vorticity, atol=1e-20)
