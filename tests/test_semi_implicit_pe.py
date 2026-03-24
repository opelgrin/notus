"""Tests for the PE semi-implicit Helmholtz solver.

The 3D semi-implicit scheme couples divergence, temperature, and ln(ps)
through the vertical structure.  Unlike the shallow water 2×2 diagonal
solve, this produces an L×L system per spectral mode:

    (I - s²·λₙ·M) · δ_new = δ* - s·λₙ·(G·T* + R·T_ref·lnps*)
    T_new = T* - s·H·δ_new
    lnps_new = lnps* - s·Δσᵀ·δ_new

where M = G·H + R·T_ref⊗Δσ is the coupling matrix, H is the temperature
implicit weights, and λₙ = -n(n+1)/a² is the Laplacian eigenvalue.
"""

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import EARTH
from notus.sigma import uniform_sigma_levels
from notus.state import PrimitiveEquationState
from notus.timestepping.semi_implicit import (
    PESemiImplicitConfig,
    build_pe_semi_implicit_config,
    pe_coupling_matrix,
    pe_implicit_inverse,
    pe_implicit_terms,
    temperature_implicit_weights,
)
from notus.vertical import geopotential_weights, sigma_ratios


jax.config.update("jax_enable_x64", True)

R = 287.04
kappa = R / 1004.64

# Standard truncation and matching spectral size for implicit tests
TRUNC = 5
N_SPEC = (TRUNC + 1) * (TRUNC + 2) // 2  # 21


# ---------------------------------------------------------------------------
# temperature_implicit_weights (H matrix)
# ---------------------------------------------------------------------------
class TestTemperatureImplicitWeights:
    """Tests for the H matrix encoding adiabatic heating + T_ref vert advection."""

    def test_shape(self) -> None:
        levels = uniform_sigma_levels(5)
        h_mat = temperature_implicit_weights(levels, kappa, np.full(5, 250.0))
        assert h_mat.shape == (5, 5)

    def test_lower_triangular_for_constant_tref(self) -> None:
        """H is lower triangular when T_ref is constant (K terms vanish)."""
        levels = uniform_sigma_levels(8)
        h_mat = temperature_implicit_weights(levels, kappa, np.full(8, 250.0))
        for k in range(8):
            for j in range(k + 1, 8):
                np.testing.assert_equal(h_mat[k, j], 0.0, err_msg=f"H[{k},{j}] should be 0")

    def test_not_lower_triangular_for_varying_tref(self) -> None:
        """H has upper-triangular entries when T_ref varies (K terms nonzero)."""
        levels = uniform_sigma_levels(5)
        h_mat = temperature_implicit_weights(levels, kappa, np.linspace(200, 300, 5))
        upper_max = max(abs(h_mat[k, j]) for k in range(5) for j in range(k + 1, 5))
        assert upper_max > 0.1, "K terms should produce nonzero upper entries"

    def test_constant_tref_diagonal(self) -> None:
        """For constant T_ref (K=0): H_kk = κ·T_ref·α_k."""
        levels = uniform_sigma_levels(4)
        alpha = sigma_ratios(levels)
        t_val = 250.0
        h_mat = temperature_implicit_weights(levels, kappa, np.full(4, t_val))
        for k in range(4):
            # h0[k,k] = κ·T·α[k]/Δσ[k], then * Δσ[k] → κ·T·α[k]
            expected = kappa * t_val * alpha[k]
            np.testing.assert_allclose(
                h_mat[k, k],
                expected,
                rtol=1e-14,
                err_msg=f"H[{k},{k}] diagonal mismatch",
            )

    def test_hand_computed_2_levels(self) -> None:
        """2 uniform levels with T_ref=[280, 220]: verify against reference.

        Reference values precomputed with known-good implementation (Durran
        §8.6.5 formula with K terms for vertical advection of T_ref).
        """
        levels = uniform_sigma_levels(2)
        h_mat = temperature_implicit_weights(levels, kappa, np.array([280.0, 220.0]))
        expected = np.array([
            [58.94449154672439, -15.0],
            [67.61068791225253, 3.082873125540516],
        ])
        np.testing.assert_allclose(h_mat, expected, rtol=1e-13)

    def test_hand_computed_3_levels_varying(self) -> None:
        """3 uniform levels with T_ref=[300, 250, 200]: K terms are large."""
        levels = uniform_sigma_levels(3)
        h_mat = temperature_implicit_weights(levels, kappa, np.array([300.0, 250.0, 200.0]))
        expected = np.array([
            [63.75005046672852, -8.333333333333332, -8.333333333333334],
            [82.47992544407501, 18.24377227735681, -24.999999999999996],
            [33.34672582915905, 33.34672582915905, -6.248291992726393],
        ])
        np.testing.assert_allclose(h_mat, expected, rtol=1e-13)

    def test_hand_computed_4_levels_realistic(self) -> None:
        """4 uniform levels with realistic lapse-rate profile."""
        levels = uniform_sigma_levels(4)
        t_ref = np.array([210.0, 240.0, 265.0, 290.0])
        h_mat = temperature_implicit_weights(levels, kappa, t_ref)
        expected = np.array([
            [21.70836866004329, 3.75, 3.75, 3.75],
            [37.680728426312015, 15.014021386262538, 10.0, 10.0],
            [22.701276143229855, 22.701276143229855, 15.862877529231634, 15.625],
            [21.878593763196417, 21.878593763196417, 21.878593763196417, 20.439029674603304],
        ])
        np.testing.assert_allclose(h_mat, expected, rtol=1e-13)

    def test_5_levels_constant_matches_reference(self) -> None:
        """5 uniform levels, constant T_ref=250: K terms zero, match reference."""
        levels = uniform_sigma_levels(5)
        h_mat = temperature_implicit_weights(levels, kappa, np.full(5, 250.0))
        expected = np.array([
            [39.23615316671821, 0.0, 0.0, 0.0, 0.0],
            [57.479925444075015, 18.24377227735681, 0.0, 0.0, 0.0],
            [30.260637870971557, 30.260637870971557, 12.016865593614753, 0.0, 0.0],
            [20.992380889361407, 20.992380889361407, 20.992380889361414, 8.975515295746645, 0.0],
            [
                16.501266414162814,
                16.501266414162814,
                16.50126641416282,
                16.50126641416281,
                7.5257511184161645,
            ],
        ])
        np.testing.assert_allclose(h_mat, expected, rtol=1e-13)

    def test_k_terms_scale_with_gradient(self) -> None:
        """Doubling the T_ref gradient should double the K-term contribution."""
        levels = uniform_sigma_levels(3)
        t_ref_small = np.array([260.0, 250.0, 240.0])
        t_ref_large = np.array([270.0, 250.0, 230.0])
        h_small = temperature_implicit_weights(levels, kappa, t_ref_small)
        h_large = temperature_implicit_weights(levels, kappa, t_ref_large)

        # The h0 (adiabatic) part scales with T_ref values, but the K part
        # scales with diff(T_ref). Isolate K by subtracting constant-T_ref H.
        h_const = temperature_implicit_weights(levels, kappa, np.full(3, 250.0))
        k_small = h_small - h_const
        k_large = h_large - h_const

        # K terms don't scale exactly 2x because h0 also changes with T_ref,
        # but the upper-triangular part (pure K contribution) should scale
        # proportionally since h0 is lower triangular for any T_ref.
        k_upper_small = np.triu(k_small, k=1)
        k_upper_large = np.triu(k_large, k=1)
        # Gradient is 2x larger, so K upper part should be 2x
        np.testing.assert_allclose(k_upper_large, 2.0 * k_upper_small, rtol=1e-12)

    def test_k_terms_zero_for_constant_tref(self) -> None:
        """K terms vanish when T_ref is constant, reducing to pure h0."""
        levels = uniform_sigma_levels(4)
        alpha = sigma_ratios(levels)
        dsigma = np.asarray(levels.dsigma)
        t_val = 250.0
        h_mat = temperature_implicit_weights(levels, kappa, np.full(4, t_val))

        # For constant T_ref, H should match the old formula (h0 only)
        n = 4
        h0_only = np.zeros((n, n))
        for k in range(n):
            h0_only[k, k] = kappa * t_val * alpha[k] * dsigma[k]
            for j in range(k):
                h0_only[k, j] = (
                    kappa * t_val * dsigma[j] * (alpha[k] + alpha[k - 1]) / dsigma[k]
                ) * dsigma[j]
        # Wait, the old formula multiplied by dsigma[s] at the end.
        # Let me recompute properly for the new formula with constant T_ref.
        p = np.tril(np.ones((n, n)))
        alpha_col = alpha[:, np.newaxis]
        p_alpha = p * alpha_col
        p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
        p_alpha_shifted[0] = 0
        h0 = kappa * t_val * (p_alpha + p_alpha_shifted) / dsigma[:, np.newaxis]
        expected = h0 * dsigma  # K terms = 0

        np.testing.assert_allclose(h_mat, expected, rtol=1e-14)

    def test_single_level(self) -> None:
        levels = uniform_sigma_levels(1)
        alpha = sigma_ratios(levels)
        dsigma = np.asarray(levels.dsigma)
        h_mat = temperature_implicit_weights(levels, kappa, np.array([250.0]))
        assert h_mat.shape == (1, 1)
        # Single level: K terms are zero (no diff), H = κ·T·α·Δσ
        np.testing.assert_allclose(h_mat[0, 0], kappa * 250.0 * alpha[0] * dsigma[0], rtol=1e-14)

    def test_zero_temperature(self) -> None:
        levels = uniform_sigma_levels(5)
        h_mat = temperature_implicit_weights(levels, kappa, np.zeros(5))
        np.testing.assert_allclose(h_mat, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# pe_coupling_matrix (M = G·H + R·T_ref⊗Δσ)
# ---------------------------------------------------------------------------
class TestCouplingMatrix:
    """Tests for M = G·H + R·outer(T_ref, Δσ)."""

    def test_shape(self) -> None:
        levels = uniform_sigma_levels(5)
        m_mat = pe_coupling_matrix(levels, R, kappa, np.full(5, 250.0))
        assert m_mat.shape == (5, 5)

    def test_single_level_scalar(self) -> None:
        """Single level: M = R·T_ref·(κ·α₀² + 1)."""
        levels = uniform_sigma_levels(1)
        alpha = sigma_ratios(levels)
        t_val = 250.0
        m_mat = pe_coupling_matrix(levels, R, kappa, np.array([t_val]))
        expected = R * t_val * (kappa * alpha[0] ** 2 + 1.0)
        np.testing.assert_allclose(m_mat[0, 0], expected, rtol=1e-13)

    def test_positive_definite_isothermal(self) -> None:
        """For uniform T_ref, M should be positive definite (stability)."""
        levels = uniform_sigma_levels(10)
        m_mat = pe_coupling_matrix(levels, R, kappa, np.full(10, 250.0))
        eigvals = np.linalg.eigvalsh(m_mat)
        assert np.all(eigvals > 0), f"M not positive definite: eigenvalues {eigvals}"

    def test_positive_definite_varying_tref(self) -> None:
        """M should remain positive definite for realistic T_ref profiles."""
        for n in [3, 5, 10, 20]:
            levels = uniform_sigma_levels(n)
            t_ref = np.linspace(300.0, 200.0, n)
            m_mat = pe_coupling_matrix(levels, R, kappa, t_ref)
            eigvals = np.linalg.eigvalsh(m_mat)
            assert np.all(eigvals > 0), (
                f"M not positive definite at {n} levels: min eigenvalue {eigvals.min()}"
            )

    def test_decomposition_correct(self) -> None:
        """Verify M = G @ H + R · outer(T_ref, Δσ) explicitly."""
        levels = uniform_sigma_levels(4)
        t_ref = np.array([200.0, 230.0, 260.0, 290.0])
        g_mat = geopotential_weights(levels, R)
        h_mat = temperature_implicit_weights(levels, kappa, t_ref)
        m_mat = pe_coupling_matrix(levels, R, kappa, t_ref)
        expected = g_mat @ h_mat + R * np.outer(t_ref, np.asarray(levels.dsigma))
        np.testing.assert_allclose(m_mat, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# PESemiImplicitConfig builder
# ---------------------------------------------------------------------------
class TestBuildConfig:
    """Tests for build_pe_semi_implicit_config."""

    def test_creates_valid_config(self) -> None:
        levels = uniform_sigma_levels(5)
        t_ref = np.full(5, 250.0)
        config = build_pe_semi_implicit_config(levels, R, kappa, t_ref)
        assert config.coupling_matrix.shape == (5, 5)
        assert config.geopotential_weights.shape == (5, 5)
        assert config.temp_implicit_weights.shape == (5, 5)
        assert config.dsigma.shape == (5,)
        assert config.reference_temperature.shape == (5,)
        np.testing.assert_allclose(config.gas_constant, R)
        np.testing.assert_allclose(config.alpha, 0.5)

    def test_custom_alpha(self) -> None:
        levels = uniform_sigma_levels(3)
        config = build_pe_semi_implicit_config(
            levels,
            R,
            kappa,
            np.full(3, 250.0),
            alpha=0.6,
        )
        np.testing.assert_allclose(config.alpha, 0.6)


# ---------------------------------------------------------------------------
# pe_implicit_terms
# ---------------------------------------------------------------------------
class TestPEImplicitTerms:
    """Tests for the linear implicit tendency L(x)."""

    def _make_config(self, n_levels: int) -> PESemiImplicitConfig:
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.full(n_levels, 250.0)
        return build_pe_semi_implicit_config(levels, R, kappa, t_ref)

    def test_zero_state_gives_zero(self) -> None:
        n_levels = 5
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(N_SPEC, dtype=jnp.complex128),
        )
        tend = pe_implicit_terms(state, config, truncation=TRUNC, radius=EARTH.radius)

        np.testing.assert_allclose(jnp.abs(tend.vorticity), 0.0, atol=1e-20)
        np.testing.assert_allclose(jnp.abs(tend.divergence), 0.0, atol=1e-20)
        np.testing.assert_allclose(jnp.abs(tend.temperature), 0.0, atol=1e-20)
        np.testing.assert_allclose(jnp.abs(tend.log_surface_pressure), 0.0, atol=1e-20)

    def test_vorticity_always_zero(self) -> None:
        """Implicit terms never affect vorticity."""
        n_levels = 3
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.ones(N_SPEC, dtype=jnp.complex128),
        )
        tend = pe_implicit_terms(state, config, truncation=TRUNC, radius=EARTH.radius)
        np.testing.assert_allclose(jnp.abs(tend.vorticity), 0.0, atol=1e-20)

    def test_shapes(self) -> None:
        n_levels = 5
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.ones(N_SPEC, dtype=jnp.complex128),
        )
        tend = pe_implicit_terms(state, config, truncation=TRUNC, radius=EARTH.radius)
        assert tend.vorticity.shape == (n_levels, N_SPEC)
        assert tend.divergence.shape == (n_levels, N_SPEC)
        assert tend.temperature.shape == (n_levels, N_SPEC)
        assert tend.log_surface_pressure.shape == (N_SPEC,)

    def test_divergence_depends_on_temperature(self) -> None:
        """L_δ = -∇²(G·T + R·T_ref·lnps), so it should vary with T."""
        n_levels = 3
        config = self._make_config(n_levels)
        zero = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(N_SPEC, dtype=jnp.complex128),
        )
        nonzero = zero.replace(
            temperature=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
        )
        tend0 = pe_implicit_terms(zero, config, truncation=TRUNC, radius=EARTH.radius)
        tend1 = pe_implicit_terms(nonzero, config, truncation=TRUNC, radius=EARTH.radius)
        # L_δ = -eigenvalues * (G @ T + ...), nonzero for n>0 modes
        diff = jnp.max(jnp.abs(tend1.divergence - tend0.divergence))
        assert float(diff) > 1e-15, "L_δ should change when T changes"

    def test_temperature_depends_on_divergence(self) -> None:
        """L_T = -H·δ, so it should vary with δ."""
        n_levels = 3
        config = self._make_config(n_levels)
        zero = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(N_SPEC, dtype=jnp.complex128),
        )
        nonzero = zero.replace(
            divergence=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
        )
        tend0 = pe_implicit_terms(zero, config, truncation=TRUNC, radius=EARTH.radius)
        tend1 = pe_implicit_terms(nonzero, config, truncation=TRUNC, radius=EARTH.radius)
        assert not jnp.allclose(tend0.temperature, tend1.temperature)
        assert not jnp.allclose(tend0.log_surface_pressure, tend1.log_surface_pressure)

    def test_lnps_tendency_formula(self) -> None:
        """L_lnps = -Δσ·δ: verify by hand for 3 uniform levels."""
        n_levels = 3
        levels = uniform_sigma_levels(n_levels)
        config = build_pe_semi_implicit_config(levels, R, kappa, np.full(3, 250.0))
        # Use consistent n_spec with truncation
        div = jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128)
        # Set first 5 columns to known values for easy verification
        div = div.at[0, :5].set(jnp.array([1, 2, 3, 4, 5], dtype=jnp.complex128))
        div = div.at[1, :5].set(jnp.array([6, 7, 8, 9, 10], dtype=jnp.complex128))
        div = div.at[2, :5].set(jnp.array([11, 12, 13, 14, 15], dtype=jnp.complex128))
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=div,
            temperature=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(N_SPEC, dtype=jnp.complex128),
        )
        tend = pe_implicit_terms(state, config, truncation=TRUNC, radius=EARTH.radius)

        # L_lnps = -Σ Δσ_k · δ_k; uniform → Δσ = 1/3
        expected = -(div[0] + div[1] + div[2]) / 3.0
        np.testing.assert_allclose(tend.log_surface_pressure, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# pe_implicit_inverse
# ---------------------------------------------------------------------------
class TestPEImplicitInverse:
    """Tests for (I - s·L)⁻¹ solving the coupled (δ, T, lnps) system."""

    def _make_config(self, n_levels: int) -> PESemiImplicitConfig:
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.full(n_levels, 250.0)
        return build_pe_semi_implicit_config(levels, R, kappa, t_ref)

    def test_zero_step_identity(self) -> None:
        """With step_size=0, inverse is identity."""
        n_levels = 5
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.ones(N_SPEC, dtype=jnp.complex128),
        )
        result = pe_implicit_inverse(state, 0.0, config, TRUNC, EARTH.radius)

        np.testing.assert_allclose(result.vorticity, state.vorticity, atol=1e-14)
        np.testing.assert_allclose(result.divergence, state.divergence, atol=1e-14)
        np.testing.assert_allclose(result.temperature, state.temperature, atol=1e-14)
        np.testing.assert_allclose(
            result.log_surface_pressure,
            state.log_surface_pressure,
            atol=1e-14,
        )

    def test_zero_state_stays_zero(self) -> None:
        n_levels = 5
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(N_SPEC, dtype=jnp.complex128),
        )
        result = pe_implicit_inverse(state, 100.0, config, TRUNC, EARTH.radius)

        np.testing.assert_allclose(jnp.abs(result.divergence), 0.0, atol=1e-14)
        np.testing.assert_allclose(jnp.abs(result.temperature), 0.0, atol=1e-14)
        np.testing.assert_allclose(
            jnp.abs(result.log_surface_pressure),
            0.0,
            atol=1e-14,
        )

    def test_shapes(self) -> None:
        n_levels = 5
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.ones(N_SPEC, dtype=jnp.complex128),
        )
        result = pe_implicit_inverse(state, 100.0, config, TRUNC, EARTH.radius)

        assert result.vorticity.shape == (n_levels, N_SPEC)
        assert result.divergence.shape == (n_levels, N_SPEC)
        assert result.temperature.shape == (n_levels, N_SPEC)
        assert result.log_surface_pressure.shape == (N_SPEC,)

    def test_roundtrip_consistency(self) -> None:
        """If x₁ = (I-s·L)⁻¹·x₀, then x₀ = x₁ - s·L(x₁).

        Construct x₀ = x₁ - s·L(x₁) and verify the inverse recovers x₁.
        """
        n_levels = 3
        config = self._make_config(n_levels)
        s = 100.0

        # "True" solution x₁
        x1 = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=1e-3 * jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=0.1 * jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=0.01 * jnp.ones(N_SPEC, dtype=jnp.complex128),
        )

        # Compute L(x₁)
        lx1 = pe_implicit_terms(x1, config, TRUNC, EARTH.radius)

        # x₀ = x₁ - s·L(x₁)
        x0 = PrimitiveEquationState(
            vorticity=x1.vorticity - s * lx1.vorticity,
            divergence=x1.divergence - s * lx1.divergence,
            temperature=x1.temperature - s * lx1.temperature,
            log_surface_pressure=x1.log_surface_pressure - s * lx1.log_surface_pressure,
        )

        # (I - s·L)⁻¹ · x₀ should recover x₁
        recovered = pe_implicit_inverse(x0, s, config, TRUNC, EARTH.radius)

        np.testing.assert_allclose(recovered.divergence, x1.divergence, rtol=1e-10)
        np.testing.assert_allclose(recovered.temperature, x1.temperature, rtol=1e-10)
        np.testing.assert_allclose(
            recovered.log_surface_pressure,
            x1.log_surface_pressure,
            rtol=1e-10,
        )

    def test_n0_mode_passes_through(self) -> None:
        """The n=0 spectral mode (global mean) has eigenvalue 0.

        For divergence, the n=0 mode should pass through unchanged since
        -∇²(anything) = 0 for the global mean.
        """
        n_levels = 3
        config = self._make_config(n_levels)

        # State with only the n=0 mode nonzero (index 0)
        div = jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128)
        div = div.at[:, 0].set(1e-3 + 0j)
        state = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=div,
            temperature=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros(N_SPEC, dtype=jnp.complex128),
        )

        result = pe_implicit_inverse(state, 100.0, config, TRUNC, EARTH.radius)

        # n=0 divergence should be unchanged
        np.testing.assert_allclose(
            result.divergence[:, 0],
            state.divergence[:, 0],
            atol=1e-14,
        )
        # But T and lnps at n=0 change (back-substitution uses δ_new)
        # T_new = T* - s·H·δ_new = 0 - s·H·δ* (nonzero)
        assert jnp.any(jnp.abs(result.temperature[:, 0]) > 1e-10)

    def test_jit_compatible(self) -> None:
        """Verify the inverse can be called inside jax.jit."""
        n_levels = 3
        config = self._make_config(n_levels)
        state = PrimitiveEquationState(
            vorticity=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=jnp.ones(N_SPEC, dtype=jnp.complex128),
        )

        @jax.jit
        def solve(s: PrimitiveEquationState) -> PrimitiveEquationState:
            return pe_implicit_inverse(s, 100.0, config, TRUNC, EARTH.radius)

        result = solve(state)
        assert jnp.all(jnp.isfinite(result.divergence))
        assert jnp.all(jnp.isfinite(result.temperature))
        assert jnp.all(jnp.isfinite(result.log_surface_pressure))

    def test_single_level_finite_and_nontrivial(self) -> None:
        """Single level solve should produce finite, nontrivial results."""
        levels = uniform_sigma_levels(1)
        config = build_pe_semi_implicit_config(levels, R, kappa, np.array([250.0]))

        state = PrimitiveEquationState(
            vorticity=jnp.zeros((1, N_SPEC), dtype=jnp.complex128),
            divergence=1e-3 * jnp.ones((1, N_SPEC), dtype=jnp.complex128),
            temperature=0.1 * jnp.ones((1, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=0.01 * jnp.ones(N_SPEC, dtype=jnp.complex128),
        )

        result = pe_implicit_inverse(state, 100.0, config, TRUNC, EARTH.radius)

        assert jnp.all(jnp.isfinite(result.divergence))
        assert jnp.all(jnp.isfinite(result.temperature))
        assert jnp.all(jnp.isfinite(result.log_surface_pressure))
        # Should differ from input (implicit correction is nontrivial)
        assert not jnp.allclose(result.divergence, state.divergence)

    def test_roundtrip_varying_tref(self) -> None:
        """Roundtrip test with non-constant T_ref (exercises K terms in H)."""
        n_levels = 5
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.linspace(300.0, 200.0, n_levels)
        config = build_pe_semi_implicit_config(levels, R, kappa, t_ref)
        s = 200.0

        x1 = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=1e-4 * jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=0.05 * jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=0.005 * jnp.ones(N_SPEC, dtype=jnp.complex128),
        )

        lx1 = pe_implicit_terms(x1, config, TRUNC, EARTH.radius)
        x0 = PrimitiveEquationState(
            vorticity=x1.vorticity - s * lx1.vorticity,
            divergence=x1.divergence - s * lx1.divergence,
            temperature=x1.temperature - s * lx1.temperature,
            log_surface_pressure=x1.log_surface_pressure - s * lx1.log_surface_pressure,
        )

        recovered = pe_implicit_inverse(x0, s, config, TRUNC, EARTH.radius)

        np.testing.assert_allclose(recovered.divergence, x1.divergence, rtol=1e-10)
        np.testing.assert_allclose(recovered.temperature, x1.temperature, rtol=1e-10)
        np.testing.assert_allclose(
            recovered.log_surface_pressure,
            x1.log_surface_pressure,
            rtol=1e-10,
        )

    def test_roundtrip_many_levels(self) -> None:
        """Roundtrip at 20 levels (Held-Suarez target) with realistic T_ref."""
        n_levels = 20
        levels = uniform_sigma_levels(n_levels)
        t_ref = np.linspace(290.0, 210.0, n_levels)
        config = build_pe_semi_implicit_config(levels, R, kappa, t_ref)
        s = 600.0  # realistic: 2*dt*alpha with dt=600s

        x1 = PrimitiveEquationState(
            vorticity=jnp.zeros((n_levels, N_SPEC), dtype=jnp.complex128),
            divergence=1e-5 * jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            temperature=0.01 * jnp.ones((n_levels, N_SPEC), dtype=jnp.complex128),
            log_surface_pressure=1e-3 * jnp.ones(N_SPEC, dtype=jnp.complex128),
        )

        lx1 = pe_implicit_terms(x1, config, TRUNC, EARTH.radius)
        x0 = PrimitiveEquationState(
            vorticity=x1.vorticity - s * lx1.vorticity,
            divergence=x1.divergence - s * lx1.divergence,
            temperature=x1.temperature - s * lx1.temperature,
            log_surface_pressure=x1.log_surface_pressure - s * lx1.log_surface_pressure,
        )

        recovered = pe_implicit_inverse(x0, s, config, TRUNC, EARTH.radius)

        np.testing.assert_allclose(recovered.divergence, x1.divergence, rtol=1e-9)
        np.testing.assert_allclose(recovered.temperature, x1.temperature, rtol=1e-9)
        np.testing.assert_allclose(
            recovered.log_surface_pressure,
            x1.log_surface_pressure,
            rtol=1e-9,
        )
