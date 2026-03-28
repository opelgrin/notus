"""Tests for the Monin-Obukhov surface layer parameterization.

Verifies the Louis (1979) stability functions, neutral drag coefficients,
bulk Richardson number, and transfer coefficient computation.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from notus.physics.boundary_layer import (
    VON_KARMAN,
    SurfaceLayerConfig,
    bulk_richardson_number,
    compute_transfer_coefficients,
    default_z_ref,
    louis_stability_functions,
    neutral_drag_coefficient,
    neutral_heat_coefficient,
    reference_height_from_sigma,
)


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Neutral drag coefficient
# ---------------------------------------------------------------------------


class TestNeutralDragCoefficient:
    """Tests for the neutral log-profile drag coefficient."""

    def test_analytical_value(self) -> None:
        """C_DN matches (k / ln(z/z0))^2 for known values."""
        z_ref = 100.0
        z0 = 1e-4
        expected = (VON_KARMAN / math.log(z_ref / z0)) ** 2
        result = float(neutral_drag_coefficient(z_ref, z0))
        assert result == pytest.approx(expected, rel=1e-12)

    def test_increases_with_roughness(self) -> None:
        """Larger roughness length gives larger drag."""
        z_ref = 100.0
        c_dn_smooth = float(neutral_drag_coefficient(z_ref, 1e-4))
        c_dn_rough = float(neutral_drag_coefficient(z_ref, 0.1))
        assert c_dn_rough > c_dn_smooth

    def test_decreases_with_height(self) -> None:
        """Higher reference level gives smaller drag."""
        z0 = 1e-3
        c_dn_low = float(neutral_drag_coefficient(50.0, z0))
        c_dn_high = float(neutral_drag_coefficient(500.0, z0))
        assert c_dn_low > c_dn_high

    def test_ocean_roughness_order_of_magnitude(self) -> None:
        """Ocean-like roughness gives C_DN ~ O(1e-3)."""
        z_ref = 250.0  # typical lowest level height at L20
        z0 = 1e-4
        c_dn = float(neutral_drag_coefficient(z_ref, z0))
        assert 1e-4 < c_dn < 1e-2


class TestNeutralHeatCoefficient:
    """Tests for the neutral heat transfer coefficient."""

    def test_equals_drag_when_z0_equal(self) -> None:
        """C_HN = C_DN when z0_m = z0_h."""
        z_ref = 100.0
        z0 = 1e-3
        c_dn = float(neutral_drag_coefficient(z_ref, z0))
        c_hn = float(neutral_heat_coefficient(z_ref, z0, z0))
        assert c_hn == pytest.approx(c_dn, rel=1e-12)

    def test_smaller_than_drag_when_z0h_smaller(self) -> None:
        """C_HN < C_DN when z0_h < z0_m (typical)."""
        z_ref = 100.0
        z0_m = 1e-3
        z0_h = 1e-4
        c_dn = float(neutral_drag_coefficient(z_ref, z0_m))
        c_hn = float(neutral_heat_coefficient(z_ref, z0_m, z0_h))
        assert c_hn < c_dn


# ---------------------------------------------------------------------------
# Bulk Richardson number
# ---------------------------------------------------------------------------


class TestBulkRichardsonNumber:
    """Tests for the bulk Richardson number."""

    def test_unstable_when_surface_warmer(self) -> None:
        """Ri_b < 0 when surface is warmer than air (convective)."""
        t_sfc = jnp.float64(310.0)
        t_air = jnp.float64(280.0)
        wind = jnp.float64(5.0)
        ri = float(bulk_richardson_number(t_sfc, t_air, wind, 100.0, 9.81))
        assert ri < 0.0

    def test_stable_when_surface_cooler(self) -> None:
        """Ri_b > 0 when surface is cooler than air (inversion)."""
        t_sfc = jnp.float64(270.0)
        t_air = jnp.float64(280.0)
        wind = jnp.float64(5.0)
        ri = float(bulk_richardson_number(t_sfc, t_air, wind, 100.0, 9.81))
        assert ri > 0.0

    def test_neutral_when_equal_temperature(self) -> None:
        """Ri_b = 0 when surface and air temperatures are equal."""
        t_sfc = jnp.float64(280.0)
        t_air = jnp.float64(280.0)
        wind = jnp.float64(5.0)
        ri = float(bulk_richardson_number(t_sfc, t_air, wind, 100.0, 9.81))
        assert ri == pytest.approx(0.0, abs=1e-10)

    def test_magnitude_increases_with_dt(self) -> None:
        """Larger temperature difference gives larger |Ri_b|."""
        t_air = jnp.float64(280.0)
        wind = jnp.float64(5.0)
        ri_small = abs(float(bulk_richardson_number(jnp.float64(285.0), t_air, wind, 100.0, 9.81)))
        ri_large = abs(float(bulk_richardson_number(jnp.float64(300.0), t_air, wind, 100.0, 9.81)))
        assert ri_large > ri_small

    def test_stronger_wind_reduces_magnitude(self) -> None:
        """Stronger wind reduces |Ri_b| (more mechanical mixing)."""
        t_sfc = jnp.float64(300.0)
        t_air = jnp.float64(280.0)
        ri_calm = abs(float(bulk_richardson_number(t_sfc, t_air, jnp.float64(2.0), 100.0, 9.81)))
        ri_windy = abs(float(bulk_richardson_number(t_sfc, t_air, jnp.float64(10.0), 100.0, 9.81)))
        assert ri_windy < ri_calm


# ---------------------------------------------------------------------------
# Louis stability functions
# ---------------------------------------------------------------------------


class TestLouisStabilityFunctions:
    """Tests for the Louis (1979) stability correction factors."""

    def test_neutral_gives_unity(self) -> None:
        """f_m = f_h = 1 at Ri_b = 0 (neutral conditions)."""
        ri_b = jnp.float64(0.0)
        z_ref = 100.0
        z0_m = 1e-4
        z0_h = 1e-5
        c_dn = neutral_drag_coefficient(z_ref, z0_m)
        c_hn = neutral_heat_coefficient(z_ref, z0_m, z0_h)
        f_m, f_h = louis_stability_functions(ri_b, z_ref, z0_m, z0_h, c_dn, c_hn)
        assert float(f_m) == pytest.approx(1.0, abs=1e-10)
        assert float(f_h) == pytest.approx(1.0, abs=1e-10)

    def test_unstable_enhances_transfer(self) -> None:
        """f_m > 1 and f_h > 1 for unstable conditions (Ri_b < 0)."""
        ri_b = jnp.float64(-0.5)
        z_ref = 100.0
        z0_m = 1e-4
        z0_h = 1e-5
        c_dn = neutral_drag_coefficient(z_ref, z0_m)
        c_hn = neutral_heat_coefficient(z_ref, z0_m, z0_h)
        f_m, f_h = louis_stability_functions(ri_b, z_ref, z0_m, z0_h, c_dn, c_hn)
        assert float(f_m) > 1.0
        assert float(f_h) > 1.0

    def test_stable_suppresses_transfer(self) -> None:
        """f_m < 1 and f_h < 1 for stable conditions (Ri_b > 0)."""
        ri_b = jnp.float64(0.5)
        z_ref = 100.0
        z0_m = 1e-4
        z0_h = 1e-5
        c_dn = neutral_drag_coefficient(z_ref, z0_m)
        c_hn = neutral_heat_coefficient(z_ref, z0_m, z0_h)
        f_m, f_h = louis_stability_functions(ri_b, z_ref, z0_m, z0_h, c_dn, c_hn)
        assert float(f_m) < 1.0
        assert float(f_h) < 1.0

    def test_stable_functions_positive(self) -> None:
        """Stability functions remain positive even for very stable conditions."""
        ri_b = jnp.float64(5.0)
        z_ref = 100.0
        z0_m = 1e-4
        z0_h = 1e-5
        c_dn = neutral_drag_coefficient(z_ref, z0_m)
        c_hn = neutral_heat_coefficient(z_ref, z0_m, z0_h)
        f_m, f_h = louis_stability_functions(ri_b, z_ref, z0_m, z0_h, c_dn, c_hn)
        assert float(f_m) > 0.0
        assert float(f_h) > 0.0

    def test_monotonic_in_ri(self) -> None:
        """More stable = smaller f; more unstable = larger f."""
        z_ref = 100.0
        z0_m = 1e-4
        z0_h = 1e-5
        c_dn = neutral_drag_coefficient(z_ref, z0_m)
        c_hn = neutral_heat_coefficient(z_ref, z0_m, z0_h)

        ri_values = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        f_m, f_h = louis_stability_functions(ri_values, z_ref, z0_m, z0_h, c_dn, c_hn)

        # f_m should be monotonically decreasing with increasing Ri_b
        for i in range(len(ri_values) - 1):
            assert float(f_m[i]) > float(f_m[i + 1])
            assert float(f_h[i]) > float(f_h[i + 1])


# ---------------------------------------------------------------------------
# Reference height
# ---------------------------------------------------------------------------


class TestReferenceHeight:
    """Tests for reference height estimation from sigma levels."""

    def test_positive(self) -> None:
        """Reference height is positive."""
        z_ref = reference_height_from_sigma(0.05, jnp.array(280.0), 9.81, 287.04)
        assert float(z_ref) > 0.0

    def test_increases_with_dsigma(self) -> None:
        """Thicker lowest level gives higher reference height."""
        z_low = float(reference_height_from_sigma(0.02, jnp.array(280.0), 9.81, 287.04))
        z_high = float(reference_height_from_sigma(0.10, jnp.array(280.0), 9.81, 287.04))
        assert z_high > z_low

    def test_typical_l20_value(self) -> None:
        """For L20 (dsigma ~ 0.05), z_ref should be O(100m)."""
        z_ref = float(reference_height_from_sigma(0.05, jnp.array(280.0), 9.81, 287.04))
        assert 50.0 < z_ref < 1000.0

    def test_default_z_ref_matches(self) -> None:
        """default_z_ref gives the same result as reference_height_from_sigma."""
        dsigma = 0.05
        t_ref = 280.0
        g = 9.81
        r = 287.04
        z1 = float(reference_height_from_sigma(dsigma, jnp.array(t_ref), g, r))
        z2 = default_z_ref(dsigma, t_ref, g, r)
        assert z1 == pytest.approx(z2, rel=1e-10)


# ---------------------------------------------------------------------------
# Transfer coefficient computation (integration)
# ---------------------------------------------------------------------------


class TestComputeTransferCoefficients:
    """Tests for the full transfer coefficient computation."""

    @pytest.fixture
    def config(self) -> SurfaceLayerConfig:
        return SurfaceLayerConfig(z0_momentum=1e-4)

    def test_returns_two_arrays(self, config: SurfaceLayerConfig) -> None:
        """Returns momentum and heat transfer coefficients."""
        t_sfc = jnp.full((4, 8), 300.0)
        t_air = jnp.full((4, 8), 290.0)
        wind = jnp.full((4, 8), 5.0)
        c_d, c_h = compute_transfer_coefficients(
            t_sfc,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            config,
        )
        assert c_d.shape == (4, 8)
        assert c_h.shape == (4, 8)

    def test_positive_coefficients(self, config: SurfaceLayerConfig) -> None:
        """Transfer coefficients are always positive."""
        t_sfc = jnp.full((4, 8), 300.0)
        t_air = jnp.full((4, 8), 290.0)
        wind = jnp.full((4, 8), 5.0)
        c_d, c_h = compute_transfer_coefficients(
            t_sfc,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            config,
        )
        assert jnp.all(c_d > 0)
        assert jnp.all(c_h > 0)

    def test_unstable_larger_than_neutral(self, config: SurfaceLayerConfig) -> None:
        """Unstable conditions (warm surface) give larger coefficients than neutral."""
        wind = jnp.full((4, 8), 5.0)
        t_air = jnp.full((4, 8), 280.0)

        # Neutral
        t_sfc_neutral = jnp.full((4, 8), 280.0)
        c_d_n, c_h_n = compute_transfer_coefficients(
            t_sfc_neutral,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            config,
        )

        # Unstable
        t_sfc_warm = jnp.full((4, 8), 310.0)
        c_d_u, c_h_u = compute_transfer_coefficients(
            t_sfc_warm,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            config,
        )

        assert jnp.all(c_d_u > c_d_n)
        assert jnp.all(c_h_u > c_h_n)

    def test_stable_smaller_than_neutral(self, config: SurfaceLayerConfig) -> None:
        """Stable conditions (cool surface) give smaller coefficients than neutral."""
        wind = jnp.full((4, 8), 5.0)
        t_air = jnp.full((4, 8), 280.0)

        # Neutral
        t_sfc_neutral = jnp.full((4, 8), 280.0)
        c_d_n, c_h_n = compute_transfer_coefficients(
            t_sfc_neutral,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            config,
        )

        # Stable
        t_sfc_cool = jnp.full((4, 8), 260.0)
        c_d_s, c_h_s = compute_transfer_coefficients(
            t_sfc_cool,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            config,
        )

        assert jnp.all(c_d_s < c_d_n)
        assert jnp.all(c_h_s < c_h_n)

    def test_rougher_surface_larger_drag(self) -> None:
        """Rougher surface gives larger transfer coefficients."""
        t_sfc = jnp.full((4, 8), 290.0)
        t_air = jnp.full((4, 8), 280.0)
        wind = jnp.full((4, 8), 5.0)

        smooth = SurfaceLayerConfig(z0_momentum=1e-4)
        rough = SurfaceLayerConfig(z0_momentum=0.1)

        c_d_smooth, _ = compute_transfer_coefficients(
            t_sfc,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            smooth,
        )
        c_d_rough, _ = compute_transfer_coefficients(
            t_sfc,
            t_air,
            wind,
            0.05,
            9.81,
            287.04,
            rough,
        )

        assert jnp.all(c_d_rough > c_d_smooth)

    def test_jit_compatible(self, config: SurfaceLayerConfig) -> None:
        """compute_transfer_coefficients works under jax.jit."""
        t_sfc = jnp.full((4, 8), 300.0)
        t_air = jnp.full((4, 8), 290.0)
        wind = jnp.full((4, 8), 5.0)

        @jax.jit
        def f(ts: jnp.ndarray, ta: jnp.ndarray, w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return compute_transfer_coefficients(ts, ta, w, 0.05, 9.81, 287.04, config)

        c_d, c_h = f(t_sfc, t_air, wind)
        assert c_d.shape == (4, 8)
        assert jnp.all(jnp.isfinite(c_d))
        assert jnp.all(jnp.isfinite(c_h))


# ---------------------------------------------------------------------------
# SurfaceLayerConfig
# ---------------------------------------------------------------------------


class TestSurfaceLayerConfig:
    """Tests for SurfaceLayerConfig defaults."""

    def test_default_z0_heat(self) -> None:
        """Default z0_heat is z0_momentum / 10."""
        cfg = SurfaceLayerConfig(z0_momentum=1e-3)
        assert cfg.z0_heat_effective == pytest.approx(1e-4)

    def test_explicit_z0_heat(self) -> None:
        """Explicit z0_heat overrides the default."""
        cfg = SurfaceLayerConfig(z0_momentum=1e-3, z0_heat=5e-4)
        assert cfg.z0_heat_effective == pytest.approx(5e-4)
