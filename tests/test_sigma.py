"""Tests for the sigma vertical coordinate."""

import numpy as np
import pytest

from notus.vertical.sigma import SigmaLevels, standard_sigma_levels, uniform_sigma_levels


class TestSigmaLevels:
    """Tests for SigmaLevels construction and validation."""

    def test_basic_construction(self) -> None:
        sigma_half = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        levels = SigmaLevels(sigma_half)

        assert levels.n_levels == 4
        assert levels.sigma_half.shape == (5,)
        assert levels.sigma_full.shape == (4,)
        assert levels.dsigma.shape == (4,)

    def test_interfaces_boundary_values(self) -> None:
        levels = uniform_sigma_levels(5)
        np.testing.assert_equal(float(levels.sigma_half[0]), 0.0)
        np.testing.assert_equal(float(levels.sigma_half[-1]), 1.0)

    def test_midpoints_between_interfaces(self) -> None:
        levels = uniform_sigma_levels(10)
        for k in range(levels.n_levels):
            assert float(levels.sigma_half[k]) < float(levels.sigma_full[k])
            assert float(levels.sigma_full[k]) < float(levels.sigma_half[k + 1])

    def test_dsigma_sums_to_one(self) -> None:
        levels = uniform_sigma_levels(7)
        np.testing.assert_allclose(float(np.sum(levels.dsigma)), 1.0, atol=1e-15)

    def test_dsigma_matches_interface_differences(self) -> None:
        sigma_half = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        levels = SigmaLevels(sigma_half)
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(levels.dsigma, expected, atol=1e-15)

    def test_monotonically_increasing_interfaces(self) -> None:
        levels = standard_sigma_levels(20)
        diffs = np.diff(levels.sigma_half)
        assert np.all(diffs > 0)

    def test_monotonically_increasing_midpoints(self) -> None:
        levels = standard_sigma_levels(20)
        diffs = np.diff(levels.sigma_full)
        assert np.all(diffs > 0)

    def test_rejects_non_monotonic(self) -> None:
        with pytest.raises(ValueError, match="monotonically increasing"):
            SigmaLevels(np.array([0.0, 0.5, 0.3, 1.0]))

    def test_rejects_wrong_top(self) -> None:
        with pytest.raises(ValueError, match="start at 0"):
            SigmaLevels(np.array([0.1, 0.5, 1.0]))

    def test_rejects_wrong_surface(self) -> None:
        with pytest.raises(ValueError, match="end at 1"):
            SigmaLevels(np.array([0.0, 0.5, 0.9]))

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            SigmaLevels(np.array([0.0]))

    def test_single_level(self) -> None:
        levels = uniform_sigma_levels(1)
        assert levels.n_levels == 1
        np.testing.assert_allclose(levels.sigma_half, [0.0, 1.0])
        np.testing.assert_allclose(levels.sigma_full, [0.5])
        np.testing.assert_allclose(levels.dsigma, [1.0])


class TestUniformSigmaLevels:
    """Tests for the uniform factory function."""

    def test_equispaced(self) -> None:
        levels = uniform_sigma_levels(4)
        expected_half = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(levels.sigma_half, expected_half, atol=1e-15)

    def test_equal_thicknesses(self) -> None:
        n = 8
        levels = uniform_sigma_levels(n)
        expected_ds = np.full(n, 1.0 / n)
        np.testing.assert_allclose(levels.dsigma, expected_ds, atol=1e-15)

    def test_midpoints_centered(self) -> None:
        levels = uniform_sigma_levels(5)
        for k in range(levels.n_levels):
            expected = 0.5 * (levels.sigma_half[k] + levels.sigma_half[k + 1])
            np.testing.assert_allclose(levels.sigma_full[k], expected, atol=1e-15)

    def test_rejects_zero_levels(self) -> None:
        with pytest.raises(ValueError, match="n_levels must be >= 1"):
            uniform_sigma_levels(0)


class TestStandardSigmaLevels:
    """Tests for the standard (stretched) factory function."""

    def test_default_20_levels(self) -> None:
        levels = standard_sigma_levels()
        assert levels.n_levels == 20

    def test_boundary_values(self) -> None:
        levels = standard_sigma_levels(20)
        np.testing.assert_equal(float(levels.sigma_half[0]), 0.0)
        np.testing.assert_equal(float(levels.sigma_half[-1]), 1.0)

    def test_dsigma_sums_to_one(self) -> None:
        levels = standard_sigma_levels(20)
        np.testing.assert_allclose(float(np.sum(levels.dsigma)), 1.0, atol=1e-14)

    def test_thinner_at_top_than_bottom(self) -> None:
        """Standard profile should have thinner layers near the top."""
        levels = standard_sigma_levels(20)
        assert float(levels.dsigma[0]) < float(levels.dsigma[-1])

    def test_various_level_counts(self) -> None:
        for n in [5, 10, 20, 40]:
            levels = standard_sigma_levels(n)
            assert levels.n_levels == n
            np.testing.assert_allclose(float(np.sum(levels.dsigma)), 1.0, atol=1e-14)

    def test_rejects_zero_levels(self) -> None:
        with pytest.raises(ValueError, match="n_levels must be >= 1"):
            standard_sigma_levels(0)
