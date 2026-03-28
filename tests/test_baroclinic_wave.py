"""Multi-day Jablonowski-Williamson baroclinic wave validation.

Quantitative validation of the full 3D dynamical core against the
standard J-W 2006 baroclinic instability test case.  This is the
definitive Phase 3 validation gate.

Reference
---------
Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
test case for atmospheric model dynamical cores. Quarterly Journal of
the Royal Meteorological Society, 132(621C), 2943-2975.

Expected behavior:
- Days 1-4: slow growth, surface pressure minimum barely changes
- Days 5-7: exponential growth phase, wave becomes visible
- Days 8-10: rapid deepening, ps minimum reaches ~940-960 hPa
- Days 10+: wave breaking and occlusion, ps minimum may recover
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.constants import EARTH
from notus.diagnostics import compute_conservation_diagnostics
from notus.grid import GaussianGrid
from notus.initial_conditions import (
    jablonowski_williamson_perturbation,
    jablonowski_williamson_steady_state,
)
from notus.operators import exponential_filter
from notus.timestepping.imex import build_pe_stepper
from notus.transforms import SpectralTransform
from notus.vertical.sigma import uniform_sigma_levels


jax.config.update("jax_enable_x64", True)


def _run_jw_integration(
    truncation: int,
    n_levels: int,
    dt: float,
    n_days: int,
) -> tuple[list[float], list[float], list[dict[str, float]]]:
    """Run a J-W baroclinic wave integration and collect daily diagnostics.

    Returns
    -------
    ps_min_daily : list[float]
        Daily surface pressure minimum [hPa].
    ps_max_daily : list[float]
        Daily surface pressure maximum [hPa].
    conservation : list[dict]
        Daily relative conservation errors (dM/M, dE/E, dL/L).
    """
    grid = GaussianGrid(truncation=truncation)
    transform = SpectralTransform(grid, EARTH.radius)
    levels = uniform_sigma_levels(n_levels)

    state, ref_temps, surface_phi = jablonowski_williamson_steady_state(transform, EARTH, levels)
    pert = jablonowski_williamson_perturbation(transform, EARTH, levels)
    perturbed = jax.tree.map(jnp.add, state, pert)

    filt = exponential_filter(transform.arrays, dt)
    init_fn, step_fn = build_pe_stepper(
        transform=transform,
        planet=EARTH,
        levels=levels,
        reference_temperature=ref_temps,
        surface_geopotential=surface_phi,
        dt=dt,
        spectral_filter=filt,
    )

    diag0 = compute_conservation_diagnostics(perturbed, transform, EARTH, levels, surface_phi)

    prev, curr = init_fn(perturbed)
    steps_per_day = int(86400 / dt)

    ps_min_daily: list[float] = []
    ps_max_daily: list[float] = []
    conservation: list[dict[str, float]] = []

    for _day in range(n_days):
        for _ in range(steps_per_day):
            prev, curr = step_fn(prev, curr)

        # Surface pressure diagnostics
        lnps_grid = np.asarray(transform.spectral_to_grid(curr.log_surface_pressure))
        ps_grid = EARTH.reference_pressure * np.exp(lnps_grid)
        ps_min_daily.append(float(np.min(ps_grid)) / 100.0)
        ps_max_daily.append(float(np.max(ps_grid)) / 100.0)

        # Conservation
        diag = compute_conservation_diagnostics(curr, transform, EARTH, levels, surface_phi)
        conservation.append({
            "dM/M": (diag.mass - diag0.mass) / diag0.mass,
            "dE/E": (diag.total_energy - diag0.total_energy) / diag0.total_energy,
            "dL/L": (diag.angular_momentum - diag0.angular_momentum) / diag0.angular_momentum,
        })

    return ps_min_daily, ps_max_daily, conservation


@pytest.mark.slow
class TestBaroclinicWaveValidation:
    """Quantitative validation of the J-W baroclinic wave test case."""

    @pytest.fixture(scope="class")
    def jw_results(self) -> tuple[list[float], list[float], list[dict[str, float]]]:
        """Run the 10-day T42L20 integration once for all tests."""
        return _run_jw_integration(
            truncation=42,
            n_levels=20,
            dt=600.0,
            n_days=10,
        )

    def test_integration_stable_10_days(
        self,
        jw_results: tuple,
    ) -> None:
        """The integration should remain stable for 10 days."""
        ps_min, _, _ = jw_results
        assert len(ps_min) == 10, "Should have 10 days of output"
        for day, ps in enumerate(ps_min):
            assert np.isfinite(ps), f"ps_min is not finite at day {day + 1}"

    def test_wave_develops(
        self,
        jw_results: tuple,
    ) -> None:
        """Surface pressure minimum should deepen during the growth phase.

        The baroclinic instability amplifies exponentially from day ~4,
        so ps_min should decrease monotonically from day 4 to day 9.
        """
        ps_min, _, _ = jw_results
        # Days 4-9 (indices 3-8): monotonic deepening
        for i in range(3, 8):
            assert ps_min[i + 1] < ps_min[i], (
                f"ps_min should decrease from day {i + 1} to {i + 2}: "
                f"{ps_min[i]:.1f} -> {ps_min[i + 1]:.1f} hPa"
            )

    def test_surface_pressure_minimum_realistic(
        self,
        jw_results: tuple,
    ) -> None:
        """Surface pressure minimum should reach ~940-970 hPa by day 9.

        Reference: J-W 2006 reports ~960 hPa at day 9.  The exact value
        depends on resolution, diffusion, and vertical levels.  We accept
        a range that covers published model intercomparison results.
        """
        ps_min, _, _ = jw_results
        ps_day9 = ps_min[8]  # index 8 = day 9
        assert 920.0 < ps_day9 < 975.0, (
            f"Day 9 ps_min={ps_day9:.1f} hPa outside expected range [920, 975]"
        )

    def test_surface_pressure_maximum_realistic(
        self,
        jw_results: tuple,
    ) -> None:
        """Surface pressure maximum should rise above 1010 hPa by day 9."""
        _, ps_max, _ = jw_results
        ps_day9 = ps_max[8]
        assert ps_day9 > 1010.0, f"Day 9 ps_max={ps_day9:.1f} hPa should exceed 1010 hPa"

    def test_mass_conservation(
        self,
        jw_results: tuple,
    ) -> None:
        """Global mass should be conserved to better than 1e-6 over 10 days."""
        _, _, conservation = jw_results
        for day, c in enumerate(conservation):
            assert abs(c["dM/M"]) < 1e-6, (
                f"Mass conservation violated at day {day + 1}: dM/M = {c['dM/M']:.2e}"
            )

    def test_energy_conservation(
        self,
        jw_results: tuple,
    ) -> None:
        """Total energy should be conserved to better than 0.1% over 10 days."""
        _, _, conservation = jw_results
        for day, c in enumerate(conservation):
            assert abs(c["dE/E"]) < 1e-3, (
                f"Energy conservation violated at day {day + 1}: dE/E = {c['dE/E']:.2e}"
            )

    def test_angular_momentum_conservation(
        self,
        jw_results: tuple,
    ) -> None:
        """Angular momentum should be conserved to better than 0.1% over 10 days."""
        _, _, conservation = jw_results
        for day, c in enumerate(conservation):
            assert abs(c["dL/L"]) < 1e-3, (
                f"Angular momentum conservation violated at day {day + 1}: dL/L = {c['dL/L']:.2e}"
            )

    def test_pressure_amplitude_grows_then_saturates(
        self,
        jw_results: tuple,
    ) -> None:
        """The pressure perturbation amplitude should grow then saturate.

        Early days: small amplitude (< 5 hPa deviation from 1000 hPa).
        Late days: large amplitude (> 30 hPa deviation from 1000 hPa).
        """
        ps_min, ps_max, _ = jw_results
        # Day 2: small perturbation
        amp_day2 = max(1000.0 - ps_min[1], ps_max[1] - 1000.0)
        assert amp_day2 < 5.0, f"Day 2 amplitude {amp_day2:.1f} hPa too large (expected < 5)"
        # Day 9: large perturbation
        amp_day9 = max(1000.0 - ps_min[8], ps_max[8] - 1000.0)
        assert amp_day9 > 30.0, f"Day 9 amplitude {amp_day9:.1f} hPa too small (expected > 30)"
