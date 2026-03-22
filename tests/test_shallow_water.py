"""Integration tests for the shallow water solver.

Tests based on Williamson et al. (1992) standard test cases for
shallow water models on the sphere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from notus import EARTH, GaussianGrid, SpectralTransform
from notus.dynamics.shallow_water import shallow_water_tendencies
from notus.state import ShallowWaterState
from notus.timestepping.leapfrog import LeapfrogState, euler_step
from notus.timestepping.semi_implicit import SemiImplicitConfig, semi_implicit_correction


jax.config.update("jax_enable_x64", True)


def _williamson2_initial_state(
    grid: GaussianGrid,
    transform: SpectralTransform,
    u0: float = 38.61,
    phi0: float = 2.94e4,
) -> ShallowWaterState:
    """Initialize Williamson test case 2: steady-state nonlinear zonal flow.

    Balanced solid-body rotation with:
        u = u₀·cos(φ), v = 0
        Φ = Φ₀ − (a·Ω·u₀ + u₀²/2)·sin²(φ)

    Parameters
    ----------
    grid : GaussianGrid
    transform : SpectralTransform
    u0 : float
        Maximum wind speed [m/s]. Default: Williamson (1992) value.
    phi0 : float
        Mean geopotential [m²/s²]. Default: Williamson (1992) value.
    """
    a = EARTH.radius
    omega = EARTH.rotation_rate

    # Geopotential on the grid
    sin2 = grid.sin_lat[:, None] ** 2
    phi_grid = phi0 - (a * omega * u0 + u0**2 / 2.0) * sin2
    phi_grid = jnp.broadcast_to(phi_grid, (grid.n_lat, grid.n_lon))

    # Vorticity: ζ = 2·u₀·sin(φ)/a
    vort_grid = 2.0 * u0 * grid.sin_lat[:, None] / a * jnp.ones((1, grid.n_lon))

    div_grid = jnp.zeros((grid.n_lat, grid.n_lon))

    return ShallowWaterState(
        vorticity=transform.grid_to_spectral(vort_grid),
        divergence=transform.grid_to_spectral(div_grid),
        geopotential=transform.grid_to_spectral(phi_grid),
    )


def _run_shallow_water(
    state: ShallowWaterState,
    transform: SpectralTransform,
    n_steps: int,
    dt: float,
    mean_phi: float = 2.94e4,
) -> ShallowWaterState:
    """Run the shallow water model for n_steps.

    Uses leapfrog with RAW filter and semi-implicit gravity waves.
    The SI correction is applied AFTER the explicit leapfrog prediction.
    """
    si_config = SemiImplicitConfig(mean_geopotential=mean_phi)
    grid = transform.grid
    t = grid.truncation
    a = EARTH.radius

    # First step: forward Euler (half step for SI)
    tend = shallow_water_tendencies(state, transform, EARTH)
    euler_state = euler_step(state, tend, dt)

    # Apply SI to the Euler prediction
    delta_new, phi_new = semi_implicit_correction(
        euler_state.current.divergence,
        euler_state.current.geopotential,
        state.divergence,
        state.geopotential,
        dt,
        si_config,
        t,
        a,
    )
    lf_state = LeapfrogState(
        current=euler_state.current.replace(divergence=delta_new, geopotential=phi_new),
        previous=state,
    )

    # RAW filter constants
    raw_curr = 0.04 * 0.53 / 2.0
    raw_new = 0.04 * (1.0 - 0.53) / 2.0

    # Subsequent steps: leapfrog + SI + RAW
    for _step in range(n_steps - 1):
        tend = shallow_water_tendencies(lf_state.current, transform, EARTH)

        # Explicit leapfrog prediction
        explicit_new = jax.tree.map(
            lambda xp, f: xp + 2.0 * dt * f,
            lf_state.previous,
            tend,
        )

        # SI correction on divergence and geopotential
        delta_new, phi_new = semi_implicit_correction(
            explicit_new.divergence,
            explicit_new.geopotential,
            lf_state.previous.divergence,
            lf_state.previous.geopotential,
            dt,
            si_config,
            t,
            a,
        )
        corrected_new = explicit_new.replace(
            divergence=delta_new, geopotential=phi_new
        )

        # RAW filter
        d = jax.tree.map(
            lambda xp, xc, xn: xp - 2.0 * xc + xn,
            lf_state.previous,
            lf_state.current,
            corrected_new,
        )
        filtered_curr = jax.tree.map(
            lambda xc, dd: xc + raw_curr * dd,
            lf_state.current,
            d,
        )
        filtered_new = jax.tree.map(
            lambda xn, dd: xn - raw_new * dd,
            corrected_new,
            d,
        )

        lf_state = LeapfrogState(current=filtered_new, previous=filtered_curr)

    return lf_state.current


class TestWilliamsonCase2:
    """Williamson test case 2: steady-state nonlinear geostrophic flow.

    The balanced solid-body rotation should remain unchanged.
    """

    @pytest.mark.xfail(reason="SI scheme needs debugging — aliasing instability at large dt")
    def test_steady_state_1day(self):
        """After 1 day, the state should not have drifted significantly."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        initial = _williamson2_initial_state(grid, transform)

        dt = 1200.0  # 20-minute timestep
        n_steps = int(86400 / dt)  # 1 day

        final = _run_shallow_water(initial, transform, n_steps, dt)

        # Compare geopotential
        phi_init = transform.spectral_to_grid(initial.geopotential)
        phi_final = transform.spectral_to_grid(final.geopotential)

        rel_error = jnp.max(jnp.abs(phi_final - phi_init)) / jnp.max(jnp.abs(phi_init))
        assert rel_error < 1e-4, f"Geopotential relative error: {rel_error:.2e}"

    @pytest.mark.xfail(reason="SI scheme needs debugging — aliasing instability at large dt")
    def test_steady_state_5days(self):
        """After 5 days, the state should still be close to initial."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        initial = _williamson2_initial_state(grid, transform)

        dt = 1200.0
        n_steps = int(5 * 86400 / dt)

        final = _run_shallow_water(initial, transform, n_steps, dt)

        phi_init = transform.spectral_to_grid(initial.geopotential)
        phi_final = transform.spectral_to_grid(final.geopotential)

        rel_error = jnp.max(jnp.abs(phi_final - phi_init)) / jnp.max(jnp.abs(phi_init))
        assert rel_error < 1e-3, f"Geopotential relative error after 5 days: {rel_error:.2e}"


class TestConservation:
    """Conservation properties of the shallow water solver."""

    @pytest.mark.xfail(reason="SI scheme needs debugging — aliasing instability at large dt")
    def test_mass_conservation(self):
        """Global mean geopotential (total mass) should be conserved."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        initial = _williamson2_initial_state(grid, transform)

        dt = 1200.0
        n_steps = int(86400 / dt)  # 1 day

        final = _run_shallow_water(initial, transform, n_steps, dt)

        # Global mean is the (m=0, n=0) spectral coefficient
        mass_init = initial.geopotential[0].real
        mass_final = final.geopotential[0].real

        rel_error = jnp.abs(mass_final - mass_init) / jnp.abs(mass_init)
        assert rel_error < 1e-10, f"Mass conservation error: {rel_error:.2e}"

    @pytest.mark.xfail(reason="SI scheme needs debugging — aliasing instability at large dt")
    def test_no_blowup(self):
        """Model should not blow up after 10 days."""
        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid)
        initial = _williamson2_initial_state(grid, transform)

        dt = 1200.0
        n_steps = int(10 * 86400 / dt)

        final = _run_shallow_water(initial, transform, n_steps, dt)

        phi_final = transform.spectral_to_grid(final.geopotential)
        assert jnp.all(jnp.isfinite(phi_final)), "Model blew up (NaN/Inf in geopotential)"
        assert jnp.max(jnp.abs(phi_final)) < 1e6, "Geopotential grew unreasonably large"
