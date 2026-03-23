"""Integration tests for the shallow water solver.

Tests based on Williamson et al. (1992) standard test cases for
shallow water models on the sphere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from notus import EARTH, GaussianGrid, SpectralTransform
from notus.dynamics.shallow_water import shallow_water_tendencies
from notus.operators import exponential_filter
from notus.state import ShallowWaterState
from notus.timestepping.leapfrog import LeapfrogState
from notus.timestepping.semi_implicit import (
    SemiImplicitConfig,
    implicit_inverse,
    implicit_terms,
)


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

    Uses IMEX leapfrog (Dinosaur-style) with Robert-Asselin filter
    and exponential spectral filter for dealiasing stability.
    """
    tendency_fn = shallow_water_tendencies(transform, EARTH)

    si_config = SemiImplicitConfig(mean_geopotential=mean_phi)
    grid = transform.grid
    t = grid.truncation
    a = EARTH.radius
    alpha = si_config.alpha

    # Exponential spectral filter (Hou & Li 2007)
    exp_filter = exponential_filter(t, dt)

    def _apply_filter(s: ShallowWaterState) -> ShallowWaterState:
        return s.replace(
            vorticity=s.vorticity * exp_filter,
            divergence=s.divergence * exp_filter,
            geopotential=s.geopotential * exp_filter,
        )

    # ---- First step: backward-forward Euler ----
    explicit = tendency_fn(state)
    intermediate = jax.tree.map(lambda x, f: x + dt * f, state, explicit)
    # Implicit solve: (I - dt·L)⁻¹
    delta_new, phi_new = implicit_inverse(
        intermediate.divergence,
        intermediate.geopotential,
        dt,
        si_config,
        t,
        a,
    )
    current = _apply_filter(intermediate.replace(divergence=delta_new, geopotential=phi_new))
    lf_state = LeapfrogState(current=current, previous=state)

    # Robert-Asselin filter coefficient (Dinosaur uses 0.05)
    r = 0.05

    # ---- Subsequent steps: IMEX leapfrog ----
    for _step in range(n_steps - 1):
        explicit_current = tendency_fn(lf_state.current)

        # Implicit tendency at the previous time level
        l_div_prev, l_phi_prev = implicit_terms(
            lf_state.previous.divergence,
            lf_state.previous.geopotential,
            si_config,
            t,
            a,
        )

        # Leapfrog: intermediate = x_{n-1} + 2dt*(F(x_n) + (1-α)*L(x_{n-1}))
        intermediate = ShallowWaterState(
            vorticity=lf_state.previous.vorticity + 2.0 * dt * explicit_current.vorticity,
            divergence=lf_state.previous.divergence
            + 2.0 * dt * (explicit_current.divergence + (1.0 - alpha) * l_div_prev),
            geopotential=lf_state.previous.geopotential
            + 2.0 * dt * (explicit_current.geopotential + (1.0 - alpha) * l_phi_prev),
        )

        # Implicit solve: (I - η·L)⁻¹ where η = 2dt·α
        eta = 2.0 * dt * alpha
        delta_new, phi_new = implicit_inverse(
            intermediate.divergence,
            intermediate.geopotential,
            eta,
            si_config,
            t,
            a,
        )
        future = intermediate.replace(divergence=delta_new, geopotential=phi_new)

        # Robert-Asselin filter
        filtered_current = jax.tree.map(
            lambda p, c, f: (1.0 - 2.0 * r) * c + r * (p + f),
            lf_state.previous,
            lf_state.current,
            future,
        )

        # Exponential spectral filter on the future state
        future = _apply_filter(future)

        lf_state = LeapfrogState(current=future, previous=filtered_current)

    return lf_state.current


class TestWilliamsonCase2:
    """Williamson test case 2: steady-state nonlinear geostrophic flow.

    The balanced solid-body rotation should remain unchanged.
    """

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
