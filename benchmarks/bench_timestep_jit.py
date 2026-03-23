"""Benchmark: shallow water IMEX leapfrog throughput.

Measures wall-clock time per step at T21 and T42 resolutions,
reporting both compilation and steady-state throughput.

Usage:
    uv run python benchmarks/bench_timestep_jit.py
"""

from __future__ import annotations

import time

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


def williamson2_initial_state(
    grid: GaussianGrid,
    transform: SpectralTransform,
    u0: float = 38.61,
    phi0: float = 2.94e4,
) -> ShallowWaterState:
    a = EARTH.radius
    omega = EARTH.rotation_rate
    sin2 = grid.sin_lat[:, None] ** 2
    phi_grid = phi0 - (a * omega * u0 + u0**2 / 2.0) * sin2
    phi_grid = jnp.broadcast_to(phi_grid, (grid.n_lat, grid.n_lon))
    vort_grid = 2.0 * u0 * grid.sin_lat[:, None] / a * jnp.ones((1, grid.n_lon))
    div_grid = jnp.zeros((grid.n_lat, grid.n_lon))
    return ShallowWaterState(
        vorticity=transform.grid_to_spectral(vort_grid),
        divergence=transform.grid_to_spectral(div_grid),
        geopotential=transform.grid_to_spectral(phi_grid),
    )


def run_shallow_water(
    state: ShallowWaterState,
    transform: SpectralTransform,
    n_steps: int,
    dt: float,
    mean_phi: float = 2.94e4,
) -> ShallowWaterState:
    tendency_fn = shallow_water_tendencies(transform, EARTH)

    si_config = SemiImplicitConfig(mean_geopotential=mean_phi)
    grid = transform.grid
    t = grid.truncation
    a = EARTH.radius
    alpha = si_config.alpha
    exp_filter = exponential_filter(t, dt)
    r = 0.05

    def _apply_filter(s: ShallowWaterState) -> ShallowWaterState:
        return s.replace(
            vorticity=s.vorticity * exp_filter,
            divergence=s.divergence * exp_filter,
            geopotential=s.geopotential * exp_filter,
        )

    # First step: forward Euler + implicit solve
    explicit = tendency_fn(state)
    intermediate = jax.tree.map(lambda x, f: x + dt * f, state, explicit)
    delta_new, phi_new = implicit_inverse(
        intermediate.divergence, intermediate.geopotential,
        dt, si_config, t, a,
    )
    current = _apply_filter(
        intermediate.replace(divergence=delta_new, geopotential=phi_new)
    )
    lf_state = LeapfrogState(current=current, previous=state)

    # Subsequent steps: IMEX leapfrog
    for _ in range(n_steps - 1):
        explicit_current = tendency_fn(lf_state.current)
        l_div_prev, l_phi_prev = implicit_terms(
            lf_state.previous.divergence,
            lf_state.previous.geopotential,
            si_config, t, a,
        )
        intermediate = ShallowWaterState(
            vorticity=lf_state.previous.vorticity
            + 2.0 * dt * explicit_current.vorticity,
            divergence=lf_state.previous.divergence
            + 2.0 * dt * (explicit_current.divergence + (1.0 - alpha) * l_div_prev),
            geopotential=lf_state.previous.geopotential
            + 2.0 * dt * (explicit_current.geopotential + (1.0 - alpha) * l_phi_prev),
        )
        eta = 2.0 * dt * alpha
        delta_new, phi_new = implicit_inverse(
            intermediate.divergence, intermediate.geopotential,
            eta, si_config, t, a,
        )
        future = intermediate.replace(divergence=delta_new, geopotential=phi_new)
        filtered_current = jax.tree.map(
            lambda p, c, f: (1.0 - 2.0 * r) * c + r * (p + f),
            lf_state.previous, lf_state.current, future,
        )
        future = _apply_filter(future)
        lf_state = LeapfrogState(current=future, previous=filtered_current)

    return lf_state.current


def bench(label: str, fn, *args, warmup: int = 2, repeats: int = 5):
    """Time fn(*args), reporting warmup and steady-state median."""
    t0 = time.perf_counter()
    for _ in range(warmup):
        result = fn(*args)
        jax.block_until_ready(jax.tree.leaves(result))
    warmup_time = (time.perf_counter() - t0) / warmup

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(jax.tree.leaves(result))
        times.append(time.perf_counter() - t0)

    median = sorted(times)[len(times) // 2]
    print(f"  {label}:")
    print(f"    warmup (incl. compile): {warmup_time:.4f}s")
    print(f"    median of {repeats} runs:     {median:.4f}s")
    print(f"    all runs: {['%.4f' % t for t in times]}")
    return result, median


def main():
    print("=" * 60)
    print("Benchmark: shallow water IMEX leapfrog throughput")
    print("=" * 60)

    configs = [
        ("T21", 21, 72, 1200.0),
        ("T42", 42, 72, 600.0),
    ]

    for label, trunc, n_steps, dt in configs:
        print(f"\n--- {label} | {n_steps} steps | dt={dt}s ---")
        grid = GaussianGrid(truncation=trunc)
        transform = SpectralTransform(grid)
        initial = williamson2_initial_state(grid, transform)
        print(f"    grid: {grid.n_lat}x{grid.n_lon}, n_spectral={grid.n_spectral_coeffs}")

        result, median = bench(
            "shallow_water", run_shallow_water, initial, transform, n_steps, dt
        )
        print(f"    per-step: {median / n_steps * 1000:.2f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
