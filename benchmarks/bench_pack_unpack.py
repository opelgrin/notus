"""Benchmark: index gather/scatter vs fori_loop for spectral pack/unpack.

Compares the old fori_loop-based _pack_spectral/_unpack_spectral against
the new pre-computed index array approach, both in isolation and as part
of the full spectral transform round-trip.

Usage:
    uv run python benchmarks/bench_pack_unpack.py
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


# ---------------------------------------------------------------------------
# Old fori_loop implementations (baseline)
# ---------------------------------------------------------------------------

def _pack_spectral_old(arr_3d: jnp.ndarray, truncation: int) -> jnp.ndarray:
    max_len = truncation + 1
    n_spectral = max_len * (max_len + 1) // 2
    padded_size = n_spectral + max_len
    flat = jnp.zeros(padded_size, dtype=arr_3d.dtype)

    def body(m: int, flat: jnp.ndarray) -> jnp.ndarray:
        start = m * max_len - m * (m - 1) // 2
        row = arr_3d[m, :]
        return jax.lax.dynamic_update_slice(flat, row, (start,))

    return jax.lax.fori_loop(0, max_len, body, flat)[:n_spectral]


def _unpack_spectral_old(flat: jnp.ndarray, truncation: int) -> jnp.ndarray:
    max_len = truncation + 1
    padded = jnp.pad(flat, (0, max_len), mode="constant")
    arr = jnp.zeros((max_len, max_len), dtype=flat.dtype)

    def body(m: int, arr: jnp.ndarray) -> jnp.ndarray:
        start = m * max_len - m * (m - 1) // 2
        row = jax.lax.dynamic_slice(padded, (start,), (max_len,))
        return arr.at[m, :].set(row)

    return jax.lax.fori_loop(0, max_len, body, arr)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def bench(label: str, fn, *args, warmup: int = 3, repeats: int = 7):
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
    print(f"    all runs: {[f'{t:.4f}' for t in times]}")
    return result, median


def main():
    print("=" * 60)
    print("Benchmark: index gather vs fori_loop pack/unpack")
    print("=" * 60)

    configs = [
        ("T21", 21),
        ("T42", 42),
        ("T85", 85),
    ]

    # --- Part 1: Isolated pack/unpack ---
    print("\n--- Part 1: Isolated pack/unpack round-trip (1000 iterations) ---")
    for label, trunc in configs:
        grid = GaussianGrid(truncation=trunc)
        transform = SpectralTransform(grid)
        max_len = trunc + 1
        n_spectral = grid.n_spectral_coeffs

        coeffs = jnp.ones(n_spectral, dtype=jnp.complex128)
        arr_3d = jnp.ones((max_len, max_len), dtype=jnp.complex128)

        @jax.jit
        def old_roundtrip(c):
            def body(_, c):
                a = _unpack_spectral_old(c, trunc)
                return _pack_spectral_old(a, trunc)
            return jax.lax.fori_loop(0, 1000, body, c)

        @jax.jit
        def new_roundtrip(c):
            pack_idx = transform._pack_indices
            unpack_idx = transform._unpack_indices
            def body(_, c):
                padded = jnp.append(c, jnp.zeros(1, dtype=c.dtype))
                a = padded[unpack_idx]
                return a.ravel()[pack_idx]
            return jax.lax.fori_loop(0, 1000, body, c)

        print(f"\n  {label} (n_spectral={n_spectral}):")
        res_old, t_old = bench("    fori_loop", old_roundtrip, coeffs, warmup=2, repeats=5)
        res_new, t_new = bench("    index gather", new_roundtrip, coeffs, warmup=2, repeats=5)

        diff = float(jnp.max(jnp.abs(res_old - res_new)))
        speedup = t_old / t_new if t_new > 0 else float("inf")
        print(f"    max diff: {diff:.2e}")
        print(f"    speedup: {speedup:.2f}x")

    # --- Part 2: Full shallow water integration ---
    print("\n--- Part 2: Full shallow water integration (72 steps) ---")
    step_configs = [
        ("T21", 21, 72, 1200.0),
        ("T42", 42, 72, 600.0),
    ]

    for label, trunc, n_steps, dt in step_configs:
        print(f"\n  {label} | {n_steps} steps | dt={dt}s:")
        grid = GaussianGrid(truncation=trunc)
        transform = SpectralTransform(grid)
        initial = williamson2_initial_state(grid, transform)

        _, median = bench(
            "    index gather (current)",
            run_shallow_water, initial, transform, n_steps, dt
        )
        print(f"    per-step: {median / n_steps * 1000:.2f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
