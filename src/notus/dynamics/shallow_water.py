"""Shallow water equation tendencies on the sphere.

Solves the shallow water equations in vorticity-divergence form using the
pseudo-spectral method: nonlinear products are computed on the Gaussian
grid, then transformed to spectral space for tendency assembly.

The equations:

    dζ/dt = -div(ζ_a · v⃗)                    (vorticity)
    dδ/dt = +curl(ζ_a · v⃗) - ∇²(Φ+E)        (divergence)
    dΦ/dt = -div(Φ · v⃗)                       (continuity)

where ζ_a = ζ + f is absolute vorticity, Φ = g·h is geopotential,
and E = (u² + v²)/2 is kinetic energy.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from notus.constants import PlanetaryConstants
from notus.operators import (
    _laplacian_eigenvalues,
    _m_index_array,
    _meridional_coupling,
    _mu_derivative_coupling,
    hyperdiffusion,
    laplacian,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
)
from notus.state import ShallowWaterState
from notus.transforms import SpectralTransform


def shallow_water_tendencies(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
) -> Callable[[ShallowWaterState], ShallowWaterState]:
    """Build a JIT-compiled explicit tendency function for the shallow water equations.

    All nonlinear products are evaluated on the Gaussian grid, then
    transformed to spectral space for tendency assembly.  Includes
    biharmonic hyperdiffusion for stability.

    The transform arrays and planet constants are closed over so that
    ``jax.jit`` sees only the ``ShallowWaterState`` pytree as input.
    This lets XLA fuse all spectral transforms, grid-point products,
    and operator evaluations into a single compiled kernel.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform (arrays captured by closure).
    planet : PlanetaryConstants
        Planetary constants (radius, rotation_rate; captured by closure).
    diffusion_order : int
        Order of hyperdiffusion (4 = del-8, default). Set to 0 to disable.
    diffusion_timescale : float
        E-folding damping time for the smallest resolved scale [s].
        Default: 2 hours.

    Returns
    -------
    Callable[[ShallowWaterState], ShallowWaterState]
        JIT-compiled function mapping state to tendencies
        dζ/dt, dδ/dt, dΦ/dt in spectral space.
    """
    grid = transform.grid
    t = grid.truncation
    a = planet.radius
    rotation_rate = planet.rotation_rate
    sin_lat = grid.sin_lat
    cos_lat = grid.cos_lat

    # Pre-fetch cached operator coefficients at build time so that
    # the JIT-compiled function captures only JAX arrays, not lru_cache lookups.
    _laplacian_eigenvalues(t, a)
    _m_index_array(t)
    _meridional_coupling(t)
    _mu_derivative_coupling(t)

    @jax.jit
    def tendency(state: ShallowWaterState) -> ShallowWaterState:
        return _tendency_impl(
            state, transform, t, a, rotation_rate, sin_lat, cos_lat,
            diffusion_order, diffusion_timescale,
        )

    return tendency


def shallow_water_tendencies_eager(
    state: ShallowWaterState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
) -> ShallowWaterState:
    """Compute explicit tendencies for the shallow water equations (eager).

    Convenience wrapper that evaluates tendencies immediately without
    JIT compilation.  Useful for debugging, single-shot diagnostics,
    or when the caller manages JIT boundaries externally.

    For time-stepping loops, prefer :func:`shallow_water_tendencies`
    which returns a JIT-compiled callable.

    Parameters
    ----------
    state : ShallowWaterState
        Current state (spectral: vorticity, divergence, geopotential).
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (radius, rotation_rate).
    diffusion_order : int
        Order of hyperdiffusion (4 = del-8, default). Set to 0 to disable.
    diffusion_timescale : float
        E-folding damping time for the smallest resolved scale [s].
        Default: 2 hours.

    Returns
    -------
    ShallowWaterState
        Tendencies dζ/dt, dδ/dt, dΦ/dt in spectral space.
    """
    grid = transform.grid
    return _tendency_impl(
        state, transform,
        grid.truncation, planet.radius, planet.rotation_rate,
        grid.sin_lat, grid.cos_lat,
        diffusion_order, diffusion_timescale,
    )


def _tendency_impl(
    state: ShallowWaterState,
    transform: SpectralTransform,
    t: int,
    a: float,
    rotation_rate: float,
    sin_lat: jnp.ndarray,
    cos_lat: jnp.ndarray,
    diffusion_order: int,
    diffusion_timescale: float,
) -> ShallowWaterState:
    """Core tendency computation shared by JIT and eager paths."""
    # --- Step 1: Reconstruct cosine-weighted winds (spectral) ---
    u_cos_spec, v_cos_spec = uv_from_vordiv(
        state.vorticity, state.divergence, t, a
    )

    # --- Step 2: Transform to grid (batched) ---
    fields_spec = jnp.stack(
        [state.vorticity, u_cos_spec, v_cos_spec, state.geopotential]
    )
    fields_grid = jax.vmap(transform.spectral_to_grid)(fields_spec)
    vort_grid, u_cos_grid, v_cos_grid, phi_grid = fields_grid

    # --- Step 3: Grid-point computations ---
    f_coriolis = 2.0 * rotation_rate * sin_lat[:, None]
    zeta_a = vort_grid + f_coriolis
    cos2_inv = 1.0 / (cos_lat[:, None] ** 2)

    flux_a = zeta_a * u_cos_grid * cos2_inv
    flux_b = zeta_a * v_cos_grid * cos2_inv
    phi_flux_a = phi_grid * u_cos_grid * cos2_inv
    phi_flux_b = phi_grid * v_cos_grid * cos2_inv
    kinetic_energy = 0.5 * (u_cos_grid**2 + v_cos_grid**2) * cos2_inv

    # --- Step 4: Transform products to spectral (batched) ---
    products_grid = jnp.stack(
        [flux_a, flux_b, phi_flux_a, phi_flux_b, kinetic_energy]
    )
    products_spec = jax.vmap(transform.grid_to_spectral)(products_grid)
    flux_a_spec, flux_b_spec = products_spec[0], products_spec[1]
    phi_flux_a_spec, phi_flux_b_spec = products_spec[2], products_spec[3]
    ke_spec = products_spec[4]

    # --- Step 5: Assemble spectral tendencies ---
    vort_tend = -spectral_divergence(flux_a_spec, flux_b_spec, t, a)
    div_tend = -spectral_curl(flux_a_spec, flux_b_spec, t, a) - laplacian(
        ke_spec, t, a
    )
    phi_tend = -spectral_divergence(phi_flux_a_spec, phi_flux_b_spec, t, a)

    # --- Step 6: Hyperdiffusion for numerical stability ---
    if diffusion_order > 0:
        vort_tend += hyperdiffusion(
            state.vorticity, t, a, diffusion_order, diffusion_timescale
        )
        div_tend += hyperdiffusion(
            state.divergence, t, a, diffusion_order, diffusion_timescale
        )

    return ShallowWaterState(
        vorticity=vort_tend,
        divergence=div_tend,
        geopotential=phi_tend,
    )
