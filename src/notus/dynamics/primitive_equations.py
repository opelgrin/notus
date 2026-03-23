"""Primitive equation explicit tendencies on the sphere.

Computes the explicit part of the hydrostatic primitive equations in
vorticity-divergence form on sigma levels, extending the shallow water
formulation with vertical advection, adiabatic heating, and surface
pressure coupling.

The equations (explicit tendencies only):

    dζ/dt = -curl(F) + diffusion
    dδ/dt = -div(F) - ∇²E - ∇²(g·z_s) + diffusion
    dT/dt = -div(T·v⃗) + vertical_advection + κ·T·(ω/p) + diffusion
    d(ln ps)/dt = -Σ (v⃗·∇ln ps)·Δσ

where F = (ζ+f)(k̂×v) + σ̇·∂v/∂σ + R·T'·∇ln(ps), the combined momentum
flux including absolute vorticity, vertical advection of momentum, and
the explicit part of the pressure gradient.

The implicit terms (-∇²(Φ + R·T_ref·ln ps), -H·δ, -Σ δ·Δσ) are NOT
included here; they are handled by the semi-implicit time stepper.
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
    meridional_derivative,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
    zonal_derivative,
)
from notus.sigma import SigmaLevels
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical import (
    omega_over_pressure,
    sigma_dot,
    surface_pressure_tendency,
    vertical_advection,
)


def primitive_equation_tendencies(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    reference_temperature: jnp.ndarray,
    surface_geopotential: jnp.ndarray,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
) -> Callable[[PrimitiveEquationState], PrimitiveEquationState]:
    """Build a JIT-compiled explicit tendency function for the primitive equations.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants (captured by closure).
    levels : SigmaLevels
        Sigma vertical coordinate.
    reference_temperature : jnp.ndarray
        Reference temperature profile T_ref, shape ``(n_levels,)``.
        Used to split T = T_ref + T' for the semi-implicit scheme.
    surface_geopotential : jnp.ndarray
        Surface geopotential g·z_s in spectral space, shape ``(n_spectral,)``.
    diffusion_order : int
        Order of hyperdiffusion (4 = del-8). Set to 0 to disable.
    diffusion_timescale : float
        E-folding damping time for the smallest resolved scale [s].

    Returns
    -------
    Callable[[PrimitiveEquationState], PrimitiveEquationState]
        JIT-compiled function mapping state to explicit tendencies.
    """
    grid = transform.grid
    t = grid.truncation
    a = planet.radius
    rotation_rate = planet.rotation_rate
    gas_constant = planet.gas_constant
    kappa = planet.kappa
    sin_lat = grid.sin_lat
    cos_lat = grid.cos_lat

    # Pre-fetch cached operator coefficients
    _laplacian_eigenvalues(t, a)
    _m_index_array(t)
    _meridional_coupling(t)
    _mu_derivative_coupling(t)

    # Pre-compute the orography tendency (constant in time)
    orography_tend = -laplacian(surface_geopotential, t, a)

    # Reference temperature for broadcasting: (n_levels, 1, 1)
    t_ref_grid = jnp.asarray(reference_temperature)

    @jax.jit
    def tendency(state: PrimitiveEquationState) -> PrimitiveEquationState:
        return _tendency_impl(
            state,
            transform,
            levels,
            t,
            a,
            rotation_rate,
            gas_constant,
            kappa,
            sin_lat,
            cos_lat,
            t_ref_grid,
            orography_tend,
            diffusion_order,
            diffusion_timescale,
        )

    return tendency


def _tendency_impl(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    levels: SigmaLevels,
    t: int,
    a: float,
    rotation_rate: float,
    gas_constant: float,
    kappa: float,
    sin_lat: jnp.ndarray,
    cos_lat: jnp.ndarray,
    t_ref: jnp.ndarray,
    orography_tend: jnp.ndarray,
    diffusion_order: int,
    diffusion_timescale: float,
) -> PrimitiveEquationState:
    """Core PE explicit tendency computation."""
    n_levels = state.n_levels

    # Step 1: Reconstruct winds at each level (spectral)
    def _uv_at_level(vort_div: jnp.ndarray) -> jnp.ndarray:
        vort, div = vort_div[0], vort_div[1]
        u, v = uv_from_vordiv(vort, div, t, a)
        return jnp.stack([u, v])

    # Shape: (n_levels, 2, n_spectral) — vmap maps over the level axis
    vort_div_stacked = jnp.stack([state.vorticity, state.divergence], axis=1)
    uv_spec = jax.vmap(_uv_at_level)(vort_div_stacked)
    u_cos_spec = uv_spec[:, 0, :]
    v_cos_spec = uv_spec[:, 1, :]

    # Step 2: Surface pressure gradient in spectral space
    dlnps_dlam_spec = zonal_derivative(state.log_surface_pressure, t)
    cosphi_dlnps_dphi_spec = meridional_derivative(state.log_surface_pressure, t)

    # Step 3: Transform to grid (batched)
    all_spec = jnp.concatenate(
        [
            state.vorticity,
            state.divergence,
            u_cos_spec,
            v_cos_spec,
            state.temperature,
            jnp.stack([dlnps_dlam_spec, cosphi_dlnps_dphi_spec]),
        ],
        axis=0,
    )
    all_grid = jax.vmap(transform.spectral_to_grid)(all_spec)

    vort_grid = all_grid[:n_levels]
    div_grid = all_grid[n_levels : 2 * n_levels]
    u_cos_grid = all_grid[2 * n_levels : 3 * n_levels]
    v_cos_grid = all_grid[3 * n_levels : 4 * n_levels]
    t_grid = all_grid[4 * n_levels : 5 * n_levels]
    dlnps_dlam_grid = all_grid[5 * n_levels]
    cosphi_dlnps_dphi_grid = all_grid[5 * n_levels + 1]

    # Steps 4-6: Grid-point and vertical computations
    products_grid, lnps_tend_grid = _grid_point_tendencies(
        vort_grid,
        div_grid,
        u_cos_grid,
        v_cos_grid,
        t_grid,
        dlnps_dlam_grid,
        cosphi_dlnps_dphi_grid,
        levels,
        t_ref,
        rotation_rate,
        gas_constant,
        kappa,
        sin_lat,
        cos_lat,
    )

    # Step 7: Transform all products to spectral (batched)
    all_products = jnp.concatenate(
        [
            products_grid,
            lnps_tend_grid[None, :, :],
        ],
        axis=0,
    )
    all_products_spec = jax.vmap(transform.grid_to_spectral)(all_products)
    products_spec = all_products_spec[:-1]
    lnps_tend_spec = all_products_spec[-1]

    # Step 8: Assemble spectral tendencies per level
    return _assemble_spectral_tendencies(
        products_spec,
        state,
        lnps_tend_spec,
        orography_tend,
        n_levels,
        t,
        a,
        diffusion_order,
        diffusion_timescale,
    )


def _grid_point_tendencies(
    vort_grid: jnp.ndarray,
    div_grid: jnp.ndarray,
    u_cos_grid: jnp.ndarray,
    v_cos_grid: jnp.ndarray,
    t_grid: jnp.ndarray,
    dlnps_dlam_grid: jnp.ndarray,
    cosphi_dlnps_dphi_grid: jnp.ndarray,
    levels: SigmaLevels,
    t_ref: jnp.ndarray,
    rotation_rate: float,
    gas_constant: float,
    kappa: float,
    sin_lat: jnp.ndarray,
    cos_lat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute grid-point products and vertical tendency terms.

    Returns (products_grid, lnps_tend_grid) where products_grid has shape
    (6*n_levels, n_lat, n_lon) containing combined_u, combined_v, KE,
    T_flux_a, T_flux_b, nodal_temp_tend stacked along axis 0.
    """
    f_coriolis = 2.0 * rotation_rate * sin_lat[None, :, None]
    cos2_inv = 1.0 / (cos_lat[None, :, None] ** 2)
    zeta_a = vort_grid + f_coriolis

    t_ref_bc = t_ref[:, None, None]
    t_prime_grid = t_grid - t_ref_bc

    dlnps_dlam_bc = dlnps_dlam_grid[None, :, :]
    cosphi_dlnps_dphi_bc = cosphi_dlnps_dphi_grid[None, :, :]

    # v⃗·∇ln(ps)
    v_dot_grad_lnps = (u_cos_grid * dlnps_dlam_bc + v_cos_grid * cosphi_dlnps_dphi_bc) * cos2_inv

    # Horizontal products
    flux_a = zeta_a * u_cos_grid * cos2_inv
    flux_b = zeta_a * v_cos_grid * cos2_inv
    kinetic_energy = 0.5 * (u_cos_grid**2 + v_cos_grid**2) * cos2_inv
    t_flux_a = t_prime_grid * u_cos_grid * cos2_inv
    t_flux_b = t_prime_grid * v_cos_grid * cos2_inv

    # Vertical operations
    column_div = div_grid + v_dot_grad_lnps
    sd = sigma_dot(column_div, levels)

    # Vertical advection of temperature, split for semi-implicit scheme:
    # - T' part uses full σ̇ (D-dependent): fully explicit
    # - T_ref part uses σ̇_explicit (D-free): the D-dependent part is implicit
    #   (captured by the K terms in the temperature implicit weights matrix H)
    sd_explicit = sigma_dot(v_dot_grad_lnps, levels)
    t_ref_field = jnp.broadcast_to(t_ref_bc, t_prime_grid.shape)
    vert_adv_temp = vertical_advection(sd, t_prime_grid, levels) + vertical_advection(
        sd_explicit, t_ref_field, levels
    )

    # Adiabatic heating κ·T·(ω/p), split for semi-implicit scheme:
    # - T_ref part uses g_term = v⃗·∇ln(ps) only (D-dependent part is implicit)
    # - T' part uses g_term = D + v⃗·∇ln(ps) (full explicit contribution)
    omega_p_explicit = omega_over_pressure(v_dot_grad_lnps, v_dot_grad_lnps, levels)
    omega_p_full = omega_over_pressure(column_div, v_dot_grad_lnps, levels)
    adiabatic = kappa * (t_ref_bc * omega_p_explicit + t_prime_grid * omega_p_full)

    vert_mom_u = -vertical_advection(sd, u_cos_grid, levels)
    vert_mom_v = -vertical_advection(sd, v_cos_grid, levels)

    rt_grad_u = gas_constant * t_prime_grid * dlnps_dlam_bc
    rt_grad_v = gas_constant * t_prime_grid * cosphi_dlnps_dphi_bc

    # Combined momentum flux (Dinosaur convention for curl/div)
    combined_u = -flux_b + (vert_mom_u + rt_grad_u) * cos2_inv
    combined_v = flux_a + (vert_mom_v + rt_grad_v) * cos2_inv

    lnps_tend_grid = surface_pressure_tendency(v_dot_grad_lnps, levels)
    nodal_temp_tend = vert_adv_temp + adiabatic

    products_grid = jnp.concatenate(
        [
            combined_u,
            combined_v,
            kinetic_energy,
            t_flux_a,
            t_flux_b,
            nodal_temp_tend,
        ],
        axis=0,
    )

    return products_grid, lnps_tend_grid


def _assemble_spectral_tendencies(
    products_spec: jnp.ndarray,
    state: PrimitiveEquationState,
    lnps_tend_spec: jnp.ndarray,
    orography_tend: jnp.ndarray,
    n_levels: int,
    t: int,
    a: float,
    diffusion_order: int,
    diffusion_timescale: float,
) -> PrimitiveEquationState:
    """Assemble spectral tendencies from transformed grid products."""
    combined_u_spec = products_spec[:n_levels]
    combined_v_spec = products_spec[n_levels : 2 * n_levels]
    ke_spec = products_spec[2 * n_levels : 3 * n_levels]
    t_flux_a_spec = products_spec[3 * n_levels : 4 * n_levels]
    t_flux_b_spec = products_spec[4 * n_levels : 5 * n_levels]
    nodal_temp_tend_spec = products_spec[5 * n_levels : 6 * n_levels]

    def _assemble_level(args: jnp.ndarray) -> jnp.ndarray:
        cu, cv, ke, tfa, tfb, nodal_t, vort_k, div_k, temp_k = (
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
        )
        vort_tend = spectral_curl(cu, cv, t, a)
        div_tend = -spectral_divergence(cu, cv, t, a) - laplacian(ke, t, a) + orography_tend
        temp_tend = -spectral_divergence(tfa, tfb, t, a) + nodal_t

        if diffusion_order > 0:
            vort_tend += hyperdiffusion(vort_k, t, a, diffusion_order, diffusion_timescale)
            div_tend += hyperdiffusion(div_k, t, a, diffusion_order, diffusion_timescale)
            temp_tend += hyperdiffusion(temp_k, t, a, diffusion_order, diffusion_timescale)

        return jnp.stack([vort_tend, div_tend, temp_tend])

    level_args = jnp.stack(
        [
            combined_u_spec,
            combined_v_spec,
            ke_spec,
            t_flux_a_spec,
            t_flux_b_spec,
            nodal_temp_tend_spec,
            state.vorticity,
            state.divergence,
            state.temperature,
        ],
        axis=1,
    )

    level_tendencies = jax.vmap(_assemble_level)(level_args)

    return PrimitiveEquationState(
        vorticity=level_tendencies[:, 0, :],
        divergence=level_tendencies[:, 1, :],
        temperature=level_tendencies[:, 2, :],
        log_surface_pressure=lnps_tend_spec,
    )
