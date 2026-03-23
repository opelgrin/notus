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

from notus.constants import PlanetaryConstants
from notus.operators import (
    hyperdiffusion,
    laplacian,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
)
from notus.state import ShallowWaterState
from notus.transforms import SpectralTransform


def shallow_water_tendencies(
    state: ShallowWaterState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    diffusion_order: int = 4,
    diffusion_timescale: float = 2.0 * 3600.0,
) -> ShallowWaterState:
    """Compute explicit tendencies for the shallow water equations.

    All nonlinear products are evaluated on the grid, then transformed to
    spectral space.  Includes biharmonic hyperdiffusion for stability.

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
    t = grid.truncation
    a = planet.radius

    # --- Step 1: Reconstruct cosine-weighted winds (spectral) ---
    u_cos_spec, v_cos_spec = uv_from_vordiv(
        state.vorticity, state.divergence, t, a
    )

    # --- Step 2: Transform to grid ---
    vort_grid = transform.spectral_to_grid(state.vorticity)
    u_cos_grid = transform.spectral_to_grid(u_cos_spec)
    v_cos_grid = transform.spectral_to_grid(v_cos_spec)
    phi_grid = transform.spectral_to_grid(state.geopotential)

    # --- Step 3: Grid-point computations ---
    # Coriolis parameter: f = 2Ω·sin(φ)
    f_coriolis = 2.0 * planet.rotation_rate * grid.sin_lat[:, None]

    # Absolute vorticity
    zeta_a = vort_grid + f_coriolis

    # 1/cos²(φ) factor — safe because Gaussian grid never touches poles
    cos2_inv = 1.0 / (grid.cos_lat[:, None] ** 2)

    # Flux terms divided by cos(φ) for spectral curl/divergence:
    #   A = ζ_a·u/cos(φ) = ζ_a·U/cos²(φ)
    #   B = ζ_a·v/cos(φ) = ζ_a·V/cos²(φ)
    flux_a = zeta_a * u_cos_grid * cos2_inv
    flux_b = zeta_a * v_cos_grid * cos2_inv

    # Geopotential flux terms:
    #   C = Φ·u/cos(φ) = Φ·U/cos²(φ)
    #   D = Φ·v/cos(φ) = Φ·V/cos²(φ)
    phi_flux_a = phi_grid * u_cos_grid * cos2_inv
    phi_flux_b = phi_grid * v_cos_grid * cos2_inv

    # Kinetic energy: E = (u² + v²)/2 = (U² + V²)/(2·cos²(φ))
    kinetic_energy = 0.5 * (u_cos_grid**2 + v_cos_grid**2) * cos2_inv

    # --- Step 4: Transform products to spectral ---
    flux_a_spec = transform.grid_to_spectral(flux_a)
    flux_b_spec = transform.grid_to_spectral(flux_b)
    phi_flux_a_spec = transform.grid_to_spectral(phi_flux_a)
    phi_flux_b_spec = transform.grid_to_spectral(phi_flux_b)
    ke_spec = transform.grid_to_spectral(kinetic_energy)

    # --- Step 5: Assemble spectral tendencies ---
    # The spectral tendency formulas (Hoskins & Simmons 1975):
    #   dζ/dt = -div(ζ_a·v⃗)
    #   dδ/dt = +curl(ζ_a·v⃗) - ∇²(Φ+E)
    #   dΦ/dt = -div(Φ·v⃗)
    #
    # The EXPLICIT tendencies exclude the linear gravity-wave coupling
    # (-∇²Φ for divergence, -Φ₀δ for geopotential) which is handled
    # implicitly by the semi-implicit time stepper.  Only -∇²(KE) remains.
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
