"""Analytic initial conditions for dynamical core validation.

Implements the Jablonowski & Williamson (2006) baroclinic instability
test case and the Held-Suarez (1994) isothermal rest state.

References
----------
Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
test case for atmospheric model dynamical cores. Quarterly Journal of
the Royal Meteorological Society, 132(621C), 2943-2975.

Held, I. M. & Suarez, M. J. (1994). A proposal for the intercomparison
of the dynamical cores of atmospheric general circulation models.
BAMS 75(10), 1825-1830.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.physics.moisture import saturation_specific_humidity
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


@dataclass(frozen=True, slots=True)
class JWConfig:
    """Parameters for the Jablonowski-Williamson test case.

    Attributes
    ----------
    u0 : float
        Zonal jet strength [m/s].
    t0 : float
        Horizontal mean temperature at the surface [K].
    delta_t : float
        Empirical temperature difference [K].
    gamma : float
        Temperature lapse rate [K/m].
    eta_tropo : float
        Tropopause level in eta (sigma) coordinates.
    eta0 : float
        Eta constant for vertical profile.
    """

    u0: float = 35.0
    t0: float = 288.0
    delta_t: float = 4.8e5
    gamma: float = 0.005
    eta_tropo: float = 0.2
    eta0: float = 0.252


def _reference_temperature(
    eta: float,
    planet: PlanetaryConstants,
    config: JWConfig,
) -> float:
    """Compute reference temperature T_ref at a single eta level."""
    exponent = planet.gas_constant * config.gamma / planet.gravity
    t_mean = config.t0 * eta**exponent
    if eta < config.eta_tropo:
        return t_mean + config.delta_t * (config.eta_tropo - eta) ** 5
    return t_mean


def _reference_geopotential(
    eta: float,
    planet: PlanetaryConstants,
    config: JWConfig,
) -> float:
    """Compute reference geopotential Phi_ref at a single eta level."""
    exponent = planet.gas_constant * config.gamma / planet.gravity
    phi_mean = (config.t0 * planet.gravity / config.gamma) * (1.0 - eta**exponent)
    if eta < config.eta_tropo:
        et = config.eta_tropo
        return phi_mean - planet.gas_constant * config.delta_t * (
            (np.log(eta / et) + 137.0 / 60.0) * et**5
            - 5.0 * eta * et**4
            + 5.0 * eta**2 * et**3
            - (10.0 / 3.0) * et**2 * eta**3
            + (5.0 / 4.0) * et * eta**4
            - eta**5 / 5.0
        )
    return phi_mean


def jablonowski_williamson_steady_state(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    config: JWConfig | None = None,
) -> tuple[PrimitiveEquationState, np.ndarray, jnp.ndarray]:
    """Construct the J-W steady-state initial condition.

    This balanced state should remain stationary for up to 30 simulation
    days when integrated with sufficient precision (float64).

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    config : JWConfig or None
        J-W parameters.  None uses defaults from the paper.

    Returns
    -------
    state : PrimitiveEquationState
        Balanced initial state in spectral space.
    reference_temperatures : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential g·z_s in spectral space, shape ``(n_spectral,)``.
    """
    if config is None:
        config = JWConfig()

    grid = transform.grid
    a = planet.radius
    omega = planet.rotation_rate
    r_gas = planet.gas_constant

    etas = np.array(levels.sigma_full)
    lat = np.array(grid.latitudes)  # shape (n_lat,)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    # -- Reference profiles (1D, per level) --
    reference_temperatures = np.array([
        _reference_temperature(float(eta), planet, config) for eta in etas
    ])

    # -- Latitude-dependent factors (shared by geopotential & temperature) --
    # A(lat) = -2 sin^6(lat) (cos^2(lat) + 1/3) + 10/63
    a_lat = -2.0 * sin_lat**6 * (cos_lat**2 + 1.0 / 3.0) + 10.0 / 63.0
    # B(lat) = 1.6 cos^3(lat) (sin^2(lat) + 2/3) - pi/4
    b_lat = 1.6 * cos_lat**3 * (sin_lat**2 + 2.0 / 3.0) - np.pi / 4.0

    # -- Per-level grid fields --
    n_levels = levels.n_levels
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    vorticity_grid = np.zeros((n_levels, n_lat, n_lon))
    temperature_grid = np.zeros((n_levels, n_lat, n_lon))

    for k, eta in enumerate(etas):
        eta_nu = (float(eta) - config.eta0) * np.pi / 2.0
        cos_enu = np.cos(eta_nu)
        sin_enu = np.sin(eta_nu)

        # Vorticity: ζ(lat, η) = (-4u0/a) cos^1.5(η_ν) sin(lat) cos(lat) (2 - 5sin²(lat))
        vort = (-4.0 * config.u0 / a) * cos_enu**1.5 * sin_lat * cos_lat * (2.0 - 5.0 * sin_lat**2)
        vorticity_grid[k, :, :] = vort[:, None]

        # Temperature variation:
        # T'(lat,η) = 0.75·(η·π·u0/R)·sin(η_ν)·√cos(η_ν) ·
        #   [A(lat)·2u0·cos^1.5(η_ν) + B(lat)·a·Ω]
        t_var = (
            0.75
            * (float(eta) * np.pi * config.u0 / r_gas)
            * sin_enu
            * np.sqrt(cos_enu)
            * (a_lat * 2.0 * config.u0 * cos_enu**1.5 + b_lat * a * omega)
        )
        # Absolute temperature = T_ref + T'
        temperature_grid[k, :, :] = (reference_temperatures[k] + t_var)[:, None]

    # -- Surface geopotential: Φ_s = Φ(lat, η=1) --
    eta_s = 1.0
    eta_nu_s = (eta_s - config.eta0) * np.pi / 2.0
    cos_enu_s = np.cos(eta_nu_s)
    phi_ref_s = _reference_geopotential(eta_s, planet, config)
    phi_s = phi_ref_s + config.u0 * cos_enu_s**1.5 * (
        a_lat * config.u0 * cos_enu_s**1.5 + b_lat * a * omega
    )
    # Broadcast to (n_lat, n_lon) for transform
    phi_s_grid = np.broadcast_to(phi_s[:, None], (n_lat, n_lon))

    # -- Transform to spectral space (batched) --
    n_spec = grid.n_spectral_coeffs
    all_grid = jnp.array(np.concatenate([vorticity_grid, temperature_grid], axis=0))
    all_spec = jax.vmap(transform.grid_to_spectral)(all_grid)
    vorticity_spec = np.asarray(all_spec[:n_levels])
    temperature_spec = np.asarray(all_spec[n_levels:])

    surface_geopotential = transform.grid_to_spectral(jnp.array(phi_s_grid))

    # Surface pressure is uniform p0, so ln(ps/p0) = 0
    log_surface_pressure = jnp.zeros(n_spec, dtype=jnp.complex128)

    state = PrimitiveEquationState(
        vorticity=jnp.array(vorticity_spec),
        divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        temperature=jnp.array(temperature_spec),
        log_surface_pressure=log_surface_pressure,
    )

    return state, reference_temperatures, surface_geopotential


def jablonowski_williamson_perturbation(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    u_perturb: float = 1.0,
    lon_location: float = np.pi / 9.0,
    lat_location: float = 2.0 * np.pi / 9.0,
    perturbation_radius: float = 0.1,
) -> PrimitiveEquationState:
    """Construct a localized perturbation to trigger baroclinic instability.

    Added to the steady state from :func:`jablonowski_williamson_steady_state`,
    this perturbation initiates growing baroclinic waves.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    u_perturb : float
        Velocity scale of the perturbation [m/s].
    lon_location : float
        Longitude centre of the perturbation [rad].
    lat_location : float
        Latitude centre of the perturbation [rad].
    perturbation_radius : float
        Ratio of perturbation spatial scale to planet radius.

    Returns
    -------
    PrimitiveEquationState
        Perturbation state (add to steady state to get perturbed IC).
    """
    grid = transform.grid
    a = planet.radius
    big_r = a * perturbation_radius

    lat = np.array(grid.latitudes)  # (n_lat,)
    lon = np.array(grid.longitudes)  # (n_lon,)
    sin_lat = np.sin(lat)[:, None]  # (n_lat, 1) — broadcasts with lon
    cos_lat = np.cos(lat)[:, None]
    lon_bc = lon[None, :]  # (1, n_lon)

    # Great-circle distance factor
    x = np.sin(lat_location) * sin_lat + np.cos(lat_location) * cos_lat * np.cos(
        lon_bc - lon_location
    )
    r = a * np.arccos(np.clip(x, -1.0, 1.0))
    sqrt_val = np.sqrt(np.maximum(1.0 - x**2, 1e-12))
    arccos_x = np.arccos(np.clip(x, -1.0, 1.0))

    # Vorticity perturbation
    exp_decay = np.exp(-((r / big_r) ** 2))
    vort_pert = (
        (u_perturb / a)
        * exp_decay
        * (
            np.tan(lat[:, None])
            - 2.0
            * (a / big_r) ** 2
            * arccos_x
            * (
                np.sin(lat_location) * cos_lat
                - np.cos(lat_location) * sin_lat * np.cos(lon_bc - lon_location)
            )
            / sqrt_val
        )
    )

    # Divergence perturbation
    div_pert = (
        (-2.0 * u_perturb * a / big_r**2)
        * exp_decay
        * arccos_x
        * (np.cos(lat_location) * np.sin(lon_bc - lon_location))
        / sqrt_val
    )

    # Transform once and broadcast (perturbation is level-independent)
    n_levels = levels.n_levels
    n_spec = grid.n_spectral_coeffs
    vort_spec_single = transform.grid_to_spectral(jnp.array(vort_pert))
    div_spec_single = transform.grid_to_spectral(jnp.array(div_pert))
    vort_spec = jnp.tile(vort_spec_single[None, :], (n_levels, 1))
    div_spec = jnp.tile(div_spec_single[None, :], (n_levels, 1))

    return PrimitiveEquationState(
        vorticity=vort_spec,
        divergence=div_spec,
        temperature=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
    )


def held_suarez_initial_state(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    *,
    initial_temperature: float = 264.0,
    perturbation_amplitude: float = 1.0,
    seed: int = 0,
) -> tuple[PrimitiveEquationState, np.ndarray, jnp.ndarray]:
    """Construct an isothermal rest-state initial condition for Held-Suarez.

    The atmosphere starts at rest with a uniform temperature profile and
    a small random temperature perturbation to break hemispheric symmetry.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    initial_temperature : float
        Uniform initial temperature [K].
    perturbation_amplitude : float
        Amplitude of random temperature perturbation [K].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    state : PrimitiveEquationState
        Initial state in spectral space.
    reference_temperatures : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential (zero for flat surface), shape ``(n_spectral,)``.
    """
    grid = transform.grid
    n_levels = levels.n_levels
    n_spec = grid.n_spectral_coeffs
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Uniform reference temperature
    reference_temperatures = np.full(n_levels, initial_temperature)

    # Flat surface
    surface_geopotential = jnp.zeros(n_spec, dtype=jnp.complex128)

    # Uniform temperature in spectral space + small random perturbation
    key = jax.random.PRNGKey(seed)
    t_pert_grid = perturbation_amplitude * jax.random.normal(
        key, (n_levels, n_lat, n_lon), dtype=jnp.float64
    )
    t_uniform_grid = jnp.full((n_levels, n_lat, n_lon), initial_temperature)
    t_grid = t_uniform_grid + t_pert_grid
    t_spec = jax.vmap(transform.grid_to_spectral)(t_grid)

    state = PrimitiveEquationState(
        vorticity=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        divergence=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        temperature=t_spec,
        log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
    )

    return state, reference_temperatures, surface_geopotential


def simple_physics_initial_state(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    *,
    initial_temperature: float = 264.0,
    perturbation_amplitude: float = 1.0,
    seed: int = 0,
) -> tuple[PrimitiveEquationState, np.ndarray, jnp.ndarray]:
    """Construct an isothermal rest-state initial condition for simple physics.

    Identical to :func:`held_suarez_initial_state` (flat surface, isothermal
    with random perturbation).  Provided as a named entry point for clarity.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    initial_temperature : float
        Uniform initial temperature [K].
    perturbation_amplitude : float
        Amplitude of random temperature perturbation [K].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    state : PrimitiveEquationState
        Initial state in spectral space.
    reference_temperatures : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential (zero for flat surface), shape ``(n_spectral,)``.
    """
    return held_suarez_initial_state(
        transform,
        planet,
        levels,
        initial_temperature=initial_temperature,
        perturbation_amplitude=perturbation_amplitude,
        seed=seed,
    )


def moist_aquaplanet_initial_state(
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    *,
    initial_temperature: float = 264.0,
    perturbation_amplitude: float = 1.0,
    initial_rh: float = 0.7,
    rh_stratosphere: float = 0.0,
    sigma_tropopause: float = 0.3,
    seed: int = 0,
) -> tuple[PrimitiveEquationState, np.ndarray, jnp.ndarray]:
    """Construct an initial condition with humidity for moist aquaplanet.

    Starts from the dry simple-physics initial state and adds a specific
    humidity profile based on a prescribed relative humidity that decays
    above the tropopause.

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform.
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    initial_temperature : float
        Uniform initial temperature [K].
    perturbation_amplitude : float
        Amplitude of random temperature perturbation [K].
    initial_rh : float
        Tropospheric relative humidity (0-1).
    rh_stratosphere : float
        Stratospheric relative humidity (0-1).
    sigma_tropopause : float
        Sigma level of the tropopause (RH transitions above this).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    state : PrimitiveEquationState
        Initial state in spectral space (with humidity).
    reference_temperatures : np.ndarray
        Reference temperature profile, shape ``(n_levels,)``.
    surface_geopotential : jnp.ndarray
        Surface geopotential, shape ``(n_spectral,)``.
    """
    state, ref_temps, surf_geo = simple_physics_initial_state(
        transform, planet, levels,
        initial_temperature=initial_temperature,
        perturbation_amplitude=perturbation_amplitude,
        seed=seed,
    )

    # Build RH profile: initial_rh in troposphere, decaying above
    sigma_full = np.asarray(levels.sigma_full)
    rh_profile = np.where(
        sigma_full > sigma_tropopause,
        initial_rh,
        rh_stratosphere + (initial_rh - rh_stratosphere)
        * (sigma_full / sigma_tropopause),
    )

    # Compute q = RH * q_sat(T, p) at each level, capped at the
    # surface value to avoid unphysically large q at low pressures
    # (where q_sat diverges for isothermal atmospheres).
    p_ref = planet.reference_pressure
    p_levels = sigma_full * p_ref
    q_sat_profile = saturation_specific_humidity(
        jnp.array(np.full_like(sigma_full, initial_temperature)),
        jnp.array(p_levels),
        planet.epsilon_moisture,
    )
    q_sat_surface = float(q_sat_profile[-1])
    q_sat_capped = np.minimum(np.asarray(q_sat_profile), q_sat_surface)
    q_profile = np.asarray(rh_profile) * q_sat_capped

    # Put the horizontally-uniform q profile into spectral space
    # Mode (0,0) coefficient = value * sqrt(4π)
    n_spec = transform.grid.n_spectral_coeffs
    n_levels = levels.n_levels
    sqrt4pi = np.sqrt(4.0 * np.pi)
    q_spec = jnp.zeros((n_levels, n_spec), dtype=jnp.complex128)
    q_spec = q_spec.at[:, 0].set(
        jnp.array(q_profile * sqrt4pi, dtype=jnp.complex128)
    )

    moist_state = state.replace(humidity=q_spec)
    return moist_state, ref_temps, surf_geo
