"""Analytic initial conditions for dynamical core validation.

Implements the Jablonowski & Williamson (2006) baroclinic instability
test case — the standard 3D dycore validation benchmark.

Reference
---------
Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
test case for atmospheric model dynamical cores. Quarterly Journal of
the Royal Meteorological Society, 132(621C), 2943-2975.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.sigma import SigmaLevels
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform


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
    reference_temperatures = np.array(
        [_reference_temperature(float(eta), planet, config) for eta in etas]
    )

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
        vort = (
            (-4.0 * config.u0 / a)
            * cos_enu**1.5
            * sin_lat
            * cos_lat
            * (2.0 - 5.0 * sin_lat**2)
        )
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

    # -- Transform to spectral space --
    n_spec = grid.n_spectral_coeffs
    vorticity_spec = np.zeros((n_levels, n_spec), dtype=np.complex128)
    temperature_spec = np.zeros((n_levels, n_spec), dtype=np.complex128)

    for k in range(n_levels):
        vorticity_spec[k] = np.asarray(
            transform.grid_to_spectral(jnp.array(vorticity_grid[k]))
        )
        temperature_spec[k] = np.asarray(
            transform.grid_to_spectral(jnp.array(temperature_grid[k]))
        )

    surface_geopotential = jnp.array(
        np.asarray(transform.grid_to_spectral(jnp.array(phi_s_grid)))
    )

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
    sin_lat_1d = np.sin(lat)
    cos_lat_1d = np.cos(lat)

    # 2D meshes: (n_lat, n_lon)
    lat_2d = lat[:, None] * np.ones(grid.n_lon)[None, :]
    lon_2d = np.ones(grid.n_lat)[:, None] * lon[None, :]
    sin_lat_2d = sin_lat_1d[:, None] * np.ones(grid.n_lon)[None, :]
    cos_lat_2d = cos_lat_1d[:, None] * np.ones(grid.n_lon)[None, :]

    # Great-circle distance factor
    x = (
        np.sin(lat_location) * sin_lat_2d
        + np.cos(lat_location) * cos_lat_2d * np.cos(lon_2d - lon_location)
    )
    r = a * np.arccos(np.clip(x, -1.0, 1.0))
    sqrt_val = np.sqrt(np.maximum(1.0 - x**2, 1e-12))
    arccos_x = np.arccos(np.clip(x, -1.0, 1.0))

    # Vorticity perturbation
    exp_decay = np.exp(-(r / big_r) ** 2)
    vort_pert = (u_perturb / a) * exp_decay * (
        np.tan(lat_2d)
        - 2.0 * (a / big_r) ** 2 * arccos_x
        * (
            np.sin(lat_location) * cos_lat_2d
            - np.cos(lat_location) * sin_lat_2d * np.cos(lon_2d - lon_location)
        )
        / sqrt_val
    )

    # Divergence perturbation
    div_pert = (
        -2.0 * u_perturb * a / big_r**2
    ) * exp_decay * arccos_x * (
        np.cos(lat_location) * np.sin(lon_2d - lon_location)
    ) / sqrt_val

    # Stack to all levels (perturbation is level-independent)
    n_levels = levels.n_levels
    n_spec = grid.n_spectral_coeffs
    vort_spec = np.zeros((n_levels, n_spec), dtype=np.complex128)
    div_spec = np.zeros((n_levels, n_spec), dtype=np.complex128)

    for k in range(n_levels):
        vort_spec[k] = np.asarray(
            transform.grid_to_spectral(jnp.array(vort_pert))
        )
        div_spec[k] = np.asarray(
            transform.grid_to_spectral(jnp.array(div_pert))
        )

    return PrimitiveEquationState(
        vorticity=jnp.array(vort_spec),
        divergence=jnp.array(div_spec),
        temperature=jnp.zeros((n_levels, n_spec), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros(n_spec, dtype=jnp.complex128),
    )
