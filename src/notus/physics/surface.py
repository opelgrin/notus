"""Surface boundary conditions and fluxes.

Prescribed SST profiles, bulk-aerodynamic sensible and latent heat fluxes,
slab ocean thermodynamics, and bucket land surface model.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from notus.physics.convection import betts_miller_convection, large_scale_condensation
from notus.physics.moisture import saturation_specific_humidity
from notus.physics.radiation import STEFAN_BOLTZMANN


@dataclasses.dataclass(frozen=True, slots=True)
class PrescribedSST:
    """Frierson et al. (2006) aquaplanet SST profile.

    Gaussian profile with a temperature floor::

        T_s(phi) = max(T_min, T_min + T_delta * exp(-0.5 * (phi / phi_w)^2))

    Parameters
    ----------
    t_min : float
        Minimum SST (temperature floor) [K].
    t_delta : float
        Equator-to-floor temperature difference [K].
    phi_w : float
        Latitude width parameter [rad].
    """

    t_min: float = 271.0
    t_delta: float = 29.0
    phi_w: float = 26.0 * jnp.pi / 180.0


def compute_sst(
    config: PrescribedSST,
    latitudes: jnp.ndarray,
) -> jnp.ndarray:
    """Compute SST at each latitude.

    Parameters
    ----------
    config : PrescribedSST
        SST profile parameters.
    latitudes : jnp.ndarray
        Latitude in radians, shape ``(n_lat,)``.

    Returns
    -------
    jnp.ndarray
        SST in Kelvin, shape ``(n_lat,)``. Zonally uniform.
    """
    return config.t_min + config.t_delta * jnp.exp(
        -0.5 * (latitudes / config.phi_w) ** 2,
    )


def surface_sensible_heat_flux(
    t_surface: jnp.ndarray,
    t_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    gravity: float,
    specific_heat_cp: float,
    gas_constant: float,
    dsigma_lowest: float,
    *,
    drag_coefficient: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute surface sensible heat flux tendency for the lowest level.

    Bulk aerodynamic formula::

        H = rho * cp * C_D * |v| * (T_s - T_a)

    converted to a temperature tendency for the lowest model level::

        dT/dt = H / (dp/g) / cp = g * C_D * |v| * (T_s - T_a) / dp

    where dp = dsigma_lowest * ps is the pressure thickness of the
    lowest layer.

    Parameters
    ----------
    t_surface : jnp.ndarray
        Surface temperature, shape ``(n_lat,)``.
    t_air : jnp.ndarray
        Temperature at the lowest model level, shape ``(n_lat, n_lon)``.
    wind_speed : jnp.ndarray
        Wind speed at the lowest model level, shape ``(n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    drag_coefficient : float or jnp.ndarray
        Surface drag coefficient C_D (dimensionless).  Scalar for
        constant drag, or array ``(n_lat, n_lon)`` for spatially
        varying (e.g. Monin-Obukhov).

    Returns
    -------
    jnp.ndarray
        Temperature tendency [K/s] for the lowest model level,
        shape ``(n_lat, n_lon)``.
    """
    dp = dsigma_lowest * surface_pressure  # (n_lat, n_lon)

    # Surface density from ideal gas law: rho = p_sfc / (R * T_a)
    # where p_sfc ~ sigma_lowest * ps.
    # Tendency: dT/dt = g * H / (dp * cp), with H = rho * cp * C_D * |v| * (T_s - T_a).
    sigma_lowest = 1.0 - 0.5 * dsigma_lowest
    t_air_safe = jnp.maximum(t_air, 1.0)
    dp_safe = jnp.maximum(dp, 1.0)
    rho_sfc = surface_pressure * sigma_lowest / (gas_constant * t_air_safe)
    t_sfc_bc = t_surface[:, None] if t_surface.ndim == 1 else t_surface
    flux = rho_sfc * specific_heat_cp * drag_coefficient * wind_speed * (t_sfc_bc - t_air)

    return gravity * flux / (dp_safe * specific_heat_cp)


def surface_latent_heat_flux(
    t_surface: jnp.ndarray,
    q_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    gravity: float,
    gas_constant: float,
    dsigma_lowest: float,
    epsilon: float,
    *,
    drag_coefficient: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute surface evaporation tendency for the lowest level.

    Bulk aerodynamic formula::

        E = rho * C_D * |v| * (q_sat(T_s, p_s) - q_a)

    converted to a specific humidity tendency for the lowest level::

        dq/dt = g * E / dp

    where dp = dsigma_lowest * ps.

    Parameters
    ----------
    t_surface : jnp.ndarray
        Surface temperature, shape ``(n_lat,)``.
    q_air : jnp.ndarray
        Specific humidity at the lowest level, shape ``(n_lat, n_lon)``.
    wind_speed : jnp.ndarray
        Wind speed at the lowest level, shape ``(n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    dsigma_lowest : float
        Sigma thickness of the lowest model level.
    epsilon : float
        Ratio R_d / R_v (≈ 0.622).
    drag_coefficient : float or jnp.ndarray
        Surface drag coefficient C_D (dimensionless).  Scalar or
        array ``(n_lat, n_lon)``.

    Returns
    -------
    jnp.ndarray
        Specific humidity tendency [kg/kg/s] for the lowest level,
        shape ``(n_lat, n_lon)``.
    """
    dp = dsigma_lowest * surface_pressure
    dp_safe = jnp.maximum(dp, 1.0)

    # Surface density
    sigma_lowest = 1.0 - 0.5 * dsigma_lowest
    t_sfc_bc = t_surface[:, None] if t_surface.ndim == 1 else t_surface
    t_air_safe = jnp.maximum(t_sfc_bc, 1.0)
    rho_sfc = surface_pressure * sigma_lowest / (gas_constant * t_air_safe)

    # Saturation specific humidity at the surface
    q_sat_sfc = saturation_specific_humidity(
        t_sfc_bc,
        surface_pressure,
        epsilon,
    )

    # Evaporation flux (positive = upward, moistening the atmosphere)
    evap = rho_sfc * drag_coefficient * wind_speed * (q_sat_sfc - q_air)

    return gravity * evap / dp_safe


# ---------------------------------------------------------------------------
# Slab ocean
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SlabOceanConfig:
    """Configuration for the slab ocean model.

    Parameters
    ----------
    mixed_layer_depth : float
        Mixed-layer depth [m].
    density_water : float
        Sea-water density [kg/m³].
    specific_heat_water : float
        Sea-water specific heat [J/(kg·K)].
    """

    mixed_layer_depth: float = 50.0
    density_water: float = 1025.0
    specific_heat_water: float = 3994.0

    @property
    def heat_capacity(self) -> float:
        """Ocean heat capacity per unit area [J/(m²·K)]."""
        return self.density_water * self.specific_heat_water * self.mixed_layer_depth


@dataclasses.dataclass(frozen=True, slots=True)
class OceanState:
    """State for the slab ocean model.

    Attributes
    ----------
    surface_temperature : jnp.ndarray
        Sea surface temperature [K], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    """

    surface_temperature: jnp.ndarray

    def replace(self, **kwargs: jnp.ndarray) -> OceanState:
        """Return a new state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


def _ocean_flatten(
    state: OceanState,
) -> tuple[tuple[jnp.ndarray, ...], None]:
    return (state.surface_temperature,), None


def _ocean_unflatten(
    _aux: None,
    children: tuple[jnp.ndarray, ...],
) -> OceanState:
    return OceanState(surface_temperature=children[0])


jax.tree_util.register_pytree_node(OceanState, _ocean_flatten, _ocean_unflatten)


# ---------------------------------------------------------------------------
# Bucket land surface
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BucketLandConfig:
    """Configuration for the Frierson (2006) / Manabe (1969) bucket land model.

    Parameters
    ----------
    soil_heat_capacity : float
        Thermal inertia per unit area [J/(m²·K)].  Default 4×10⁶
        represents ~2 m of moist soil (~50× less than a 50 m slab ocean).
    bucket_capacity : float
        Maximum bucket water depth W_max [m].
    bucket_critical_fraction : float
        Fraction of W_max at which evaporation becomes unlimited (β = 1).
    albedo_dry : float
        Land albedo when bucket is empty.
    albedo_wet : float
        Land albedo when bucket is full.
    moisture_dependent_albedo : bool
        If True, albedo varies linearly between dry and wet values
        based on bucket fill fraction.
    """

    soil_heat_capacity: float = 4.0e6
    bucket_capacity: float = 0.15
    bucket_critical_fraction: float = 0.75
    albedo_dry: float = 0.35
    albedo_wet: float = 0.20
    moisture_dependent_albedo: bool = False

    @property
    def w_crit(self) -> float:
        """Critical bucket depth below which evaporation is limited [m]."""
        return self.bucket_critical_fraction * self.bucket_capacity


@dataclasses.dataclass(frozen=True, slots=True)
class LandState:
    """State for the bucket land surface model.

    Attributes
    ----------
    soil_temperature : jnp.ndarray
        Soil / skin temperature [K], shape ``(n_lat, n_lon)``.
    bucket_depth : jnp.ndarray
        Bucket water depth [m], shape ``(n_lat, n_lon)``, in [0, W_max].
    """

    soil_temperature: jnp.ndarray
    bucket_depth: jnp.ndarray

    def replace(self, **kwargs: jnp.ndarray) -> LandState:
        """Return a new state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


def _land_flatten(
    state: LandState,
) -> tuple[tuple[jnp.ndarray, ...], None]:
    return (state.soil_temperature, state.bucket_depth), None


def _land_unflatten(
    _aux: None,
    children: tuple[jnp.ndarray, ...],
) -> LandState:
    return LandState(soil_temperature=children[0], bucket_depth=children[1])


jax.tree_util.register_pytree_node(LandState, _land_flatten, _land_unflatten)


@dataclasses.dataclass(frozen=True, slots=True)
class SurfaceState:
    """Combined ocean + land surface state.

    Wraps :class:`OceanState` and an optional :class:`LandState` so that
    the coupled stepper signature stays at three arguments.

    Attributes
    ----------
    ocean : OceanState
        Slab ocean state (always present).
    land : LandState or None
        Bucket land state, or ``None`` for pure aquaplanet.
    """

    ocean: OceanState
    land: LandState | None = None

    def replace(self, **kwargs: Any) -> SurfaceState:
        """Return a new state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


def _surface_state_flatten(
    state: SurfaceState,
) -> tuple[tuple[Any, ...], None]:
    return (state.ocean, state.land), None


def _surface_state_unflatten(
    _aux: None,
    children: tuple[OceanState, LandState | None],
) -> SurfaceState:
    return SurfaceState(ocean=children[0], land=children[1])


jax.tree_util.register_pytree_node(
    SurfaceState,
    _surface_state_flatten,
    _surface_state_unflatten,
)


def init_land_state(
    land_fraction: jnp.ndarray,
    initial_sst: jnp.ndarray,
    land_config: BucketLandConfig,
) -> LandState:
    """Create initial land state from SST and land mask.

    Parameters
    ----------
    land_fraction : jnp.ndarray
        Land fraction, shape ``(n_lat, n_lon)``.
    initial_sst : jnp.ndarray
        Initial SST [K], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    land_config : BucketLandConfig
        Bucket land configuration.

    Returns
    -------
    LandState
        Land state with soil temperature set to SST (broadcast to 2-D)
        and bucket depth at 75 % of capacity.
    """
    sst_2d = initial_sst[:, None] if initial_sst.ndim == 1 else initial_sst
    sst_2d = jnp.broadcast_to(sst_2d, land_fraction.shape)
    return LandState(
        soil_temperature=sst_2d,
        bucket_depth=jnp.full_like(land_fraction, 0.75 * land_config.bucket_capacity),
    )


def compute_net_surface_flux(
    surface_temperature: jnp.ndarray,
    t_air: jnp.ndarray,
    q_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    sw_down_surface: jnp.ndarray,
    lw_down: jnp.ndarray,
    *,
    gravity: float,
    gas_constant: float,
    specific_heat_cp: float,
    epsilon: float,
    latent_heat: float,
    drag_coefficient: float | jnp.ndarray,
    surface_albedo: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute net downward surface energy flux [W/m²].

    Positive = warming the ocean.

    ``F_net = SW_absorbed + LW_down − σ·T_s⁴ − H − L·E``

    The SW flux reaching the surface (``sw_down_surface``) should come
    from :func:`~notus.physics.radiation.shortwave_heating`, ensuring
    column-consistent energy conservation.

    Parameters
    ----------
    surface_temperature : jnp.ndarray
        SST [K], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    t_air : jnp.ndarray
        Lowest-level air temperature [K], shape ``(n_lat, n_lon)``.
    q_air : jnp.ndarray
        Lowest-level specific humidity [kg/kg], shape ``(n_lat, n_lon)``.
    wind_speed : jnp.ndarray
        Lowest-level wind speed [m/s], shape ``(n_lat, n_lon)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    sw_down_surface : jnp.ndarray
        Downward SW flux at the surface [W/m²], shape ``(n_lat, n_lon)``
        or ``(n_lat,)``.  From :func:`shortwave_heating`.
    lw_down : jnp.ndarray
        Downward LW flux at the surface [W/m²], shape ``(n_lat, n_lon)``.
    gravity : float
        Gravitational acceleration [m/s²].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    epsilon : float
        Ratio R_d / R_v.
    latent_heat : float
        Latent heat of vaporization [J/kg].
    drag_coefficient : float or jnp.ndarray
        Surface drag coefficient C_D.  Scalar or array ``(n_lat, n_lon)``.
    surface_albedo : float or jnp.ndarray
        Surface albedo (0-1).  Scalar or array ``(n_lat, n_lon)``.

    Returns
    -------
    jnp.ndarray
        Net surface flux [W/m²], shape ``(n_lat, n_lon)``.
    """
    # Broadcast SST to (n_lat, n_lon) if needed
    t_s = surface_temperature[:, None] if surface_temperature.ndim == 1 else surface_temperature

    # SW absorbed at surface: consistent with atmospheric absorption
    sw_sfc = sw_down_surface[:, None] if sw_down_surface.ndim == 1 else sw_down_surface
    sw_surface = sw_sfc * (1.0 - surface_albedo)

    # LW up from surface
    lw_up = STEFAN_BOLTZMANN * t_s**4

    # Surface density
    t_air_safe = jnp.maximum(t_air, 1.0)
    rho_sfc = surface_pressure / (gas_constant * t_air_safe)

    # Sensible heat flux (positive = upward, cooling the ocean)
    h_flux = rho_sfc * specific_heat_cp * drag_coefficient * wind_speed * (t_s - t_air)

    # Latent heat flux (positive = upward, cooling the ocean)
    q_sat_sfc = saturation_specific_humidity(t_s, surface_pressure, epsilon)
    e_flux = rho_sfc * latent_heat * drag_coefficient * wind_speed * (q_sat_sfc - q_air)

    return sw_surface + lw_down - lw_up - h_flux - e_flux


def step_slab_ocean(
    ocean: OceanState,
    net_surface_flux: jnp.ndarray,
    q_flux: jnp.ndarray,
    heat_capacity: float,
    dt: float,
) -> OceanState:
    """Advance the slab ocean by one timestep (forward Euler).

    Parameters
    ----------
    ocean : OceanState
        Current ocean state.
    net_surface_flux : jnp.ndarray
        Net downward surface flux [W/m²], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    q_flux : jnp.ndarray
        Prescribed ocean heat transport [W/m²], shape ``(n_lat,)`` or ``(n_lat, n_lon)``.
    heat_capacity : float
        Ocean heat capacity per unit area [J/(m²·K)].
    dt : float
        Timestep [s].

    Returns
    -------
    OceanState
        Updated ocean state.
    """
    sst_new = ocean.surface_temperature + dt * (net_surface_flux + q_flux) / heat_capacity
    return OceanState(surface_temperature=sst_new)


def step_slab_ocean_implicit(
    ocean: OceanState,
    net_surface_flux: jnp.ndarray,
    dflux_dt_s: jnp.ndarray,
    q_flux: jnp.ndarray,
    heat_capacity: float,
    dt: float,
) -> OceanState:
    r"""Advance the slab ocean by one timestep (linearized implicit).

    Linearizes the surface flux around the current SST and solves
    implicitly for unconditional stability::

        C · (T_new - T_old) / dt = F(T_old) + dF/dT_s · (T_new - T_old) + Q

    giving::

        T_new = T_old + dt · (F + Q) / (C - dt · dF/dT_s)

    where ``dF/dT_s < 0`` (more emission and sensible/latent flux at
    higher SST), so the denominator ``C - dt · dF/dT_s > C``, ensuring
    unconditional stability for any timestep.

    Parameters
    ----------
    ocean : OceanState
        Current ocean state.
    net_surface_flux : jnp.ndarray
        Net downward surface flux evaluated at current SST [W/m²].
    dflux_dt_s : jnp.ndarray
        Derivative of net flux with respect to SST [W/(m²·K)].
        Should be negative (more cooling at higher T_s).
    q_flux : jnp.ndarray
        Prescribed ocean heat transport [W/m²].
    heat_capacity : float
        Ocean heat capacity per unit area [J/(m²·K)].
    dt : float
        Timestep [s].

    Returns
    -------
    OceanState
        Updated ocean state.
    """
    sst_new = ocean.surface_temperature + dt * (net_surface_flux + q_flux) / (
        heat_capacity - dt * dflux_dt_s
    )
    return OceanState(surface_temperature=sst_new)


def surface_flux_derivative(
    surface_temperature: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    *,
    gas_constant: float,
    specific_heat_cp: float,
    epsilon: float,
    latent_heat: float,
    drag_coefficient: float | jnp.ndarray,
) -> jnp.ndarray:
    r"""Derivative of net surface flux with respect to SST.

    .. math::
        \frac{dF}{dT_s} = -4\sigma T_s^3
            - \rho \, c_p \, C_H \, |V|
            - \rho \, L \, C_H \, |V| \, \frac{dq_{sat}}{dT_s}

    The Clausius-Clapeyron derivative is::

        dq_sat/dT_s ≈ L · ε · e_sat / (R_d · T_s²) · q_sat / e_sat
                     = L · q_sat / (R_v · T_s²)

    This derivative is always negative (higher SST → more cooling),
    ensuring the implicit denominator is always > C.

    Parameters
    ----------
    surface_temperature : jnp.ndarray
        SST [K].
    wind_speed : jnp.ndarray
        Lowest-level wind speed [m/s].
    surface_pressure : jnp.ndarray
        Surface pressure [Pa].
    gas_constant : float
        Specific gas constant for dry air [J/(kg·K)].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    epsilon : float
        Ratio R_d / R_v.
    latent_heat : float
        Latent heat of vaporization [J/kg].
    drag_coefficient : float or jnp.ndarray
        Surface drag coefficient C_H.

    Returns
    -------
    jnp.ndarray
        dF/dT_s [W/(m²·K)], always negative.
    """
    t_s = surface_temperature[:, None] if surface_temperature.ndim == 1 else surface_temperature

    # Surface density
    t_safe = jnp.maximum(t_s, 1.0)
    rho_sfc = surface_pressure / (gas_constant * t_safe)

    # LW up derivative: -4σT_s³
    dlw_dt = -4.0 * STEFAN_BOLTZMANN * t_s**3

    # Sensible heat flux derivative: -ρ·cp·C_H·|V|
    dh_dt = -rho_sfc * specific_heat_cp * drag_coefficient * wind_speed

    # Latent heat flux derivative: -ρ·L·C_H·|V|·dq_sat/dT_s
    # dq_sat/dT_s = L·q_sat / (R_v·T_s²)  where R_v = R_d/ε
    r_v = gas_constant / epsilon
    q_sat = saturation_specific_humidity(t_s, surface_pressure, epsilon)
    dqsat_dt = latent_heat * q_sat / (r_v * t_s**2)
    dle_dt = -rho_sfc * latent_heat * drag_coefficient * wind_speed * dqsat_dt

    return dlw_dt + dh_dt + dle_dt


# ---------------------------------------------------------------------------
# Bucket land physics
# ---------------------------------------------------------------------------


def beta_function(
    bucket_depth: jnp.ndarray,
    w_crit: float,
) -> jnp.ndarray:
    """Evaporation resistance factor (Manabe 1969).

    Linear ramp from 0 (dry soil) to 1 (soil at or above critical depth)::

        β = min(1, W / W_crit)

    Parameters
    ----------
    bucket_depth : jnp.ndarray
        Current bucket water depth [m].
    w_crit : float
        Critical depth above which β = 1 [m].

    Returns
    -------
    jnp.ndarray
        Evaporation availability factor in [0, 1].
    """
    return jnp.minimum(1.0, bucket_depth / jnp.maximum(w_crit, 1.0e-10))


def compute_net_land_flux(
    land_temperature: jnp.ndarray,
    t_air: jnp.ndarray,
    q_air: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    sw_down_surface: jnp.ndarray,
    lw_down: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    gravity: float,
    gas_constant: float,
    specific_heat_cp: float,
    epsilon: float,
    latent_heat: float,
    drag_coefficient: float | jnp.ndarray,
    surface_albedo: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute net downward surface energy flux over land [W/m²].

    Same energy balance as :func:`compute_net_surface_flux` but with
    evaporation scaled by the soil moisture availability factor β::

        F_net = SW_absorbed + LW_down − σ·T_land⁴ − H − β·L·E_pot

    Parameters
    ----------
    land_temperature : jnp.ndarray
        Soil temperature [K], shape ``(n_lat, n_lon)``.
    t_air, q_air, wind_speed, surface_pressure : jnp.ndarray
        Lowest-level atmospheric fields, shape ``(n_lat, n_lon)``.
    sw_down_surface : jnp.ndarray
        Downward SW flux at the surface [W/m²], shape ``(n_lat, n_lon)``.
        From :func:`shortwave_heating`.
    lw_down : jnp.ndarray
        Downward LW flux at the surface [W/m²], shape ``(n_lat, n_lon)``.
    beta : jnp.ndarray
        Evaporation availability factor [0, 1], shape ``(n_lat, n_lon)``.
    gravity, gas_constant, specific_heat_cp, epsilon, latent_heat : float
        Physical constants.
    drag_coefficient : float or jnp.ndarray
        Surface transfer coefficient C_H.
    surface_albedo : float or jnp.ndarray
        Surface albedo (0-1).

    Returns
    -------
    net_flux : jnp.ndarray
        Net downward surface flux [W/m²], shape ``(n_lat, n_lon)``.
    evaporation : jnp.ndarray
        Evaporation rate [kg/(m²·s)], shape ``(n_lat, n_lon)``.
        Positive = upward (moisture leaving the surface).
    """
    t_s = land_temperature

    # SW absorbed at surface: consistent with atmospheric absorption
    sw_surface = sw_down_surface * (1.0 - surface_albedo)

    # LW up from surface
    lw_up = STEFAN_BOLTZMANN * t_s**4

    # Surface density
    t_air_safe = jnp.maximum(t_air, 1.0)
    rho_sfc = surface_pressure / (gas_constant * t_air_safe)

    # Sensible heat flux (positive = upward)
    h_flux = rho_sfc * specific_heat_cp * drag_coefficient * wind_speed * (t_s - t_air)

    # Potential evaporation (positive = upward)
    q_sat_sfc = saturation_specific_humidity(t_s, surface_pressure, epsilon)
    e_pot = rho_sfc * drag_coefficient * wind_speed * (q_sat_sfc - q_air)
    evaporation = beta * jnp.maximum(e_pot, 0.0)

    # Latent heat flux (positive = upward)
    le_flux = latent_heat * evaporation

    return sw_surface + lw_down - lw_up - h_flux - le_flux, evaporation


def land_flux_derivative(
    land_temperature: jnp.ndarray,
    wind_speed: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    gas_constant: float,
    specific_heat_cp: float,
    epsilon: float,
    latent_heat: float,
    drag_coefficient: float | jnp.ndarray,
) -> jnp.ndarray:
    r"""Derivative of net land surface flux with respect to soil temperature.

    Same as :func:`surface_flux_derivative` but the latent term is scaled
    by β (evaporation availability)::

        dF/dT = -4σT³ - ρ·cp·C_H·|V| - β·ρ·L·C_H·|V|·dq_sat/dT

    Parameters
    ----------
    land_temperature : jnp.ndarray
        Soil temperature [K], shape ``(n_lat, n_lon)``.
    wind_speed : jnp.ndarray
        Lowest-level wind speed [m/s].
    surface_pressure : jnp.ndarray
        Surface pressure [Pa].
    beta : jnp.ndarray
        Evaporation availability [0, 1].
    gas_constant, specific_heat_cp, epsilon, latent_heat : float
        Physical constants.
    drag_coefficient : float or jnp.ndarray
        Surface transfer coefficient C_H.

    Returns
    -------
    jnp.ndarray
        dF/dT_land [W/(m²·K)], always negative.
    """
    t_s = land_temperature
    t_safe = jnp.maximum(t_s, 1.0)
    rho_sfc = surface_pressure / (gas_constant * t_safe)

    dlw_dt = -4.0 * STEFAN_BOLTZMANN * t_s**3
    dh_dt = -rho_sfc * specific_heat_cp * drag_coefficient * wind_speed

    r_v = gas_constant / epsilon
    q_sat = saturation_specific_humidity(t_s, surface_pressure, epsilon)
    dqsat_dt = latent_heat * q_sat / (r_v * t_s**2)
    dle_dt = -beta * rho_sfc * latent_heat * drag_coefficient * wind_speed * dqsat_dt

    return dlw_dt + dh_dt + dle_dt


def step_land_implicit(
    land: LandState,
    net_flux: jnp.ndarray,
    dflux_dt: jnp.ndarray,
    heat_capacity: float,
    dt: float,
) -> LandState:
    r"""Advance soil temperature by one timestep (linearized implicit).

    Same linearization as :func:`step_slab_ocean_implicit`::

        T_new = T_old + dt · F / (C - dt · dF/dT)

    Parameters
    ----------
    land : LandState
        Current land state.
    net_flux : jnp.ndarray
        Net downward surface flux [W/m²], shape ``(n_lat, n_lon)``.
    dflux_dt : jnp.ndarray
        Derivative of net flux w.r.t. soil temperature [W/(m²·K)].
    heat_capacity : float
        Soil heat capacity per unit area [J/(m²·K)].
    dt : float
        Timestep [s].

    Returns
    -------
    LandState
        Updated land state (only soil_temperature changed).
    """
    t_new = land.soil_temperature + dt * net_flux / (heat_capacity - dt * dflux_dt)
    return land.replace(soil_temperature=t_new)


def step_bucket_hydrology(
    land: LandState,
    precipitation: jnp.ndarray,
    evaporation: jnp.ndarray,
    bucket_capacity: float,
    dt: float,
) -> LandState:
    """Advance bucket water depth: dW/dt = P − E − R.

    Overflow runoff occurs when the bucket exceeds capacity.
    The bucket is clamped to [0, W_max].

    Parameters
    ----------
    land : LandState
        Current land state.
    precipitation : jnp.ndarray
        Precipitation rate [kg/(m²·s)], shape ``(n_lat, n_lon)``.
        Converted to water depth via ρ_w = 1000 kg/m³.
    evaporation : jnp.ndarray
        Evaporation rate [kg/(m²·s)], shape ``(n_lat, n_lon)``.
    bucket_capacity : float
        Maximum bucket depth W_max [m].
    dt : float
        Timestep [s].

    Returns
    -------
    LandState
        Updated land state (only bucket_depth changed).
    """
    rho_water = 1000.0  # kg/m³
    w_new = land.bucket_depth + dt * (precipitation - evaporation) / rho_water
    w_new = jnp.clip(w_new, 0.0, bucket_capacity)
    return land.replace(bucket_depth=w_new)


def diagnose_precipitation(
    t_grid: jnp.ndarray,
    q_grid: jnp.ndarray,
    pressure: jnp.ndarray,
    dsigma: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    *,
    gravity: float,
    epsilon: float,
    latent_heat: float,
    specific_heat_cp: float,
    gas_constant: float,
    tau_bm: float = 7200.0,
    rh_ref: float = 0.7,
    n_condensation_iterations: int = 3,
    rh_condensation: float = 1.0,
    tau_adjustment: float = 43200.0,
) -> jnp.ndarray:
    """Diagnose precipitation rate from the current atmospheric state.

    Runs Betts-Miller convection and large-scale condensation
    diagnostically, then column-integrates the negative humidity
    tendencies to obtain a precipitation rate.

    Parameters
    ----------
    t_grid : jnp.ndarray
        Temperature on the grid, shape ``(n_levels, n_lat, n_lon)``.
    q_grid : jnp.ndarray
        Specific humidity on the grid, shape ``(n_levels, n_lat, n_lon)``.
    pressure : jnp.ndarray
        Pressure at full levels [Pa], shape ``(n_levels, n_lat, n_lon)``.
    dsigma : jnp.ndarray
        Layer thickness Δσ, shape ``(n_levels,)``.
    surface_pressure : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    gravity, epsilon, latent_heat, specific_heat_cp, gas_constant : float
        Physical constants.
    tau_bm : float
        Betts-Miller relaxation timescale [s].
    rh_ref : float
        Betts-Miller reference relative humidity.
    n_condensation_iterations : int
        Condensation iterations.
    rh_condensation : float
        Condensation RH threshold.
    tau_adjustment : float
        Relaxation timescale for condensation tendency [s].

    Returns
    -------
    jnp.ndarray
        Precipitation rate [kg/(m²·s)], shape ``(n_lat, n_lon)``.
        Always non-negative.
    """
    # Betts-Miller convective tendencies
    _dt_bm, dq_bm = betts_miller_convection(
        t_grid,
        q_grid,
        pressure,
        dsigma,
        epsilon,
        latent_heat,
        specific_heat_cp,
        gas_constant,
        tau_bm=tau_bm,
        rh_ref=rh_ref,
    )

    # Large-scale condensation tendencies
    _t_cond, q_cond, _condensate = large_scale_condensation(
        t_grid,
        q_grid,
        pressure,
        epsilon,
        latent_heat,
        specific_heat_cp,
        gas_constant,
        n_iterations=n_condensation_iterations,
        rh_threshold=rh_condensation,
    )
    dq_cond = (q_cond - q_grid) / tau_adjustment

    # Total humidity tendency [kg/kg/s]
    dq_total = dq_bm + dq_cond

    # Column-integrate negative tendencies: P = -∫(dq * dp / g)
    # dp = Δσ * ps for each level
    dp = dsigma[:, None, None] * surface_pressure[None, :, :]
    precip = -jnp.sum(jnp.minimum(dq_total, 0.0) * dp, axis=0) / gravity
    return jnp.maximum(precip, 0.0)
