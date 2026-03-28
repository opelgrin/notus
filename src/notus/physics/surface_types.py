"""Surface type classification and properties.

Provides per-gridpoint surface properties (albedo, roughness lengths)
and idealized land-sea mask generators for aquaplanet and continent
configurations.

All data structures are JAX pytrees for JIT compatibility.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Default physical parameters
# ---------------------------------------------------------------------------

# Ocean: smooth surface, low albedo
OCEAN_Z0_MOMENTUM: float = 1.0e-4  # [m]
OCEAN_Z0_HEAT: float = 1.0e-5  # [m]
OCEAN_ALBEDO: float = 0.06

# Land: rougher surface, higher albedo
LAND_Z0_MOMENTUM: float = 0.05  # [m] (grassland/shrub)
LAND_Z0_HEAT: float = 0.005  # [m]
LAND_ALBEDO: float = 0.25


# ---------------------------------------------------------------------------
# SurfaceProperties
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SurfaceProperties:
    """Per-gridpoint surface properties for land-ocean coupling.

    All arrays have shape ``(n_lat, n_lon)``.

    Attributes
    ----------
    land_fraction : jnp.ndarray
        Fraction of each grid cell that is land, in [0, 1].
    albedo : jnp.ndarray
        Surface albedo (dimensionless, 0-1).
    z0_momentum : jnp.ndarray
        Roughness length for momentum [m].
    z0_heat : jnp.ndarray
        Roughness length for heat and moisture [m].
    """

    land_fraction: jnp.ndarray
    albedo: jnp.ndarray
    z0_momentum: jnp.ndarray
    z0_heat: jnp.ndarray

    @property
    def n_lat(self) -> int:
        return self.land_fraction.shape[0]

    @property
    def n_lon(self) -> int:
        return self.land_fraction.shape[1]

    def replace(self, **kwargs: jnp.ndarray) -> SurfaceProperties:
        """Return a new instance with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


def _surface_props_flatten(
    props: SurfaceProperties,
) -> tuple[tuple[jnp.ndarray, ...], None]:
    return (
        props.land_fraction,
        props.albedo,
        props.z0_momentum,
        props.z0_heat,
    ), None


def _surface_props_unflatten(
    _aux: None,
    children: tuple[jnp.ndarray, ...],
) -> SurfaceProperties:
    return SurfaceProperties(
        land_fraction=children[0],
        albedo=children[1],
        z0_momentum=children[2],
        z0_heat=children[3],
    )


jax.tree_util.register_pytree_node(
    SurfaceProperties,
    _surface_props_flatten,
    _surface_props_unflatten,
)


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------


def aquaplanet_surface(
    n_lat: int,
    n_lon: int,
    *,
    ocean_albedo: float = OCEAN_ALBEDO,
    ocean_z0_momentum: float = OCEAN_Z0_MOMENTUM,
    ocean_z0_heat: float = OCEAN_Z0_HEAT,
) -> SurfaceProperties:
    """All-ocean surface (the default for aquaplanet experiments).

    Parameters
    ----------
    n_lat, n_lon : int
        Grid dimensions.
    ocean_albedo : float
        Ocean surface albedo.
    ocean_z0_momentum, ocean_z0_heat : float
        Ocean roughness lengths [m].

    Returns
    -------
    SurfaceProperties
        All-ocean surface with uniform properties.
    """
    shape = (n_lat, n_lon)
    return SurfaceProperties(
        land_fraction=jnp.zeros(shape),
        albedo=jnp.full(shape, ocean_albedo),
        z0_momentum=jnp.full(shape, ocean_z0_momentum),
        z0_heat=jnp.full(shape, ocean_z0_heat),
    )


def flat_continent_surface(
    latitudes: np.ndarray | jnp.ndarray,
    longitudes: np.ndarray | jnp.ndarray,
    *,
    lat_south: float = -90.0,
    lat_north: float = 90.0,
    lon_west: float = 0.0,
    lon_east: float = 180.0,
    ocean_albedo: float = OCEAN_ALBEDO,
    land_albedo: float = LAND_ALBEDO,
    ocean_z0_momentum: float = OCEAN_Z0_MOMENTUM,
    ocean_z0_heat: float = OCEAN_Z0_HEAT,
    land_z0_momentum: float = LAND_Z0_MOMENTUM,
    land_z0_heat: float = LAND_Z0_HEAT,
) -> SurfaceProperties:
    """Rectangular continent on an otherwise ocean-covered sphere.

    The continent covers grid points where latitude is between
    ``lat_south`` and ``lat_north`` (degrees) and longitude is between
    ``lon_west`` and ``lon_east`` (degrees).  Useful for Frierson-style
    idealized land experiments.

    Parameters
    ----------
    latitudes : array
        Latitude values [radians], shape ``(n_lat,)``.
    longitudes : array
        Longitude values [radians], shape ``(n_lon,)``.
    lat_south, lat_north : float
        Southern and northern edges of the continent [degrees].
    lon_west, lon_east : float
        Western and eastern edges of the continent [degrees].
    ocean_albedo, land_albedo : float
        Albedo for ocean and land points.
    ocean_z0_momentum, ocean_z0_heat : float
        Roughness lengths for ocean [m].
    land_z0_momentum, land_z0_heat : float
        Roughness lengths for land [m].

    Returns
    -------
    SurfaceProperties
        Surface with a rectangular continent.
    """
    lat_deg = jnp.degrees(jnp.asarray(latitudes))  # (n_lat,)
    lon_deg = jnp.degrees(jnp.asarray(longitudes))  # (n_lon,)

    # 2D masks via broadcasting
    lat_mask = (lat_deg >= lat_south) & (lat_deg <= lat_north)  # (n_lat,)
    lon_mask = (lon_deg >= lon_west) & (lon_deg <= lon_east)  # (n_lon,)
    land = (lat_mask[:, None] & lon_mask[None, :]).astype(jnp.float64)  # (n_lat, n_lon)

    ocean = 1.0 - land
    albedo = ocean * ocean_albedo + land * land_albedo
    z0_m = ocean * ocean_z0_momentum + land * land_z0_momentum
    z0_h = ocean * ocean_z0_heat + land * land_z0_heat

    return SurfaceProperties(
        land_fraction=land,
        albedo=albedo,
        z0_momentum=z0_m,
        z0_heat=z0_h,
    )


def moisture_dependent_albedo(
    bucket_depth: jnp.ndarray,
    bucket_capacity: float,
    albedo_dry: float,
    albedo_wet: float,
    land_fraction: jnp.ndarray,
    ocean_albedo: float | jnp.ndarray,
) -> jnp.ndarray:
    """Frierson (2006) moisture-dependent land albedo.

    Land albedo varies linearly with bucket fill fraction::

        α_land = α_wet + (α_dry − α_wet) · (1 − W / W_max)

    Blended with ocean albedo using land fraction::

        α = (1 − f) · α_ocean + f · α_land

    Parameters
    ----------
    bucket_depth : jnp.ndarray
        Current bucket water depth [m], shape ``(n_lat, n_lon)``.
    bucket_capacity : float
        Maximum bucket depth W_max [m].
    albedo_dry : float
        Land albedo when bucket is empty.
    albedo_wet : float
        Land albedo when bucket is full.
    land_fraction : jnp.ndarray
        Land fraction [0, 1], shape ``(n_lat, n_lon)``.
    ocean_albedo : float or jnp.ndarray
        Ocean albedo.

    Returns
    -------
    jnp.ndarray
        Blended surface albedo, shape ``(n_lat, n_lon)``.
    """
    fill = jnp.clip(bucket_depth / jnp.maximum(bucket_capacity, 1.0e-10), 0.0, 1.0)
    alpha_land = albedo_wet + (albedo_dry - albedo_wet) * (1.0 - fill)
    return (1.0 - land_fraction) * ocean_albedo + land_fraction * alpha_land
