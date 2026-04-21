"""Xarray conversion utilities for Notus.

Convert between Notus spectral model state and xarray Datasets on both
Gaussian and regular latitude-longitude grids.

The spectral-to-regular interpolation is exact up to the truncation
wavenumber: spherical harmonic coefficients are evaluated at target
grid points via Legendre polynomials and inverse FFT.

Functions
---------
state_to_dataset
    Spectral state → xarray Dataset on the native Gaussian grid.
to_regular_latlon
    Spectral state → xarray Dataset on a regular lat/lon grid.
dataset_to_state
    Gaussian-grid xarray Dataset → :class:`DatasetContents`.
from_regular_latlon
    Regular lat/lon xarray Dataset → :class:`DatasetContents`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from notus.constants import PlanetaryConstants
from notus.operators import spectral_curl, spectral_divergence, uv_from_vordiv
from notus.physics.surface import LandState, OceanState, SeaIceState, SurfaceState
from notus.physics.surface_types import SurfaceProperties
from notus.spherical_harmonics import compute_legendre_polynomials
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetContents:
    """Everything extracted from an xarray Dataset.

    Returned by :func:`dataset_to_state` and :func:`from_regular_latlon`.
    Only ``state`` is guaranteed to be non-None; optional fields are
    populated when the corresponding variables are present in the dataset.

    Attributes
    ----------
    state : PrimitiveEquationState
        Atmospheric state in spectral space.
    surface : SurfaceState or None
        Ocean (and optional land) surface state.
    surface_geopotential : jnp.ndarray or None
        Surface geopotential Φₛ = g·zₛ in spectral space, ``(n_spectral,)``.
    reference_temperature : np.ndarray or None
        Semi-implicit reference temperature profile, ``(n_levels,)``.
    q_flux : jnp.ndarray or None
        Prescribed ocean heat transport [W/m²], ``(n_lat,)``.
    surface_properties : SurfaceProperties or None
        Land fraction, albedo, and roughness lengths.
    """

    state: PrimitiveEquationState
    surface: SurfaceState | None = None
    surface_geopotential: jnp.ndarray | None = None
    reference_temperature: np.ndarray | None = None
    q_flux: jnp.ndarray | None = None
    surface_properties: SurfaceProperties | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spectral_flat_index(truncation: int, m: int, n: int) -> int:
    """Linear index into lower-triangular spectral storage."""
    return m * (truncation + 1) - m * (m - 1) // 2 + (n - m)


def _build_unpack_indices(truncation: int) -> jnp.ndarray:
    """Build flat → 2D unpack index array for inverse spectral transforms.

    Invalid entries point one past the coefficient array so that a
    zero-padded gather yields zeros for out-of-triangle positions.
    """
    t1 = truncation + 1
    n_spectral = t1 * (t1 + 1) // 2
    idx = np.full((t1, t1), n_spectral, dtype=np.int32)
    for m in range(t1):
        for k in range(truncation - m + 1):
            idx[m, k] = _spectral_flat_index(truncation, m, m + k)
    return jnp.array(idx)


def _build_legendre_3d(truncation: int, sin_lat: jnp.ndarray) -> jnp.ndarray:
    """Build 3D Legendre array for spectral evaluation at given latitudes.

    Returns shape ``(T+1, n_lat, T+1)`` with ``[m, j, k] = P̄ₙᵐ(sin φⱼ)``,
    ``k = n − m``.
    """
    t1 = truncation + 1
    n_lat = sin_lat.shape[0]
    leg_flat = np.asarray(compute_legendre_polynomials(truncation, sin_lat))
    leg3d = np.zeros((t1, n_lat, t1))
    for m in range(t1):
        for k in range(truncation - m + 1):
            leg3d[m, :, k] = leg_flat[:, _spectral_flat_index(truncation, m, m + k)]
    return jnp.array(leg3d)


@functools.partial(jax.jit, static_argnums=(3, 4))
def _inverse_spectral(
    coeffs: jnp.ndarray,
    legendre_3d: jnp.ndarray,
    unpack_indices: jnp.ndarray,
    truncation: int,
    n_lon: int,
) -> jnp.ndarray:
    """Evaluate spectral coefficients on a grid via inverse Legendre + FFT.

    Target latitudes are defined by *legendre_3d*; longitudes are uniform
    in [0, 2π) with *n_lon* points.
    """
    t1 = truncation + 1
    padded = jnp.append(coeffs, jnp.zeros(1, dtype=coeffs.dtype))
    coeffs_3d = padded[unpack_indices]
    fourier = jnp.einsum("mk,mjk->jm", coeffs_3d, legendre_3d)
    n_rfft = n_lon // 2 + 1
    fourier_full = jnp.zeros((fourier.shape[0], n_rfft), dtype=jnp.complex128)
    fourier_full = fourier_full.at[:, :t1].set(fourier)
    return jnp.fft.irfft(fourier_full * n_lon, n=n_lon, axis=1)


def _regrid_to_gaussian(
    field: np.ndarray,
    lat_in: np.ndarray,
    lon_in: np.ndarray,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation from a regular grid to Gaussian grid points.

    Parameters
    ----------
    field : np.ndarray
        Input on regular grid, shape ``(n_lat_in, n_lon_in)``.
    lat_in : np.ndarray
        Input latitudes [degrees], ascending or descending.
    lon_in : np.ndarray
        Input longitudes [degrees], uniform and ascending in [0, 360).
    lat_out : np.ndarray
        Target latitudes [degrees].
    lon_out : np.ndarray
        Target longitudes [degrees] in [0, 360).

    Returns
    -------
    np.ndarray
        Interpolated field, shape ``(len(lat_out), len(lon_out))``.
    """
    # Ensure ascending latitude for searchsorted
    if lat_in[0] > lat_in[-1]:
        lat_in = lat_in[::-1]
        field = field[::-1]

    n_lon_in = len(lon_in)

    # Latitude: find bracket and fractional weight
    j0 = np.searchsorted(lat_in, lat_out, side="right") - 1
    j0 = np.clip(j0, 0, len(lat_in) - 2)
    dlat = lat_in[j0 + 1] - lat_in[j0]
    wj = np.where(dlat != 0, (lat_out - lat_in[j0]) / dlat, 0.0)
    wj = np.clip(wj, 0.0, 1.0)

    # Longitude: uniform grid with periodic wrapping
    dlon = lon_in[1] - lon_in[0]
    lon_norm = lon_out % 360.0
    i0 = (np.floor(lon_norm / dlon).astype(int)) % n_lon_in
    i1 = (i0 + 1) % n_lon_in
    wi = (lon_norm - lon_in[i0]) / dlon
    wi = np.clip(wi, 0.0, 1.0)

    # Gather corners: (n_lat_out, n_lon_out) via outer-product indexing
    f00 = field[j0[:, None], i0[None, :]]
    f01 = field[j0[:, None], i1[None, :]]
    f10 = field[j0[:, None] + 1, i0[None, :]]
    f11 = field[j0[:, None] + 1, i1[None, :]]

    wj2 = wj[:, None]
    wi2 = wi[None, :]
    return (1.0 - wj2) * ((1.0 - wi2) * f00 + wi2 * f01) + wj2 * ((1.0 - wi2) * f10 + wi2 * f11)


def _winds_to_vordiv(
    u_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    transform: SpectralTransform,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-level grid winds → spectral (vorticity, divergence).

    Uses the spectral curl/divergence operators on ``u/cos φ`` and
    ``v/cos φ`` (the flux components divided by cos²φ).
    """
    cos_lat = transform.grid.cos_lat[:, None]
    safe_cos = jnp.maximum(cos_lat, 1e-30)
    a_hat = transform.grid_to_spectral(u_grid / safe_cos)
    b_hat = transform.grid_to_spectral(v_grid / safe_cos)
    return (
        spectral_curl(a_hat, b_hat, transform.arrays),
        spectral_divergence(a_hat, b_hat, transform.arrays),
    )


def _transform_state_fields(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    to_grid: Callable[[jnp.ndarray], jnp.ndarray],
    cos_lat: jnp.ndarray,
) -> dict[str, np.ndarray]:
    """Transform all spectral state fields to grid-point numpy arrays.

    Parameters
    ----------
    to_grid
        Maps spectral coefficients ``(n_spectral,)`` to a grid field.
    cos_lat
        Cosine of target latitudes, shape ``(n_lat,)``.
    """
    arrays = transform.arrays
    n_levels = state.n_levels

    def _uv(vort: jnp.ndarray, div: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return uv_from_vordiv(vort, div, arrays)

    u_cos_spec, v_cos_spec = jax.vmap(_uv)(state.vorticity, state.divergence)

    # Batch spectral → grid for all per-level fields
    specs: list[jnp.ndarray] = [u_cos_spec, v_cos_spec, state.temperature]
    if state.has_humidity:
        specs.append(state.humidity)  # type: ignore[arg-type]
    all_spec = jnp.concatenate(specs, axis=0)
    all_grid = jax.vmap(to_grid)(all_spec)

    idx = 0
    u_cos = all_grid[idx : idx + n_levels]
    idx += n_levels
    v_cos = all_grid[idx : idx + n_levels]
    idx += n_levels
    t_grid = all_grid[idx : idx + n_levels]
    idx += n_levels

    cos3d = jnp.maximum(cos_lat[None, :, None], 1e-30)
    fields: dict[str, np.ndarray] = {
        "u": np.asarray(u_cos / cos3d),
        "v": np.asarray(v_cos / cos3d),
        "temperature": np.asarray(t_grid),
    }
    if state.has_humidity:
        fields["specific_humidity"] = np.asarray(all_grid[idx : idx + n_levels])

    lnps_grid = to_grid(state.log_surface_pressure)
    fields["surface_pressure"] = np.asarray(planet.reference_pressure * jnp.exp(lnps_grid))
    return fields


def _gridpoint_to_spectral_state(
    u_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    t_grid: jnp.ndarray,
    ps_grid: jnp.ndarray,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    q_grid: jnp.ndarray | None = None,
) -> PrimitiveEquationState:
    """Convert grid-point fields on the Gaussian grid to spectral state.

    Parameters
    ----------
    u_grid, v_grid : jnp.ndarray
        Wind components [m/s], shape ``(n_levels, n_lat, n_lon)``.
    t_grid : jnp.ndarray
        Temperature [K], shape ``(n_levels, n_lat, n_lon)``.
    ps_grid : jnp.ndarray
        Surface pressure [Pa], shape ``(n_lat, n_lon)``.
    q_grid : jnp.ndarray, optional
        Specific humidity [kg/kg], shape ``(n_levels, n_lat, n_lon)``.
    """
    # Temperature → spectral
    temperature = jax.vmap(transform.grid_to_spectral)(t_grid)

    # Winds → spectral vorticity and divergence
    def _vordiv(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _winds_to_vordiv(u, v, transform)

    vorticity, divergence = jax.vmap(_vordiv)(u_grid, v_grid)

    # Surface pressure → log(ps/p₀) → spectral
    lnps = jnp.log(ps_grid / planet.reference_pressure)
    log_surface_pressure = transform.grid_to_spectral(lnps)

    humidity = None
    if q_grid is not None:
        humidity = jax.vmap(transform.grid_to_spectral)(q_grid)

    return PrimitiveEquationState(
        vorticity=vorticity,
        divergence=divergence,
        temperature=temperature,
        log_surface_pressure=log_surface_pressure,
        humidity=humidity,
    )


_DIMS_3D: list[str] = ["sigma", "lat", "lon"]
_DIMS_2D: list[str] = ["lat", "lon"]

# Field metadata: (units, long_name)
_FIELD_META: dict[str, tuple[str, str]] = {
    "u": ("m/s", "zonal wind"),
    "v": ("m/s", "meridional wind"),
    "temperature": ("K", "temperature"),
    "specific_humidity": ("kg/kg", "specific humidity"),
    "surface_pressure": ("Pa", "surface pressure"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def state_to_dataset(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    *,
    surface: SurfaceState | None = None,
    surface_geopotential: jnp.ndarray | None = None,
    reference_temperature: np.ndarray | None = None,
    q_flux: jnp.ndarray | None = None,
    surface_properties: SurfaceProperties | None = None,
) -> xr.Dataset:
    """Convert spectral state to xarray Dataset on the native Gaussian grid.

    Parameters
    ----------
    state : PrimitiveEquationState
        Atmospheric state in spectral space.
    transform : SpectralTransform
        Spectral transform (defines the Gaussian grid and truncation).
    planet : PlanetaryConstants
        Planetary constants (for reference pressure).
    levels : SigmaLevels
        Sigma vertical coordinate.
    surface : SurfaceState, optional
        Surface state (ocean SST and optional land fields).
    surface_geopotential : jnp.ndarray, optional
        Surface geopotential Φₛ = g·zₛ in spectral space, ``(n_spectral,)``.
        Stored as ``"orography"`` in metres.
    reference_temperature : np.ndarray, optional
        Semi-implicit reference temperature profile, ``(n_levels,)``.
    q_flux : jnp.ndarray, optional
        Prescribed ocean heat transport [W/m²], ``(n_lat,)``.
    surface_properties : SurfaceProperties, optional
        Land fraction, albedo, and roughness lengths.

    Returns
    -------
    xr.Dataset
        Grid-point fields with coordinates ``lat`` (degrees north, Gaussian),
        ``lon`` (degrees east), ``sigma``.

    Examples
    --------
    >>> ds = state_to_dataset(state, transform, EARTH, levels)
    >>> ds["temperature"].sel(sigma=0.5, method="nearest").plot()
    """
    grid = transform.grid
    fields = _transform_state_fields(
        state,
        transform,
        planet,
        transform.spectral_to_grid,
        grid.cos_lat,
    )

    lat = np.degrees(np.asarray(grid.latitudes))
    lon = np.degrees(np.asarray(grid.longitudes))
    sigma = np.asarray(levels.sigma_full)

    data_vars: dict[str, xr.Variable] = {}
    for name, arr in fields.items():
        units, long_name = _FIELD_META[name]
        is_3d = arr.ndim == len(_DIMS_3D)
        dims = _DIMS_3D if is_3d else _DIMS_2D
        data_vars[name] = xr.Variable(dims, arr, {"units": units, "long_name": long_name})

    if surface is not None:
        _add_surface_vars(data_vars, surface)
    _add_static_fields(
        data_vars,
        transform.spectral_to_grid,
        planet,
        surface_geopotential,
        reference_temperature,
        q_flux,
        surface_properties,
    )

    return xr.Dataset(
        data_vars,
        coords={
            "lat": xr.Variable(
                "lat",
                lat,
                {"units": "degrees_north", "long_name": "latitude"},
            ),
            "lon": xr.Variable(
                "lon",
                lon,
                {"units": "degrees_east", "long_name": "longitude"},
            ),
            "sigma": xr.Variable(
                "sigma",
                sigma,
                {"long_name": "sigma level", "positive": "down"},
            ),
        },
        attrs={
            "title": "Notus GCM output",
            "grid_type": "gaussian",
            "truncation": grid.truncation,
        },
    )


def to_regular_latlon(
    state: PrimitiveEquationState,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    levels: SigmaLevels,
    *,
    n_lat: int | None = None,
    n_lon: int | None = None,
    surface: SurfaceState | None = None,
    surface_geopotential: jnp.ndarray | None = None,
    reference_temperature: np.ndarray | None = None,
    q_flux: jnp.ndarray | None = None,
    surface_properties: SurfaceProperties | None = None,
) -> xr.Dataset:
    """Convert spectral state to xarray Dataset on a regular lat/lon grid.

    Evaluates the spherical harmonic expansion at target grid points,
    giving exact interpolation up to the truncation wavenumber (no
    interpolation error for resolved scales).

    Parameters
    ----------
    state : PrimitiveEquationState
        Atmospheric state in spectral space.
    transform : SpectralTransform
        Spectral transform (provides truncation and operator arrays).
    planet : PlanetaryConstants
        Planetary constants.
    levels : SigmaLevels
        Sigma vertical coordinate.
    n_lat : int, optional
        Number of target latitudes. Default: ``2 * truncation``.
    n_lon : int, optional
        Number of target longitudes. Default: ``2 * n_lat``.
        Must be ≥ ``2 * (truncation + 1)`` to avoid aliasing.
    surface : SurfaceState, optional
        Surface state to include (regridded via spectral interpolation).
    surface_geopotential : jnp.ndarray, optional
        Surface geopotential Φₛ = g·zₛ in spectral space, ``(n_spectral,)``.
        Regridded via spectral interpolation and stored as ``"orography"`` [m].
    reference_temperature : np.ndarray, optional
        Semi-implicit reference temperature profile, ``(n_levels,)``.
    q_flux : jnp.ndarray, optional
        Prescribed ocean heat transport [W/m²], ``(n_lat,)``.
        Regridded via spectral interpolation.
    surface_properties : SurfaceProperties, optional
        Regridded via spectral interpolation.

    Returns
    -------
    xr.Dataset
        Grid-point fields on the regular lat/lon grid.

    Examples
    --------
    >>> ds = to_regular_latlon(state, transform, EARTH, levels, n_lat=90)
    >>> ds["u"].sel(sigma=0.5, method="nearest").plot()
    """
    trunc = transform.grid.truncation
    if n_lat is None:
        n_lat = 2 * trunc
    if n_lon is None:
        n_lon = 2 * n_lat
    min_lon = 2 * (trunc + 1)
    if n_lon < min_lon:
        msg = f"n_lon={n_lon} too small for T{trunc} truncation (need ≥ {min_lon})"
        raise ValueError(msg)

    # Regular grid: cell centres, poles excluded
    dlat = 180.0 / n_lat
    target_lat = np.linspace(
        90.0 - dlat / 2,
        -90.0 + dlat / 2,
        n_lat,
    )
    target_lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)

    # Legendre polynomials and unpack indices for target latitudes
    target_sin = jnp.sin(jnp.deg2rad(jnp.array(target_lat)))
    target_cos = jnp.cos(jnp.deg2rad(jnp.array(target_lat)))
    leg3d = _build_legendre_3d(trunc, target_sin)
    unpack_idx = _build_unpack_indices(trunc)

    def to_regular(coeffs: jnp.ndarray) -> jnp.ndarray:
        return _inverse_spectral(
            coeffs,
            leg3d,
            unpack_idx,
            trunc,
            n_lon,
        )

    fields = _transform_state_fields(
        state,
        transform,
        planet,
        to_regular,
        target_cos,
    )

    sigma = np.asarray(levels.sigma_full)

    data_vars: dict[str, xr.Variable] = {}
    for name, arr in fields.items():
        units, long_name = _FIELD_META[name]
        is_3d = arr.ndim == len(_DIMS_3D)
        dims = _DIMS_3D if is_3d else _DIMS_2D
        data_vars[name] = xr.Variable(
            dims,
            arr,
            {"units": units, "long_name": long_name},
        )

    # Regrid surface fields via spectral interpolation
    if surface is not None:
        _add_surface_vars_spectral(
            data_vars,
            surface,
            transform,
            to_regular,
        )
    _add_static_fields_spectral(
        data_vars,
        transform,
        to_regular,
        planet,
        surface_geopotential,
        reference_temperature,
        q_flux,
        surface_properties,
    )

    return xr.Dataset(
        data_vars,
        coords={
            "lat": xr.Variable(
                "lat",
                target_lat,
                {"units": "degrees_north", "long_name": "latitude"},
            ),
            "lon": xr.Variable(
                "lon",
                target_lon,
                {"units": "degrees_east", "long_name": "longitude"},
            ),
            "sigma": xr.Variable(
                "sigma",
                sigma,
                {"long_name": "sigma level", "positive": "down"},
            ),
        },
        attrs={
            "title": "Notus GCM output",
            "grid_type": "regular",
            "truncation": trunc,
        },
    )


def dataset_to_state(
    ds: xr.Dataset,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
) -> DatasetContents:
    """Convert a Gaussian-grid xarray Dataset back to spectral state.

    Expects variable names and coordinates as produced by
    :func:`state_to_dataset`.  All optional fields (surface state,
    orography, reference temperature, etc.) are extracted when present.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset on the Gaussian grid.
    transform : SpectralTransform
        Spectral transform (must match the grid of *ds*).
    planet : PlanetaryConstants
        Planetary constants.

    Returns
    -------
    DatasetContents
        Spectral state and any optional fields found in the dataset.
    """
    u_grid = jnp.array(ds["u"].values)
    v_grid = jnp.array(ds["v"].values)
    t_grid = jnp.array(ds["temperature"].values)
    ps_grid = jnp.array(ds["surface_pressure"].values)
    q_grid = jnp.array(ds["specific_humidity"].values) if "specific_humidity" in ds else None

    state = _gridpoint_to_spectral_state(
        u_grid,
        v_grid,
        t_grid,
        ps_grid,
        transform,
        planet,
        q_grid,
    )

    return DatasetContents(
        state=state,
        surface=_extract_surface(ds),
        surface_geopotential=_extract_surface_geopotential(ds, transform, planet),
        reference_temperature=_extract_reference_temperature(ds),
        q_flux=_extract_q_flux(ds, transform),
        surface_properties=_extract_surface_properties(ds),
    )


def from_regular_latlon(
    ds: xr.Dataset,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
) -> DatasetContents:
    """Convert a regular lat/lon xarray Dataset to spectral state.

    Bilinearly interpolates the regular-grid fields to the Gaussian
    grid, then performs spectral analysis.

    Expects variables ``u``, ``v``, ``temperature``, ``surface_pressure``,
    and optionally ``specific_humidity``, ``orography``, etc. with
    coordinates ``lat``, ``lon``, ``sigma``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset on a regular lat/lon grid.
    transform : SpectralTransform
        Spectral transform defining the target Gaussian grid.
    planet : PlanetaryConstants
        Planetary constants.

    Returns
    -------
    DatasetContents
        Spectral state and any optional fields found in the dataset.
    """
    grid = transform.grid
    target_lat = np.degrees(np.asarray(grid.latitudes))
    target_lon = np.degrees(np.asarray(grid.longitudes))

    lat_in = np.array(ds["lat"].values, dtype=np.float64)
    lon_in = np.array(ds["lon"].values, dtype=np.float64) % 360.0

    n_levels = ds.sizes["sigma"]
    required = ["u", "v", "temperature", "surface_pressure"]
    for var in required:
        if var not in ds:
            msg = f"Missing required variable '{var}' in dataset"
            raise KeyError(msg)

    def _regrid_3d(data: np.ndarray) -> jnp.ndarray:
        """Regrid a (n_levels, n_lat, n_lon) field to the Gaussian grid."""
        out = np.empty((n_levels, len(target_lat), len(target_lon)))
        for k in range(n_levels):
            out[k] = _regrid_to_gaussian(
                data[k],
                lat_in,
                lon_in,
                target_lat,
                target_lon,
            )
        return jnp.array(out)

    def _regrid_2d(data: np.ndarray) -> jnp.ndarray:
        """Regrid a (n_lat, n_lon) field to the Gaussian grid."""
        return jnp.array(
            _regrid_to_gaussian(data, lat_in, lon_in, target_lat, target_lon),
        )

    u_gauss = _regrid_3d(np.asarray(ds["u"].values, dtype=np.float64))
    v_gauss = _regrid_3d(np.asarray(ds["v"].values, dtype=np.float64))
    t_gauss = _regrid_3d(
        np.asarray(ds["temperature"].values, dtype=np.float64),
    )

    ps_np = np.asarray(ds["surface_pressure"].values, dtype=np.float64)
    ps_gauss = _regrid_2d(ps_np)

    q_gauss = None
    if "specific_humidity" in ds:
        q_gauss = _regrid_3d(
            np.asarray(ds["specific_humidity"].values, dtype=np.float64),
        )

    state = _gridpoint_to_spectral_state(
        u_gauss,
        v_gauss,
        t_gauss,
        ps_gauss,
        transform,
        planet,
        q_gauss,
    )

    extras = _extract_optional_regular(
        ds,
        _regrid_2d,
        transform,
        planet,
        lat_in,
        lon_in,
    )
    return DatasetContents(
        state=state,
        surface_geopotential=extras.get("surface_geopotential"),  # type: ignore[arg-type]
        reference_temperature=extras.get("reference_temperature"),  # type: ignore[arg-type]
        q_flux=extras.get("q_flux"),  # type: ignore[arg-type]
        surface_properties=extras.get("surface_properties"),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Surface state helpers
# ---------------------------------------------------------------------------


def _add_surface_vars(
    data_vars: dict[str, xr.Variable],
    surface: SurfaceState,
) -> None:
    """Add surface state variables to *data_vars* (Gaussian grid, in-place)."""
    sst = np.asarray(surface.ocean.surface_temperature)
    dims: list[str] = ["lat"] if sst.ndim == 1 else ["lat", "lon"]
    data_vars["sst"] = xr.Variable(
        dims,
        sst,
        {"units": "K", "long_name": "sea surface temperature"},
    )
    if surface.land is not None:
        data_vars["soil_temperature"] = xr.Variable(
            ["lat", "lon"],
            np.asarray(surface.land.soil_temperature),
            {"units": "K", "long_name": "soil temperature"},
        )
        data_vars["bucket_depth"] = xr.Variable(
            ["lat", "lon"],
            np.asarray(surface.land.bucket_depth),
            {"units": "m", "long_name": "bucket water depth"},
        )
    if surface.ice is not None:
        data_vars["ice_thickness"] = xr.Variable(
            ["lat"],
            np.asarray(surface.ice.ice_thickness),
            {"units": "m", "long_name": "sea ice thickness"},
        )
        data_vars["ice_fraction"] = xr.Variable(
            ["lat"],
            np.asarray(surface.ice.ice_fraction),
            {"units": "1", "long_name": "sea ice fraction"},
        )


def _add_surface_vars_spectral(
    data_vars: dict[str, xr.Variable],
    surface: SurfaceState,
    transform: SpectralTransform,
    to_grid: Callable[[jnp.ndarray], jnp.ndarray],
) -> None:
    """Add surface vars regridded via spectral interpolation (in-place)."""
    n_lon = transform.grid.n_lon

    sst = np.asarray(surface.ocean.surface_temperature)
    if sst.ndim == 1:
        sst = np.broadcast_to(sst[:, None], (sst.shape[0], n_lon)).copy()
    sst_spec = transform.grid_to_spectral(jnp.array(sst))
    data_vars["sst"] = xr.Variable(
        ["lat", "lon"],
        np.asarray(to_grid(sst_spec)),
        {"units": "K", "long_name": "sea surface temperature"},
    )

    if surface.land is not None:
        for name, field, units, long_name in [
            (
                "soil_temperature",
                surface.land.soil_temperature,
                "K",
                "soil temperature",
            ),
            (
                "bucket_depth",
                surface.land.bucket_depth,
                "m",
                "bucket water depth",
            ),
        ]:
            spec = transform.grid_to_spectral(jnp.array(np.asarray(field)))
            data_vars[name] = xr.Variable(
                ["lat", "lon"],
                np.asarray(to_grid(spec)),
                {"units": units, "long_name": long_name},
            )
    if surface.ice is not None:
        data_vars["ice_thickness"] = xr.Variable(
            ["lat"],
            np.asarray(surface.ice.ice_thickness),
            {"units": "m", "long_name": "sea ice thickness"},
        )
        data_vars["ice_fraction"] = xr.Variable(
            ["lat"],
            np.asarray(surface.ice.ice_fraction),
            {"units": "1", "long_name": "sea ice fraction"},
        )


def _extract_surface(ds: xr.Dataset) -> SurfaceState | None:
    """Extract surface state from a dataset, if present."""
    if "sst" not in ds:
        return None

    ocean = OceanState(
        surface_temperature=jnp.array(ds["sst"].values),
    )
    land = None
    if "soil_temperature" in ds and "bucket_depth" in ds:
        land = LandState(
            soil_temperature=jnp.array(ds["soil_temperature"].values),
            bucket_depth=jnp.array(ds["bucket_depth"].values),
        )
    ice = None
    if "ice_thickness" in ds and "ice_fraction" in ds:
        ice = SeaIceState(
            ice_thickness=jnp.array(ds["ice_thickness"].values),
            ice_fraction=jnp.array(ds["ice_fraction"].values),
        )
    return SurfaceState(ocean=ocean, land=land, ice=ice)


# ---------------------------------------------------------------------------
# Static field helpers
# ---------------------------------------------------------------------------


def _add_static_fields(
    data_vars: dict[str, xr.Variable],
    to_grid: Callable[[jnp.ndarray], jnp.ndarray],
    planet: PlanetaryConstants,
    surface_geopotential: jnp.ndarray | None,
    reference_temperature: np.ndarray | None,
    q_flux: jnp.ndarray | None,
    surface_properties: SurfaceProperties | None,
) -> None:
    """Add static/auxiliary fields to *data_vars* (Gaussian grid, in-place)."""
    if surface_geopotential is not None:
        orog_grid = to_grid(surface_geopotential) / planet.gravity
        data_vars["orography"] = xr.Variable(
            _DIMS_2D,
            np.asarray(orog_grid),
            {"units": "m", "long_name": "surface elevation"},
        )

    if reference_temperature is not None:
        data_vars["reference_temperature"] = xr.Variable(
            ["sigma"],
            np.asarray(reference_temperature),
            {"units": "K", "long_name": "reference temperature profile"},
        )

    if q_flux is not None:
        data_vars["q_flux"] = xr.Variable(
            ["lat"],
            np.asarray(q_flux),
            {"units": "W/m2", "long_name": "prescribed ocean heat transport"},
        )

    if surface_properties is not None:
        _add_surface_properties(data_vars, surface_properties)


def _add_static_fields_spectral(
    data_vars: dict[str, xr.Variable],
    transform: SpectralTransform,
    to_grid: Callable[[jnp.ndarray], jnp.ndarray],
    planet: PlanetaryConstants,
    surface_geopotential: jnp.ndarray | None,
    reference_temperature: np.ndarray | None,
    q_flux: jnp.ndarray | None,
    surface_properties: SurfaceProperties | None,
) -> None:
    """Add static fields regridded via spectral interpolation (in-place).

    For 1D fields (reference_temperature), no regridding needed.
    For 2D fields on the Gaussian grid, spectral interpolation is used.
    """
    if surface_geopotential is not None:
        orog_regular = to_grid(surface_geopotential) / planet.gravity
        data_vars["orography"] = xr.Variable(
            _DIMS_2D,
            np.asarray(orog_regular),
            {"units": "m", "long_name": "surface elevation"},
        )

    if reference_temperature is not None:
        data_vars["reference_temperature"] = xr.Variable(
            ["sigma"],
            np.asarray(reference_temperature),
            {"units": "K", "long_name": "reference temperature profile"},
        )

    if q_flux is not None:
        # q_flux is (n_lat,) on Gaussian grid; broadcast, transform, regrid
        n_lon = transform.grid.n_lon
        q_2d = np.broadcast_to(
            np.asarray(q_flux)[:, None],
            (len(q_flux), n_lon),
        ).copy()
        q_spec = transform.grid_to_spectral(jnp.array(q_2d))
        q_regular = np.asarray(to_grid(q_spec))
        # Take zonal mean since q_flux is latitude-only
        data_vars["q_flux"] = xr.Variable(
            ["lat"],
            np.mean(q_regular, axis=-1),
            {"units": "W/m2", "long_name": "prescribed ocean heat transport"},
        )

    if surface_properties is not None:
        _add_surface_properties_spectral(data_vars, surface_properties, transform, to_grid)


def _add_surface_properties(
    data_vars: dict[str, xr.Variable],
    props: SurfaceProperties,
) -> None:
    """Add SurfaceProperties fields to *data_vars* (Gaussian grid)."""
    for name, field, units, long_name in [
        ("land_fraction", props.land_fraction, "1", "land fraction"),
        ("albedo", props.albedo, "1", "surface albedo"),
        ("z0_momentum", props.z0_momentum, "m", "roughness length for momentum"),
        ("z0_heat", props.z0_heat, "m", "roughness length for heat"),
    ]:
        data_vars[name] = xr.Variable(
            _DIMS_2D,
            np.asarray(field),
            {"units": units, "long_name": long_name},
        )


def _add_surface_properties_spectral(
    data_vars: dict[str, xr.Variable],
    props: SurfaceProperties,
    transform: SpectralTransform,
    to_grid: Callable[[jnp.ndarray], jnp.ndarray],
) -> None:
    """Add SurfaceProperties regridded via spectral interpolation."""
    for name, field, units, long_name in [
        ("land_fraction", props.land_fraction, "1", "land fraction"),
        ("albedo", props.albedo, "1", "surface albedo"),
        ("z0_momentum", props.z0_momentum, "m", "roughness length for momentum"),
        ("z0_heat", props.z0_heat, "m", "roughness length for heat"),
    ]:
        spec = transform.grid_to_spectral(jnp.array(np.asarray(field)))
        data_vars[name] = xr.Variable(
            _DIMS_2D,
            np.asarray(to_grid(spec)),
            {"units": units, "long_name": long_name},
        )


def _extract_optional_regular(
    ds: xr.Dataset,
    regrid_2d: Callable[[np.ndarray], jnp.ndarray],
    transform: SpectralTransform,
    planet: PlanetaryConstants,
    lat_in: np.ndarray,
    lon_in: np.ndarray,
) -> dict[str, jnp.ndarray | np.ndarray | SurfaceProperties | None]:
    """Extract optional fields from a regular-grid dataset after regridding."""
    result: dict[str, jnp.ndarray | np.ndarray | SurfaceProperties | None] = {}

    if "orography" in ds:
        orog = regrid_2d(np.asarray(ds["orography"].values, dtype=np.float64))
        result["surface_geopotential"] = transform.grid_to_spectral(planet.gravity * orog)

    if "reference_temperature" in ds:
        result["reference_temperature"] = np.asarray(
            ds["reference_temperature"].values,
            dtype=np.float64,
        )

    if "q_flux" in ds:
        q_1d = np.asarray(ds["q_flux"].values, dtype=np.float64)
        q_2d = np.broadcast_to(q_1d[:, None], (len(lat_in), len(lon_in))).copy()
        result["q_flux"] = jnp.mean(regrid_2d(q_2d), axis=-1)

    if "land_fraction" in ds:
        result["surface_properties"] = SurfaceProperties(
            land_fraction=regrid_2d(np.asarray(ds["land_fraction"].values, dtype=np.float64)),
            albedo=regrid_2d(np.asarray(ds["albedo"].values, dtype=np.float64)),
            z0_momentum=regrid_2d(np.asarray(ds["z0_momentum"].values, dtype=np.float64)),
            z0_heat=regrid_2d(np.asarray(ds["z0_heat"].values, dtype=np.float64)),
        )

    return result


def _extract_surface_geopotential(
    ds: xr.Dataset,
    transform: SpectralTransform,
    planet: PlanetaryConstants,
) -> jnp.ndarray | None:
    """Extract orography → spectral surface geopotential."""
    if "orography" not in ds:
        return None
    orog = jnp.array(ds["orography"].values)
    return transform.grid_to_spectral(planet.gravity * orog)


def _extract_reference_temperature(ds: xr.Dataset) -> np.ndarray | None:
    """Extract reference temperature profile."""
    if "reference_temperature" not in ds:
        return None
    return np.asarray(ds["reference_temperature"].values, dtype=np.float64)


def _extract_q_flux(
    ds: xr.Dataset,
    transform: SpectralTransform,
) -> jnp.ndarray | None:
    """Extract prescribed ocean heat transport."""
    if "q_flux" not in ds:
        return None
    return jnp.array(ds["q_flux"].values)


def _extract_surface_properties(ds: xr.Dataset) -> SurfaceProperties | None:
    """Extract SurfaceProperties from a dataset, if present."""
    if "land_fraction" not in ds:
        return None
    return SurfaceProperties(
        land_fraction=jnp.array(ds["land_fraction"].values),
        albedo=jnp.array(ds["albedo"].values),
        z0_momentum=jnp.array(ds["z0_momentum"].values),
        z0_heat=jnp.array(ds["z0_heat"].values),
    )
