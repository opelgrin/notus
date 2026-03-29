"""Hovmöller (longitude-time and latitude-time) diagram plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure


_DIVERGING_CMAPS = {"RdBu_r", "RdBu", "coolwarm", "bwr", "seismic", "PuOr_r", "PiYG_r"}

_N_AUTO_LEVELS = 15


@overload
def plot_hovmoller(
    field: xr.DataArray,
    *,
    ax: None = ...,
    cmap: str | None = ...,
    levels: np.ndarray | int | None = ...,
    extend: str = ...,
    add_colorbar: bool = ...,
    colorbar_label: str | None = ...,
    title: str | None = ...,
    figsize: tuple[float, float] = ...,
    **contourf_kwargs: Any,
) -> tuple[Figure, Axes]: ...


@overload
def plot_hovmoller(
    field: xr.DataArray,
    *,
    ax: Axes,
    cmap: str | None = ...,
    levels: np.ndarray | int | None = ...,
    extend: str = ...,
    add_colorbar: bool = ...,
    colorbar_label: str | None = ...,
    title: str | None = ...,
    figsize: tuple[float, float] = ...,
    **contourf_kwargs: Any,
) -> QuadContourSet: ...


def plot_hovmoller(
    field: xr.DataArray,
    *,
    ax: Axes | None = None,
    cmap: str | None = None,
    levels: np.ndarray | int | None = None,
    extend: str = "both",
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    **contourf_kwargs: Any,
) -> tuple[Figure, Axes] | QuadContourSet:
    """Plot a Hovmöller (space-time) diagram.

    The field must be 2D with one spatial dimension (``lat`` or ``lon``)
    and one time-like dimension (any other dimension, typically from
    stacking snapshots).

    Parameters
    ----------
    field : xr.DataArray
        2D field, e.g. shape ``(time, lon)`` for a longitude-time diagram,
        or ``(time, lat)`` for a latitude-time diagram.
    ax : Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    cmap : str, optional
        Colormap.  Default ``"RdBu_r"``.
    levels : array-like or int, optional
        Contour levels.  Computed from data range if not given.
    extend : str
        Colorbar extend mode.  Default ``"both"``.
    add_colorbar : bool
        Add a colorbar.  Default ``True``.
    colorbar_label : str, optional
        Colorbar label.
    title : str, optional
        Axes title.
    figsize : tuple
        Figure size when creating a new figure.
    **contourf_kwargs
        Passed to ``ax.contourf``.

    Returns
    -------
    tuple[Figure, Axes]
        When *ax* is ``None``.
    QuadContourSet
        When *ax* is provided.

    Examples
    --------
    Longitude-time Hovmöller at a fixed latitude:

    >>> # snapshots: list of DataArrays with dims (lat, lon)
    >>> hov = xr.concat(snapshots, dim="day")
    >>> hov_eq = hov.sel(lat=0, method="nearest")  # equatorial slice
    >>> fig, ax = plot_hovmoller(hov_eq, title="Equatorial Hovmöller")

    Latitude-time Hovmöller of zonal-mean field:

    >>> hov_zm = xr.concat([s.mean("lon") for s in snapshots], dim="day")
    >>> fig, ax = plot_hovmoller(hov_zm, cmap="Spectral_r")
    """
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Identify spatial and time dimensions
    spatial_dim, time_dim = _identify_dims(field)
    spatial_vals = np.asarray(field.coords[spatial_dim].values)
    time_vals = np.asarray(field.coords[time_dim].values, dtype=np.float64)
    data = np.asarray(field.values)

    # Ensure data is (time, space)
    dim_order = list(field.dims)
    if dim_order[0] != time_dim:
        data = data.T

    cmap = cmap or "RdBu_r"
    if levels is None:
        levels_arr = np.linspace(
            float(np.nanmin(data)),
            float(np.nanmax(data)),
            _N_AUTO_LEVELS,
        )
    elif isinstance(levels, int):
        levels_arr = np.linspace(
            float(np.nanmin(data)),
            float(np.nanmax(data)),
            levels,
        )
    else:
        levels_arr = np.asarray(levels)

    cf = ax.contourf(
        spatial_vals,
        time_vals,
        data,
        levels=levels_arr,
        cmap=cmap,
        extend=extend,
        **contourf_kwargs,
    )

    if add_colorbar and fig is not None:
        label = colorbar_label or ""
        plt.colorbar(cf, ax=ax, label=label)

    # Axis labels
    if spatial_dim == "lon":
        ax.set_xlabel("Longitude [°]")
    elif spatial_dim == "lat":
        ax.set_xlabel("Latitude [°]")
    else:
        ax.set_xlabel(spatial_dim)
    ax.set_ylabel(time_dim.capitalize())

    if title is not None:
        ax.set_title(title)

    if created_fig:
        return fig, ax
    return cf


def _identify_dims(field: xr.DataArray) -> tuple[str, str]:
    """Identify (spatial_dim, time_dim) from a 2D DataArray."""
    dims = list(field.dims)
    expected_ndim = 2
    if len(dims) != expected_ndim:
        msg = f"Hovmöller plot requires a 2D DataArray, got {len(dims)}D with dims {dims}"
        raise ValueError(msg)

    spatial_dims = {"lat", "lon", "sigma"}
    spatial = [d for d in dims if d in spatial_dims]
    non_spatial = [d for d in dims if d not in spatial_dims]

    if len(spatial) == 1 and len(non_spatial) == 1:
        return str(spatial[0]), str(non_spatial[0])

    # Fallback: assume first dim is time, second is space
    return str(dims[1]), str(dims[0])
