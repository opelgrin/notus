"""Global map plots for 2D lat/lon fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure


# Defaults keyed by variable name: (cmap, levels, units_label, title)
_FIELD_DEFAULTS: dict[str, tuple[str, np.ndarray | None, str, str]] = {
    "surface_pressure": ("RdBu_r", np.arange(970, 1035, 5), "hPa", "Surface Pressure"),
    "temperature": ("Spectral_r", np.arange(190, 315, 10), "K", "Temperature"),
    "orography": ("terrain", None, "m", "Orography"),
    "sst": ("coolwarm", np.arange(270, 310, 2), "K", "Sea Surface Temperature"),
    "specific_humidity": ("YlGnBu", None, "g/kg", "Specific Humidity"),
    "land_fraction": ("YlGn", np.linspace(0, 1, 11), "", "Land Fraction"),
    "albedo": ("Greys_r", np.linspace(0, 0.6, 13), "", "Surface Albedo"),
    "u": ("RdBu_r", None, "m/s", "Zonal Wind"),
    "v": ("RdBu_r", None, "m/s", "Meridional Wind"),
}

# Unit conversions: (scale, offset) → display_value = raw_value * scale + offset
_UNIT_SCALE: dict[str, float] = {
    "surface_pressure": 0.01,  # Pa → hPa
    "specific_humidity": 1000.0,  # kg/kg → g/kg
}

_DIVERGING_CMAPS = {"RdBu_r", "RdBu", "coolwarm", "bwr", "seismic", "PuOr_r", "PiYG_r"}

_N_AUTO_LEVELS = 15


def _resolve_defaults(
    field: xr.DataArray,
    cmap: str | None,
    levels: np.ndarray | int | None,
    colorbar_label: str | None,
    title: str | None,
) -> tuple[str, np.ndarray, str, str, np.ndarray]:
    """Resolve plotting defaults and return (cmap, levels, label, title, data)."""
    name = field.name if isinstance(field.name, str) else ""
    defaults = _FIELD_DEFAULTS.get(name, ("viridis", None, "", ""))
    d_cmap, d_levels, d_label, d_title = defaults

    cmap = cmap or d_cmap
    title = title or d_title or name
    colorbar_label = colorbar_label or d_label

    scale = _UNIT_SCALE.get(name, 1.0)
    data = np.asarray(field.values) * scale

    if levels is None:
        if d_levels is not None:
            levels_arr = d_levels
        else:
            levels_arr = np.linspace(float(np.nanmin(data)), float(np.nanmax(data)), _N_AUTO_LEVELS)
    elif isinstance(levels, int):
        levels_arr = np.linspace(float(np.nanmin(data)), float(np.nanmax(data)), levels)
    else:
        levels_arr = np.asarray(levels)

    return cmap, levels_arr, colorbar_label, title, data


@overload
def plot_map(
    field: xr.DataArray,
    *,
    ax: None = ...,
    projection: Any = ...,
    cmap: str | None = ...,
    levels: np.ndarray | int | None = ...,
    extend: str = ...,
    add_colorbar: bool = ...,
    colorbar_label: str | None = ...,
    title: str | None = ...,
    coastlines: bool = ...,
    contour_lines: bool = ...,
    figsize: tuple[float, float] = ...,
    **contourf_kwargs: Any,
) -> tuple[Figure, Axes]: ...


@overload
def plot_map(
    field: xr.DataArray,
    *,
    ax: Axes,
    projection: Any = ...,
    cmap: str | None = ...,
    levels: np.ndarray | int | None = ...,
    extend: str = ...,
    add_colorbar: bool = ...,
    colorbar_label: str | None = ...,
    title: str | None = ...,
    coastlines: bool = ...,
    contour_lines: bool = ...,
    figsize: tuple[float, float] = ...,
    **contourf_kwargs: Any,
) -> QuadContourSet: ...


def plot_map(
    field: xr.DataArray,
    *,
    ax: Axes | None = None,
    projection: Any = None,
    cmap: str | None = None,
    levels: np.ndarray | int | None = None,
    extend: str = "both",
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    coastlines: bool = True,
    contour_lines: bool = True,
    figsize: tuple[float, float] = (10, 5),
    **contourf_kwargs: Any,
) -> tuple[Figure, Axes] | QuadContourSet:
    """Plot a 2D lat/lon field on a map projection.

    Parameters
    ----------
    field : xr.DataArray
        2D field with ``lat`` and ``lon`` coordinates.  If the field has a
        ``sigma`` dimension, select a level first with ``.sel()`` or
        ``.isel()``.
    ax : Axes, optional
        Axes to plot on (must have a cartopy projection if using cartopy).
        If ``None``, a new figure with *projection* is created.
    projection : cartopy.crs.Projection, optional
        Map projection for the new figure.  Default: ``ccrs.Mollweide()``.
        Ignored when *ax* is provided.
    cmap : str, optional
        Colormap.  Auto-detected from variable name.
    levels : array-like or int, optional
        Contour levels.  Auto-detected or computed from data range.
    extend : str
        Colorbar extend mode.  Default ``"both"``.
    add_colorbar : bool
        Add a colorbar.  Default ``True``.
    colorbar_label : str, optional
        Colorbar label.  Auto-detected.
    title : str, optional
        Axes title.  Auto-detected.
    coastlines : bool
        Draw coastlines (requires cartopy GeoAxes).  Default ``True``.
    contour_lines : bool
        Overlay thin black contour lines.  Default ``True``.
    figsize : tuple
        Figure size when creating a new figure.
    **contourf_kwargs
        Passed to ``ax.contourf``.

    Returns
    -------
    tuple[Figure, Axes]
        When *ax* is ``None`` (new figure created).
    QuadContourSet
        When *ax* is provided.

    Examples
    --------
    >>> ds = state_to_dataset(state, transform, EARTH, levels)
    >>> fig, ax = plot_map(ds["surface_pressure"])

    Multi-panel with cartopy:

    >>> import cartopy.crs as ccrs
    >>> fig, axes = plt.subplots(1, 2, subplot_kw={"projection": ccrs.Robinson()})
    >>> plot_map(ds["sst"], ax=axes[0])
    >>> plot_map(ds["orography"], ax=axes[1])
    """
    import cartopy.crs as ccrs

    created_fig = ax is None
    if ax is None:
        if projection is None:
            projection = ccrs.Mollweide()
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection},
            figsize=figsize,
        )
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    r_cmap, r_levels, r_label, r_title, data = _resolve_defaults(
        field,
        cmap,
        levels,
        colorbar_label,
        title,
    )

    lat = np.asarray(field.coords["lat"].values)
    lon = np.asarray(field.coords["lon"].values)
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    cf = ax.contourf(
        lon_2d,
        lat_2d,
        data,
        levels=r_levels,
        cmap=r_cmap,
        extend=extend,
        transform=ccrs.PlateCarree(),
        **contourf_kwargs,
    )

    if contour_lines:
        ax.contour(
            lon_2d,
            lat_2d,
            data,
            levels=r_levels,
            colors="k",
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
        )

    if coastlines and hasattr(ax, "coastlines"):
        ax.coastlines(linewidth=0.5, color="gray")

    if hasattr(ax, "set_global"):
        ax.set_global()

    if add_colorbar and fig is not None:
        plt.colorbar(
            cf,
            ax=ax,
            label=r_label,
            orientation="horizontal",
            pad=0.05,
            shrink=0.8,
        )

    ax.set_title(r_title)

    if created_fig:
        return fig, ax
    return cf
