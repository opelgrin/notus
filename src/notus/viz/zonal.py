"""Zonal-mean latitude-height cross-section plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure

    from notus.diagnostics import ZonalMeanState


# Defaults keyed by variable name: (cmap, levels, units_label, title)
_FIELD_DEFAULTS: dict[str, tuple[str, np.ndarray | None, str, str]] = {
    "u": ("RdBu_r", np.arange(-10, 36, 5), "m/s", "Zonal-Mean Zonal Wind"),
    "v": ("RdBu_r", np.arange(-5, 5.5, 0.5), "m/s", "Zonal-Mean Meridional Wind"),
    "temperature": ("Spectral_r", np.arange(190, 315, 10), "K", "Zonal-Mean Temperature"),
    "specific_humidity": (
        "YlGnBu",
        np.arange(0, 22, 2),
        "g/kg",
        "Zonal-Mean Specific Humidity",
    ),
    "eke": ("YlOrRd", np.arange(0, 175, 25), "m²/s²", "Eddy Kinetic Energy"),
    "uv_prime": ("RdBu_r", None, "m²/s²", "Eddy Momentum Flux [u'v']"),
    "vt_prime": ("RdBu_r", None, "K·m/s", "Eddy Heat Flux [v'T']"),
    "streamfunction": ("RdBu_r", None, "×10⁹ kg/s", "Meridional Streamfunction"),
}

_UNIT_SCALE: dict[str, float] = {
    "specific_humidity": 1000.0,  # kg/kg → g/kg
    "streamfunction": 1e-9,  # kg/s → 10⁹ kg/s
}

_DIVERGING_CMAPS = {"RdBu_r", "RdBu", "coolwarm", "bwr", "seismic", "PuOr_r", "PiYG_r"}

_N_AUTO_LEVELS = 15


def _resolve_defaults(
    field: xr.DataArray,
    cmap: str | None,
    levels: np.ndarray | int | None,
    colorbar_label: str | None,
    title: str | None,
    extend: str | None,
) -> tuple[str, np.ndarray, str, str, str]:
    """Resolve plotting defaults from field name and provided overrides."""
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

    if extend is None:
        extend = "both" if cmap in _DIVERGING_CMAPS else "max"

    return cmap, levels_arr, colorbar_label, title, extend


@overload
def plot_zonal_mean(
    field: xr.DataArray,
    *,
    ax: None = ...,
    cmap: str | None = ...,
    levels: np.ndarray | int | None = ...,
    extend: str | None = ...,
    add_colorbar: bool = ...,
    colorbar_label: str | None = ...,
    title: str | None = ...,
    contour_lines: bool = ...,
    contour_labels: bool = ...,
    zero_line: bool = ...,
    figsize: tuple[float, float] = ...,
    **contourf_kwargs: Any,
) -> tuple[Figure, Axes]: ...


@overload
def plot_zonal_mean(
    field: xr.DataArray,
    *,
    ax: Axes,
    cmap: str | None = ...,
    levels: np.ndarray | int | None = ...,
    extend: str | None = ...,
    add_colorbar: bool = ...,
    colorbar_label: str | None = ...,
    title: str | None = ...,
    contour_lines: bool = ...,
    contour_labels: bool = ...,
    zero_line: bool = ...,
    figsize: tuple[float, float] = ...,
    **contourf_kwargs: Any,
) -> QuadContourSet: ...


def plot_zonal_mean(
    field: xr.DataArray,
    *,
    ax: Axes | None = None,
    cmap: str | None = None,
    levels: np.ndarray | int | None = None,
    extend: str | None = None,
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    contour_lines: bool = True,
    contour_labels: bool = False,
    zero_line: bool = True,
    figsize: tuple[float, float] = (8, 5),
    **contourf_kwargs: Any,
) -> tuple[Figure, Axes] | QuadContourSet:
    """Plot a zonal-mean field as a latitude-sigma cross-section.

    Parameters
    ----------
    field : xr.DataArray
        2D field with ``lat`` and ``sigma`` coordinates, typically obtained
        via ``ds["temperature"].mean("lon")`` or from
        :func:`zonal_mean_to_dataset`.
    ax : Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    cmap : str, optional
        Colormap.  Auto-detected from variable name if not given.
    levels : array-like or int, optional
        Contour levels.  Auto-detected or computed from data range.
    extend : str, optional
        Colorbar extend mode.  Default: ``"both"`` for diverging cmaps,
        ``"max"`` for sequential.
    add_colorbar : bool
        Add a colorbar.  Default ``True``.
    colorbar_label : str, optional
        Colorbar label.  Auto-detected from variable name.
    title : str, optional
        Axes title.  Auto-detected from variable name.
    contour_lines : bool
        Overlay thin black contour lines.  Default ``True``.
    contour_labels : bool
        Add inline labels to contour lines.  Default ``False``.
    zero_line : bool
        Draw a thick zero contour for diverging fields.  Default ``True``.
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
    """
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    r_cmap, r_levels, r_label, r_title, r_extend = _resolve_defaults(
        field,
        cmap,
        levels,
        colorbar_label,
        title,
        extend,
    )

    lat = np.asarray(field.coords["lat"].values)
    sigma = np.asarray(field.coords["sigma"].values)
    name = field.name if isinstance(field.name, str) else ""
    scale = _UNIT_SCALE.get(name, 1.0)
    data = np.asarray(field.values) * scale

    cf = ax.contourf(
        lat,
        sigma,
        data,
        levels=r_levels,
        cmap=r_cmap,
        extend=r_extend,
        **contourf_kwargs,
    )

    if contour_lines:
        cs = ax.contour(lat, sigma, data, levels=r_levels, colors="k", linewidths=0.3)
        if contour_labels:
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    if zero_line and r_cmap in _DIVERGING_CMAPS:
        ax.contour(lat, sigma, data, levels=[0], colors="k", linewidths=1.0)

    if add_colorbar and fig is not None:
        plt.colorbar(cf, ax=ax, label=r_label)

    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("Latitude [°]")
    ax.set_ylabel("Sigma")
    ax.set_title(r_title)

    if created_fig:
        return fig, ax
    return cf


def zonal_mean_to_dataset(
    zm: ZonalMeanState,
    lat: np.ndarray,
    sigma: np.ndarray,
    *,
    streamfunction: np.ndarray | None = None,
) -> xr.Dataset:
    """Convert a :class:`~notus.diagnostics.ZonalMeanState` to xarray.

    Parameters
    ----------
    zm : ZonalMeanState
        Zonal-mean fields from
        :func:`~notus.diagnostics.compute_zonal_mean_state`.
    lat : np.ndarray
        Latitude values in degrees, shape ``(n_lat,)``.
    sigma : np.ndarray
        Sigma level values, shape ``(n_levels,)``.
    streamfunction : np.ndarray, optional
        Meridional streamfunction [kg/s], shape ``(n_levels, n_lat)``,
        from :func:`~notus.diagnostics.compute_streamfunction`.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``u``, ``v``, ``temperature``, ``eke``,
        ``u_prime_sq``, ``v_prime_sq``, ``uv_prime``, ``vt_prime``,
        and optionally ``streamfunction`` on ``(sigma, lat)`` coordinates.
    """
    dims = ["sigma", "lat"]
    coords = {
        "sigma": ("sigma", sigma, {"long_name": "sigma level", "positive": "down"}),
        "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
    }
    data_vars: dict[str, tuple[list[str], np.ndarray]] = {
        "u": (dims, np.asarray(zm.u)),
        "v": (dims, np.asarray(zm.v)),
        "temperature": (dims, np.asarray(zm.temperature)),
        "eke": (dims, np.asarray(zm.eke)),
        "u_prime_sq": (dims, np.asarray(zm.u_prime_sq)),
        "v_prime_sq": (dims, np.asarray(zm.v_prime_sq)),
        "uv_prime": (dims, np.asarray(zm.uv_prime)),
        "vt_prime": (dims, np.asarray(zm.vt_prime)),
    }
    if streamfunction is not None:
        data_vars["streamfunction"] = (dims, np.asarray(streamfunction))
    return xr.Dataset(data_vars, coords=coords)
