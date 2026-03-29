"""Animation helpers for time-evolving fields."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import xarray as xr


if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def animate_field(
    frames: Sequence[xr.DataArray] | xr.DataArray,
    *,
    plot_fn: str = "map",
    interval: int = 200,
    title_fmt: str = "Day {i}",
    figsize: tuple[float, float] = (10, 5),
    **plot_kwargs: Any,
) -> tuple[Figure, FuncAnimation]:
    """Create an animation from a sequence of 2D fields.

    Parameters
    ----------
    frames : sequence of DataArray, or DataArray with a leading dim
        Fields to animate.  If a single DataArray with 3+ dims, the first
        non-``lat``/``lon``/``sigma`` dimension is iterated over.
        If a sequence, each element is one frame.
    plot_fn : str
        Plotting function: ``"map"`` (default) uses
        :func:`~notus.viz.maps.plot_map`, ``"zonal"`` uses
        :func:`~notus.viz.zonal.plot_zonal_mean`.
    interval : int
        Delay between frames in milliseconds.
    title_fmt : str
        Format string for the title.  ``{i}`` is the frame index.
        ``{value}`` is the coordinate value on the iterated dimension
        (if available).
    figsize : tuple
        Figure size.
    **plot_kwargs
        Passed to the underlying plot function (e.g., ``cmap``, ``levels``).

    Returns
    -------
    tuple[Figure, FuncAnimation]
        The figure and animation.  Call ``anim.save("out.gif",
        writer="pillow")`` or ``anim.save("out.mp4")`` to export.

    Examples
    --------
    >>> snapshots = [state_to_dataset(s, ...) for s in states]
    >>> ps_frames = [ds["surface_pressure"] for ds in snapshots]
    >>> fig, anim = animate_field(ps_frames)
    >>> anim.save("spinup.gif", writer="pillow", fps=5)
    """
    from matplotlib.animation import FuncAnimation as FuncAnim

    from notus.viz.maps import plot_map
    from notus.viz.zonal import plot_zonal_mean

    # Normalise frames to a list of DataArrays
    frame_list: list[xr.DataArray]
    coord_values: list[float | None] = []

    min_spatial_dims = 2
    if isinstance(frames, xr.DataArray) and frames.ndim > min_spatial_dims:
        # Find the "extra" dimension to iterate over
        spatial_dims = {"lat", "lon", "sigma"}
        extra_dims = [d for d in frames.dims if d not in spatial_dims]
        if not extra_dims:
            msg = "DataArray must have a non-spatial dimension to animate over"
            raise ValueError(msg)
        iter_dim = extra_dims[0]
        frame_list = [frames.isel({iter_dim: i}) for i in range(frames.sizes[iter_dim])]
        for i in range(frames.sizes[iter_dim]):
            val = frames.coords[iter_dim].values[i]
            coord_values.append(float(val))
    else:
        frame_list = list(frames)
        coord_values = [None] * len(frame_list)

    if not frame_list:
        msg = "No frames provided"
        raise ValueError(msg)

    # Select plot function
    if plot_fn == "map":
        plotter: Any = plot_map
    elif plot_fn == "zonal":
        plotter = plot_zonal_mean
    else:
        msg = f"Unknown plot_fn: {plot_fn!r}. Use 'map' or 'zonal'."
        raise ValueError(msg)

    # Create figure from first frame
    plot_kwargs["add_colorbar"] = plot_kwargs.get("add_colorbar", True)
    fig, ax = plotter(frame_list[0], **plot_kwargs)
    _set_frame_title(ax, title_fmt, 0, coord_values[0])

    def _update(i: int) -> list[Any]:
        ax.clear()
        plotter(frame_list[i], ax=ax, **plot_kwargs)
        _set_frame_title(ax, title_fmt, i, coord_values[i])
        return []

    anim = FuncAnim(
        fig,
        _update,
        frames=len(frame_list),
        interval=interval,
        blit=False,
    )
    return fig, anim


def _set_frame_title(
    ax: Axes,
    fmt: str,
    i: int,
    value: float | None,
) -> None:
    """Set the axes title for a given frame."""
    kwargs: dict[str, float | int] = {"i": i}
    if value is not None:
        kwargs["value"] = value
    try:
        ax.set_title(fmt.format(**kwargs))
    except KeyError:
        ax.set_title(fmt.format(i=i))
