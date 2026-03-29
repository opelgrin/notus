"""Kinetic energy spectrum plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import matplotlib.pyplot as plt
import numpy as np


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


@overload
def plot_spectrum(
    wavenumber: np.ndarray,
    spectrum: np.ndarray,
    *,
    ax: None = ...,
    reference_slopes: bool = ...,
    title: str = ...,
    ylabel: str = ...,
    figsize: tuple[float, float] = ...,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]: ...


@overload
def plot_spectrum(
    wavenumber: np.ndarray,
    spectrum: np.ndarray,
    *,
    ax: Axes,
    reference_slopes: bool = ...,
    title: str = ...,
    ylabel: str = ...,
    figsize: tuple[float, float] = ...,
    **plot_kwargs: Any,
) -> Line2D: ...


def plot_spectrum(
    wavenumber: np.ndarray,
    spectrum: np.ndarray,
    *,
    ax: Axes | None = None,
    reference_slopes: bool = True,
    title: str = "Kinetic Energy Spectrum",
    ylabel: str = "Energy [m²/s²]",
    figsize: tuple[float, float] = (7, 5),
    **plot_kwargs: Any,
) -> tuple[Figure, Axes] | Line2D:
    """Plot a kinetic energy spectrum on a log-log axes.

    Parameters
    ----------
    wavenumber : np.ndarray
        Total wavenumber n, shape ``(N,)``.
    spectrum : np.ndarray
        Energy per wavenumber, shape ``(N,)``.
    ax : Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    reference_slopes : bool
        Draw n⁻³ and n⁻⁵ʹ³ reference lines.  Default ``True``.
    title : str
        Axes title.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size when creating a new figure.
    **plot_kwargs
        Passed to ``ax.loglog``.

    Returns
    -------
    tuple[Figure, Axes]
        When *ax* is ``None``.
    Line2D
        When *ax* is provided.

    Examples
    --------
    >>> wn, ke = compute_ke_spectrum(state, transform, levels)
    >>> fig, ax = plot_spectrum(wn, ke)
    """
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    plot_kwargs.setdefault("color", "k")
    plot_kwargs.setdefault("linewidth", 1.5)
    (line,) = ax.loglog(wavenumber, spectrum, **plot_kwargs)

    if reference_slopes:
        _add_reference_slopes(ax, wavenumber, spectrum)

    ax.set_xlabel("Total Wavenumber n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    if created_fig:
        return fig, ax
    return line


def _add_reference_slopes(
    ax: Axes,
    wavenumber: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    """Draw n⁻³ and n⁻⁵ʹ³ reference lines."""
    # Anchor at a point near the peak of the spectrum
    peak_idx = np.argmax(spectrum)
    n_ref = max(wavenumber[peak_idx], wavenumber[len(wavenumber) // 4])
    e_ref = spectrum[np.searchsorted(wavenumber, n_ref)]

    n_range = wavenumber[wavenumber >= n_ref]
    min_points = 2
    if len(n_range) < min_points:
        return

    # n⁻³ line
    slope_3 = e_ref * (n_range / n_ref) ** (-3)
    ax.loglog(n_range, slope_3, "--", color="0.5", linewidth=0.8, label="n⁻³")

    # n⁻⁵ʹ³ line
    slope_53 = e_ref * (n_range / n_ref) ** (-5.0 / 3.0)
    ax.loglog(n_range, slope_53, ":", color="0.5", linewidth=0.8, label="n⁻⁵ᐟ³")

    ax.legend(fontsize=9, loc="lower left")
