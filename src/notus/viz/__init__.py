"""Notus visualization utilities.

Convenience plotting functions for xarray Datasets produced by
:func:`~notus.xarray.state_to_dataset` and
:func:`~notus.xarray.to_regular_latlon`.

All functions accept xarray DataArrays and return matplotlib objects
for composability.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from notus.viz.animate import animate_field
    from notus.viz.hovmoller import plot_hovmoller
    from notus.viz.maps import plot_map
    from notus.viz.spectra import plot_spectrum
    from notus.viz.zonal import plot_zonal_mean, zonal_mean_to_dataset


__all__ = [
    "animate_field",
    "plot_hovmoller",
    "plot_map",
    "plot_spectrum",
    "plot_zonal_mean",
    "zonal_mean_to_dataset",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "animate_field": ("notus.viz.animate", "animate_field"),
    "plot_hovmoller": ("notus.viz.hovmoller", "plot_hovmoller"),
    "plot_map": ("notus.viz.maps", "plot_map"),
    "plot_spectrum": ("notus.viz.spectra", "plot_spectrum"),
    "plot_zonal_mean": ("notus.viz.zonal", "plot_zonal_mean"),
    "zonal_mean_to_dataset": ("notus.viz.zonal", "zonal_mean_to_dataset"),
}


def __getattr__(name: str) -> Callable[..., object]:
    """Lazily resolve plotting helpers so matplotlib/cartopy stay optional."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attr_name = target
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        msg = (
            "Visualization dependencies are optional. Install dev extras "
            "(e.g. matplotlib/cartopy) to use notus.viz plotting helpers."
        )
        raise ModuleNotFoundError(msg) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily-resolved plotting helpers in dir()."""
    return sorted(set(globals()) | set(__all__))
