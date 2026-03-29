"""Notus visualization utilities.

Convenience plotting functions for xarray Datasets produced by
:func:`~notus.xarray.state_to_dataset` and
:func:`~notus.xarray.to_regular_latlon`.

All functions accept xarray DataArrays and return matplotlib objects
for composability.
"""

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
