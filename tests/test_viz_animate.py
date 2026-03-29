"""Tests for animation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from notus.viz.animate import animate_field


pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning",
)


def _make_zonal_frame(scale: float) -> xr.DataArray:
    """Create a small zonal-mean-like frame for animation tests."""
    lat = np.linspace(-90.0, 90.0, 9)
    sigma = np.linspace(0.1, 1.0, 5)
    data = scale * np.outer(np.linspace(1.0, 2.0, sigma.size), np.cos(np.deg2rad(lat)))
    return xr.DataArray(
        data,
        dims=("sigma", "lat"),
        coords={"sigma": sigma, "lat": lat},
        name="temperature",
    )


def test_animate_field_does_not_add_colorbar_axes_per_frame() -> None:
    """Animation updates should not keep adding colorbar axes."""
    frames = [_make_zonal_frame(1.0), _make_zonal_frame(1.2), _make_zonal_frame(0.8)]
    fig, anim = animate_field(frames, plot_fn="zonal")
    n_axes_initial = len(fig.axes)

    anim._func(1)  # type: ignore[attr-defined]
    n_axes_after_first_update = len(fig.axes)
    anim._func(2)  # type: ignore[attr-defined]
    n_axes_after_second_update = len(fig.axes)

    assert n_axes_initial == 2
    assert n_axes_after_first_update == n_axes_initial
    assert n_axes_after_second_update == n_axes_initial
