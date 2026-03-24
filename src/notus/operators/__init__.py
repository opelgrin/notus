"""Spectral differential operators on the sphere."""

from notus.operators.arrays import OperatorArrays
from notus.operators.core import (
    exponential_filter,
    hyperdiffusion,
    inverse_laplacian,
    laplacian,
    meridional_derivative,
    zonal_derivative,
)
from notus.operators.vector import spectral_curl, spectral_divergence, uv_from_vordiv


__all__ = [
    "OperatorArrays",
    "exponential_filter",
    "hyperdiffusion",
    "inverse_laplacian",
    "laplacian",
    "meridional_derivative",
    "spectral_curl",
    "spectral_divergence",
    "uv_from_vordiv",
    "zonal_derivative",
]
