"""Spectral differential operators on the sphere."""

from notus.operators.caches import (
    _laplacian_eigenvalues,
    _m_index_array,
    _meridional_coupling,
    _mu_derivative_coupling,
)
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
    # Private cache names re-exported for internal consumers
    "_laplacian_eigenvalues",
    "_m_index_array",
    "_meridional_coupling",
    "_mu_derivative_coupling",
    # Public operators
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
