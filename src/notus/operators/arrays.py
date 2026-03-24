"""Pre-computed spectral operator arrays.

:class:`OperatorArrays` replaces the module-level ``lru_cache`` functions in
:mod:`notus.operators.caches`.  All expensive index and eigenvalue arrays are
computed once at construction time and stored as instance attributes, giving
explicit, instance-scoped storage instead of a hidden global cache.

Typical usage::

    arrays = OperatorArrays.build(grid.truncation, planet.radius)
    # or, via SpectralTransform:
    arrays = transform.arrays
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from notus.operators.caches import (
    _compute_laplacian_eigenvalues,
    _compute_m_index_array,
    _compute_meridional_coupling,
    _compute_mu_derivative_coupling,
    _compute_n_index_array,
)


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorArrays:
    """Pre-computed spectral operator arrays for a given truncation and radius.

    All arrays are computed at construction time via :meth:`build` and stored
    as immutable attributes.  Pass this object to operator functions instead
    of ``(truncation, radius)`` or ``(truncation,)`` pairs.

    Attributes
    ----------
    truncation : int
        Triangular truncation T.
    radius : float
        Planet radius [m].
    laplacian_eigenvalues : jnp.ndarray
        ``-n(n+1)/a²`` for every spectral index, shape ``(n_spectral,)``.
    m_index : jnp.ndarray
        Zonal wavenumber m for every spectral index, shape ``(n_spectral,)``.
    n_index : jnp.ndarray
        Total wavenumber n for every spectral index, shape ``(n_spectral,)``.
    meridional_coupling : tuple
        Four ``(n_spectral,)`` arrays ``(idx_lower, idx_upper, c_lower, c_upper)``
        for the cos(φ)·∂/∂φ recurrence.
    mu_derivative_coupling : tuple
        Four ``(n_spectral,)`` arrays ``(idx_lower, idx_upper, c_lower, c_upper)``
        for the d/dμ recurrence used by spectral div/curl.
    """

    truncation: int
    radius: float
    laplacian_eigenvalues: jnp.ndarray
    m_index: jnp.ndarray
    n_index: jnp.ndarray
    meridional_coupling: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    mu_derivative_coupling: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

    @classmethod
    def build(cls, truncation: int, radius: float) -> OperatorArrays:
        """Compute all operator arrays for the given truncation and radius.

        Parameters
        ----------
        truncation : int
            Triangular truncation T.
        radius : float
            Planet radius [m].

        Returns
        -------
        OperatorArrays
        """
        return cls(
            truncation=truncation,
            radius=radius,
            laplacian_eigenvalues=_compute_laplacian_eigenvalues(truncation, radius),
            m_index=_compute_m_index_array(truncation),
            n_index=_compute_n_index_array(truncation),
            meridional_coupling=_compute_meridional_coupling(truncation),
            mu_derivative_coupling=_compute_mu_derivative_coupling(truncation),
        )
