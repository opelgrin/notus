"""Sigma (p/pₛ) vertical coordinate on a Lorenz grid.

The sigma coordinate maps pressure to the range [0, 1]:

    σ = p / pₛ

where pₛ is surface pressure.  σ = 0 at the top of atmosphere and σ = 1
at the surface.

On a **Lorenz grid**, prognostic variables (T, ζ, δ) live at **full levels**
(midpoints of each layer), while the vertical velocity σ̇ lives at **half
levels** (layer interfaces).  This is the standard staggering for spectral
GCMs (SPEEDY, SpeedyWeather, Dinosaur/NeuralGCM).

Indexing convention (top to bottom):

    half level 0:    σ = 0       (top of atmosphere)
    full level 0:    σ_full[0]   (topmost layer midpoint)
    half level 1:    σ_half[1]
    full level 1:    σ_full[1]
    ...
    half level L-1:  σ_half[L-1]
    full level L-1:  σ_full[L-1] (lowest layer midpoint)
    half level L:    σ = 1       (surface)
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class SigmaLevels:
    """Sigma vertical coordinate definition.

    Parameters
    ----------
    n_levels : int
        Number of full levels (layers).
    sigma_half : jnp.ndarray
        Interface (half-level) sigma values, shape ``(n_levels + 1,)``.
        Monotonically increasing from 0 (top) to 1 (surface).
    sigma_full : jnp.ndarray
        Full-level (midpoint) sigma values, shape ``(n_levels,)``.
        Each value lies between its bounding interfaces.
    dsigma : jnp.ndarray
        Layer thicknesses Δσ_k = σ_{k+1/2} - σ_{k-1/2}, shape ``(n_levels,)``.
        Sums to 1.
    """

    n_levels: int
    sigma_half: jnp.ndarray
    sigma_full: jnp.ndarray
    dsigma: jnp.ndarray

    def __init__(self, sigma_half: np.ndarray) -> None:
        """Construct from half-level (interface) sigma values.

        Parameters
        ----------
        sigma_half : np.ndarray
            Interface sigma values, shape ``(n_levels + 1,)``, float64.
            Must be monotonically increasing from 0 to 1.
        """
        min_interfaces = 2
        if sigma_half.ndim != 1 or len(sigma_half) < min_interfaces:
            msg = f"sigma_half must be 1-D with at least 2 elements, got shape {sigma_half.shape}"
            raise ValueError(msg)
        if not np.isclose(sigma_half[0], 0.0):
            msg = f"sigma_half must start at 0 (top of atmosphere), got {sigma_half[0]}"
            raise ValueError(msg)
        if not np.isclose(sigma_half[-1], 1.0):
            msg = f"sigma_half must end at 1 (surface), got {sigma_half[-1]}"
            raise ValueError(msg)
        if not np.all(np.diff(sigma_half) > 0):
            msg = "sigma_half must be strictly monotonically increasing"
            raise ValueError(msg)

        n_levels = len(sigma_half) - 1
        sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
        dsigma = np.diff(sigma_half)

        object.__setattr__(self, "n_levels", n_levels)
        object.__setattr__(self, "sigma_half", jnp.array(sigma_half))
        object.__setattr__(self, "sigma_full", jnp.array(sigma_full))
        object.__setattr__(self, "dsigma", jnp.array(dsigma))


def uniform_sigma_levels(n_levels: int) -> SigmaLevels:
    """Create equally spaced sigma levels.

    Parameters
    ----------
    n_levels : int
        Number of full levels (must be >= 1).

    Returns
    -------
    SigmaLevels
        Uniform vertical coordinate with Δσ = 1/n_levels for all layers.
    """
    if n_levels < 1:
        msg = f"n_levels must be >= 1, got {n_levels}"
        raise ValueError(msg)
    sigma_half = np.linspace(0.0, 1.0, n_levels + 1)
    return SigmaLevels(sigma_half)


def standard_sigma_levels(n_levels: int = 20) -> SigmaLevels:
    """Create sigma levels with enhanced resolution near the surface and tropopause.

    Builds the layer thicknesses ``Δσ`` from a smooth weighting profile
    and normalizes them to sum to 1.  The profile increases resolution
    in three physically important regions:

    - upper atmosphere (small σ) for wave propagation and stability,
    - tropopause neighborhood (σ ≈ 0.2),
    - planetary boundary layer near the surface (σ > 0.8).

    This yields a practical "standard" vertical coordinate for idealized
    spectral GCM benchmarks (Held-Suarez, aquaplanet) while keeping the
    implementation deterministic and monotone for any ``n_levels >= 1``.

    Parameters
    ----------
    n_levels : int
        Number of full levels (default: 20, standard for Held-Suarez).

    Returns
    -------
    SigmaLevels
        Vertically stretched sigma coordinate.
    """
    if n_levels < 1:
        msg = f"n_levels must be >= 1, got {n_levels}"
        raise ValueError(msg)

    # Base index in [0, 1] and layer centers.
    eta_half = np.linspace(0.0, 1.0, n_levels + 1)
    eta_mid = 0.5 * (eta_half[:-1] + eta_half[1:])

    # Stretching profile from smooth local refinements.
    # Larger weight -> smaller Δσ in that region.
    weight = (
        1.0
        + 3.0 * np.exp(-(((eta_mid - 0.03) / 0.08) ** 2))  # upper atmosphere
        + 2.2 * np.exp(-(((eta_mid - 0.20) / 0.10) ** 2))  # tropopause
        + 3.5 * np.exp(-(((eta_mid - 0.92) / 0.08) ** 2))  # surface layer
    )
    dsigma = 1.0 / weight
    dsigma /= np.sum(dsigma)

    sigma_half = np.concatenate([[0.0], np.cumsum(dsigma)])
    sigma_half[-1] = 1.0  # exact endpoint after floating-point cumsum

    return SigmaLevels(sigma_half)
