"""Physics forcing protocols and implementations.

Defines the interface that all forcing functions must satisfy, plus the
Held-Suarez (1994) forcing for the standard benchmark.

Protocols
---------
- :class:`Forcing` — base protocol (required).
- :class:`ImplicitForcing` — extends with implicit boundary-layer correction.
- :class:`MoistForcing` — extends with reference humidity for semi-implicit moisture.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from notus.constants import PlanetaryConstants
from notus.state import PrimitiveEquationState
from notus.transforms import SpectralTransform
from notus.vertical.sigma import SigmaLevels


@dataclasses.dataclass(frozen=True, slots=True)
class PhysicsDiagnostics:
    """Diagnostic fields produced by physics parameterizations.

    All flux fields are grid-space arrays with shape ``(n_lat, n_lon)``,
    or ``None`` when not available from the forcing scheme.

    Sign convention: positive = downward / into the surface.

    Attributes
    ----------
    precipitation : jnp.ndarray or None
        Surface precipitation rate [kg/m²/s].
    evaporation : jnp.ndarray or None
        Surface evaporation rate [kg/m²/s].
    olr : jnp.ndarray or None
        Outgoing longwave radiation at TOA [W/m²].
    sw_down_surface : jnp.ndarray or None
        Downward shortwave flux at the surface [W/m²].
    lw_down_surface : jnp.ndarray or None
        Downward longwave flux at the surface [W/m²].
    sensible_heat_flux : jnp.ndarray or None
        Surface sensible heat flux [W/m²].
    latent_heat_flux : jnp.ndarray or None
        Surface latent heat flux [W/m²].
    """

    precipitation: jnp.ndarray | None = None
    evaporation: jnp.ndarray | None = None
    olr: jnp.ndarray | None = None
    sw_down_surface: jnp.ndarray | None = None
    lw_down_surface: jnp.ndarray | None = None
    sensible_heat_flux: jnp.ndarray | None = None
    latent_heat_flux: jnp.ndarray | None = None


def _diag_flatten(
    d: PhysicsDiagnostics,
) -> tuple[tuple[jnp.ndarray | None, ...], None]:
    return (
        (
            d.precipitation,
            d.evaporation,
            d.olr,
            d.sw_down_surface,
            d.lw_down_surface,
            d.sensible_heat_flux,
            d.latent_heat_flux,
        ),
        None,
    )


def _diag_unflatten(
    _aux: None,
    children: tuple[jnp.ndarray | None, ...],
) -> PhysicsDiagnostics:
    return PhysicsDiagnostics(*children)


jax.tree_util.register_pytree_node(PhysicsDiagnostics, _diag_flatten, _diag_unflatten)


class Forcing(Protocol):
    """Protocol for physics forcing functions.

    A forcing function receives the current model state and returns
    tendencies and diagnostic fields.  The tendencies are added to the
    explicit dynamics tendencies at each time step.

    See Also
    --------
    ImplicitForcing : Extension with implicit boundary-layer correction.
    MoistForcing : Extension with reference humidity for semi-implicit moisture.
    """

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> tuple[PrimitiveEquationState, PhysicsDiagnostics]:
        """Compute physics tendencies and diagnostics.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field pₛ (grid space), shape ``(n_lat, n_lon)``.

        Returns
        -------
        tuple[PrimitiveEquationState, PhysicsDiagnostics]
            Tendencies (spectral) and diagnostic fields (grid).
        """
        ...


@runtime_checkable
class ImplicitForcing(Forcing, Protocol):
    """Protocol for forcings that also provide implicit boundary-layer correction.

    Extends :class:`Forcing` with an ``apply_implicit`` method for treating
    stiff surface-layer processes (Rayleigh drag, surface heat/moisture fluxes)
    with implicit exponential decay.  Called by
    :func:`~notus.timestepping.imex.build_pe_stepper` after the IMEX step.
    """

    def apply_implicit(
        self,
        state: PrimitiveEquationState,
        dt_implicit: float,
    ) -> tuple[PrimitiveEquationState, PhysicsDiagnostics]:
        """Apply implicit physics corrections after the IMEX step.

        Parameters
        ----------
        state : PrimitiveEquationState
            State after the explicit + semi-implicit IMEX step.
        dt_implicit : float
            Implicit time step [s] (typically ``2 * dt`` for leapfrog).

        Returns
        -------
        tuple[PrimitiveEquationState, PhysicsDiagnostics]
            Corrected state with implicit boundary-layer treatment, and
            surface flux diagnostics for the fluxes applied implicitly.
            Diagnostic fields other than surface fluxes are ``None``.
        """
        ...


@runtime_checkable
class MoistForcing(Forcing, Protocol):
    """Protocol for forcings that also provide a reference humidity profile.

    Extends :class:`Forcing`.  Moist physics schemes should implement this
    so that the semi-implicit solver can compute the virtual reference
    temperature ``T_v_ref`` needed for the implicit moisture coupling.
    """

    def compute_reference_humidity(
        self,
        reference_temperature: np.ndarray,
    ) -> np.ndarray:
        """Compute a reference humidity profile for the semi-implicit solver.

        Parameters
        ----------
        reference_temperature : np.ndarray
            Dry reference temperature profile, shape ``(n_levels,)``.

        Returns
        -------
        np.ndarray
            Reference specific humidity, shape ``(n_levels,)``.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class HeldSuarezConfig:
    """Configuration for Held-Suarez (1994) forcing.

    All rate parameters default to the values from the original paper.

    Parameters
    ----------
    k_a : float
        Free-atmosphere relaxation rate [s⁻¹].
    k_s : float
        Surface relaxation rate [s⁻¹].
    k_f : float
        Boundary-layer Rayleigh friction rate [s⁻¹].
    delta_t_y : float
        Equator-to-pole temperature difference [K].
    delta_theta_z : float
        Vertical temperature lapse-rate parameter [K].
    sigma_b : float
        Boundary-layer top (σ coordinate).
    t_min : float
        Minimum equilibrium temperature [K].
    p0 : float
        Reference pressure [Pa].
    """

    k_a: float = 1.0 / (40.0 * 86400.0)
    k_s: float = 1.0 / (4.0 * 86400.0)
    k_f: float = 1.0 / (1.0 * 86400.0)
    delta_t_y: float = 60.0
    delta_theta_z: float = 10.0
    sigma_b: float = 0.7
    t_min: float = 200.0
    p0: float = 1.0e5


class HeldSuarez:
    """Held-Suarez (1994) forcing for the standard GCM benchmark.

    Implements Newtonian relaxation toward a prescribed equilibrium
    temperature profile and Rayleigh drag in the boundary layer.

    The equilibrium temperature is::

        T_eq = max(T_min, [315 - ΔT_y sin²φ - Δθ_z log(p/p₀) cos²φ] (p/p₀)^κ)

    The Newtonian relaxation rate varies from k_a in the free atmosphere
    to k_s at the surface within the boundary layer (σ > σ_b)::

        k_T = k_a + (k_s - k_a) max(0, (σ - σ_b)/(1 - σ_b)) cos⁴φ

    Rayleigh friction damps vorticity and divergence in the boundary layer::

        k_v = k_f max(0, (σ - σ_b)/(1 - σ_b))

    Parameters
    ----------
    transform : SpectralTransform
        Pre-computed spectral transform (for grid ↔ spectral conversion).
    planet : PlanetaryConstants
        Planetary constants (provides κ = R/cₚ).
    levels : SigmaLevels
        Sigma vertical coordinate.
    config : HeldSuarezConfig or None
        Forcing parameters.  ``None`` uses the standard paper values.

    References
    ----------
    Held, I. M. & Suarez, M. J. (1994). A proposal for the intercomparison
    of the dynamical cores of atmospheric general circulation models.
    BAMS 75(10), 1825-1830.
    """

    def __init__(
        self,
        transform: SpectralTransform,
        planet: PlanetaryConstants,
        levels: SigmaLevels,
        config: HeldSuarezConfig | None = None,
    ) -> None:
        if config is None:
            config = HeldSuarezConfig()
        self.config = config

        self.transform = transform
        self.planet = planet
        self.levels = levels
        self.kappa = planet.kappa

        # Pre-compute boundary-layer ramp: max(0, (σ - σ_b)/(1 - σ_b))
        sigma_full = np.asarray(levels.sigma_full)
        self.sigma_frac = jnp.array(
            np.maximum(0.0, (sigma_full - config.sigma_b) / (1.0 - config.sigma_b))
        )

        # Rayleigh friction coefficient per level (level-only, no lat dependence)
        self.k_v = config.k_f * self.sigma_frac

        # Grid geometry (captured for use in __call__)
        self.sin_lat = transform.grid.sin_lat
        self.cos_lat = transform.grid.cos_lat

    def __call__(
        self,
        state: PrimitiveEquationState,
        surface_pressure: jnp.ndarray,
    ) -> tuple[PrimitiveEquationState, PhysicsDiagnostics]:
        """Compute Held-Suarez tendencies.

        Parameters
        ----------
        state : PrimitiveEquationState
            Current model state (spectral coefficients).
        surface_pressure : jnp.ndarray
            Surface pressure field pₛ (grid space), shape ``(n_lat, n_lon)``.

        Returns
        -------
        tuple[PrimitiveEquationState, PhysicsDiagnostics]
            Tendencies and empty diagnostics (Held-Suarez has no radiation
            or moisture).
        """
        cfg = self.config

        # --- Rayleigh friction (spectral space) ---
        # k_v depends only on sigma level, so multiplication is diagonal
        k_v = self.k_v[:, None]  # (n_levels, 1)
        dvort_spec = -k_v * state.vorticity
        ddiv_spec = -k_v * state.divergence

        # --- Newtonian temperature relaxation (grid space) ---
        # Transform temperature to grid
        t_grid = jax.vmap(self.transform.spectral_to_grid)(state.temperature)

        # Pressure at each level: p = σ · pₛ
        sigma = self.levels.sigma_full[:, None, None]  # (n_levels, 1, 1)
        ps = surface_pressure[None, :, :]  # (1, n_lat, n_lon)
        p = sigma * ps

        # Equilibrium temperature (H&S 1994, Eq. 1)
        sin2 = self.sin_lat[None, :, None] ** 2  # (1, n_lat, 1)
        cos2 = self.cos_lat[None, :, None] ** 2  # (1, n_lat, 1)
        p_ratio = p / cfg.p0
        t_eq = jnp.maximum(
            cfg.t_min,
            (315.0 - cfg.delta_t_y * sin2 - cfg.delta_theta_z * jnp.log(p_ratio) * cos2)
            * p_ratio**self.kappa,
        )

        # Newtonian relaxation rate (H&S 1994, Eq. 2)
        cos4 = self.cos_lat[None, :, None] ** 4  # (1, n_lat, 1)
        sigma_frac = self.sigma_frac[:, None, None]  # (n_levels, 1, 1)
        k_t = cfg.k_a + (cfg.k_s - cfg.k_a) * sigma_frac * cos4

        # Temperature tendency in grid space
        dt_grid = -k_t * (t_grid - t_eq)

        # Transform back to spectral
        dt_spec = jax.vmap(self.transform.grid_to_spectral)(dt_grid)

        # No surface pressure tendency from Held-Suarez forcing
        zero_lnps = jnp.zeros_like(state.log_surface_pressure)

        # No moisture tendency (dry benchmark)
        humidity_tend: jnp.ndarray | None = None
        if state.humidity is not None:
            humidity_tend = jnp.zeros_like(state.humidity)

        tendencies = PrimitiveEquationState(
            vorticity=dvort_spec,
            divergence=ddiv_spec,
            temperature=dt_spec,
            log_surface_pressure=zero_lnps,
            humidity=humidity_tend,
        )
        return tendencies, PhysicsDiagnostics()
