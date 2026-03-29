"""Notus — A spectral-transform General Circulation Model in JAX.

This package provides a from-scratch implementation of a spectral-transform
atmospheric model, suitable for idealized simulations (e.g. Held-Suarez)
on any rotating planet with an ideal-gas atmosphere.

Quick start
-----------
Set up a grid and transforms::

    from notus import GaussianGrid, SpectralTransform, EARTH

    grid = GaussianGrid(truncation=21)
    transform = SpectralTransform(grid, EARTH.radius)

Create initial conditions and run::

    from notus import (
        held_suarez_initial_state, standard_sigma_levels,
        HeldSuarez, build_pe_stepper,
    )

    levels = standard_sigma_levels(20)
    state, ref_temps, surface_phi = held_suarez_initial_state(
        transform, EARTH, levels,
    )
    forcing = HeldSuarez(transform, EARTH, levels)
    init_fn, step_fn = build_pe_stepper(
        transform, EARTH, levels, ref_temps, surface_phi,
        dt=1200.0, forcing=forcing,
    )
    prev, curr = init_fn(state)
    prev, curr = step_fn(prev, curr)

Core components
---------------
- Grid: :class:`GaussianGrid`
- Transforms: :class:`SpectralTransform`
- Constants: :class:`PlanetaryConstants`, :data:`EARTH`
- State: :class:`PrimitiveEquationState`, :class:`ShallowWaterState`
- Forcing: :class:`HeldSuarez`, :class:`PhysicsSuite`
- Steppers: :func:`build_pe_stepper`, :func:`build_coupled_pe_stepper`
- Vertical: :class:`SigmaLevels`, :func:`standard_sigma_levels`
- I/O: :func:`save_restart`, :func:`load_restart`
"""

from notus.constants import EARTH, PlanetaryConstants
from notus.diagnostics import (
    ConservationDiagnostics,
    ZonalMeanState,
    compute_conservation_diagnostics,
    compute_zonal_mean_state,
    grid_surface_pressure,
    grid_winds_at_level,
    sigma_integral,
    spherical_integral,
)
from notus.grid import GaussianGrid
from notus.initial_conditions import (
    JWConfig,
    held_suarez_initial_state,
    jablonowski_williamson_perturbation,
    jablonowski_williamson_steady_state,
    moist_aquaplanet_initial_state,
    physics_suite_initial_state,
)
from notus.io import load_restart, save_restart
from notus.operators import (
    OperatorArrays,
    exponential_filter,
    hyperdiffusion,
    hyperdiffusion_scaling,
    inverse_laplacian,
    laplacian,
    meridional_derivative,
    spectral_curl,
    spectral_divergence,
    uv_from_vordiv,
    zonal_derivative,
)
from notus.physics import (
    EARTH_ORBIT,
    BucketLandConfig,
    ByrneRadiation,
    Forcing,
    FriersonRadiation,
    HeldSuarez,
    HeldSuarezConfig,
    ImplicitForcing,
    LandState,
    MoistForcing,
    OceanState,
    OrbitalParameters,
    PhysicsSuite,
    PhysicsSuiteConfig,
    PrescribedSST,
    RadiationConfig,
    SlabOceanConfig,
    SpeedyRadiation,
    SurfaceLayerConfig,
    SurfaceProperties,
    SurfaceState,
    compute_sst,
    flat_continent_surface,
    init_land_state,
)
from notus.runner import SimulationResult, run_simulation
from notus.state import PrimitiveEquationState, ShallowWaterState
from notus.timestepping import (
    SpinupResult,
    build_coupled_pe_stepper,
    build_pe_stepper,
    spinup_prescribed_sst,
)
from notus.topography import (
    gaussian_mountain,
    orographic_log_surface_pressure,
    sinusoidal_mountains,
    smooth_orography,
    zonal_ridge,
)
from notus.transforms import SpectralTransform
from notus.vertical import (
    geopotential,
    omega_over_pressure,
    sigma_dot,
    surface_pressure_tendency,
    vertical_advection,
)
from notus.vertical.sigma import SigmaLevels, standard_sigma_levels, uniform_sigma_levels


__all__ = [
    "EARTH",
    "EARTH_ORBIT",
    "BucketLandConfig",
    "ByrneRadiation",
    "ConservationDiagnostics",
    "Forcing",
    "FriersonRadiation",
    "GaussianGrid",
    "HeldSuarez",
    "HeldSuarezConfig",
    "ImplicitForcing",
    "JWConfig",
    "LandState",
    "MoistForcing",
    "OceanState",
    "OperatorArrays",
    "OrbitalParameters",
    "PhysicsSuite",
    "PhysicsSuiteConfig",
    "PlanetaryConstants",
    "PrescribedSST",
    "PrimitiveEquationState",
    "RadiationConfig",
    "ShallowWaterState",
    "SigmaLevels",
    "SimulationResult",
    "SlabOceanConfig",
    "SpectralTransform",
    "SpeedyRadiation",
    "SpinupResult",
    "SurfaceLayerConfig",
    "SurfaceProperties",
    "SurfaceState",
    "ZonalMeanState",
    "build_coupled_pe_stepper",
    "build_pe_stepper",
    "compute_conservation_diagnostics",
    "compute_sst",
    "compute_zonal_mean_state",
    "exponential_filter",
    "flat_continent_surface",
    "gaussian_mountain",
    "geopotential",
    "grid_surface_pressure",
    "grid_winds_at_level",
    "held_suarez_initial_state",
    "hyperdiffusion",
    "hyperdiffusion_scaling",
    "init_land_state",
    "inverse_laplacian",
    "jablonowski_williamson_perturbation",
    "jablonowski_williamson_steady_state",
    "laplacian",
    "load_restart",
    "meridional_derivative",
    "moist_aquaplanet_initial_state",
    "omega_over_pressure",
    "orographic_log_surface_pressure",
    "physics_suite_initial_state",
    "run_simulation",
    "save_restart",
    "sigma_dot",
    "sigma_integral",
    "sinusoidal_mountains",
    "smooth_orography",
    "spectral_curl",
    "spectral_divergence",
    "spherical_integral",
    "spinup_prescribed_sst",
    "standard_sigma_levels",
    "surface_pressure_tendency",
    "uniform_sigma_levels",
    "uv_from_vordiv",
    "vertical_advection",
    "zonal_derivative",
    "zonal_ridge",
]
