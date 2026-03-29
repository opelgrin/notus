"""Time integration schemes for the GCM."""

from notus.timestepping.coupled import build_coupled_pe_stepper
from notus.timestepping.imex import build_pe_stepper, euler_init, imex_leapfrog_step
from notus.timestepping.leapfrog import LeapfrogState, euler_step, leapfrog_step
from notus.timestepping.semi_implicit_pe import (
    PESemiImplicitConfig,
    build_pe_semi_implicit_config,
    pe_coupling_matrix,
    pe_implicit_inverse,
    pe_implicit_terms,
    temperature_implicit_weights,
)
from notus.timestepping.semi_implicit_sw import (
    SemiImplicitConfig,
    implicit_inverse,
    implicit_terms,
)
from notus.timestepping.spinup import SpinupResult, spinup_prescribed_sst


__all__ = [
    "LeapfrogState",
    "PESemiImplicitConfig",
    "SemiImplicitConfig",
    "SpinupResult",
    "build_coupled_pe_stepper",
    "build_pe_semi_implicit_config",
    "build_pe_stepper",
    "euler_init",
    "euler_step",
    "imex_leapfrog_step",
    "implicit_inverse",
    "implicit_terms",
    "leapfrog_step",
    "pe_coupling_matrix",
    "pe_implicit_inverse",
    "pe_implicit_terms",
    "spinup_prescribed_sst",
    "temperature_implicit_weights",
]
