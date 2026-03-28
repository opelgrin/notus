# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Notus?

Notus is a spectral-transform atmospheric General Circulation Model (GCM) written from scratch in Python/JAX. It solves the primitive equations on a rotating sphere using spherical harmonics, targeting standard benchmarks (Held-Suarez, Jablonowski-Williamson) and aquaplanet configurations with moisture, radiation, and slab ocean coupling.

## Commands

```bash
# Install dependencies
uv sync --group dev

# Run all tests
uv run pytest tests/ -v

# Skip slow multi-day integration tests
uv run pytest tests/ -v -m 'not slow'

# Run a single test file or function
uv run pytest tests/test_grid.py -v
uv run pytest tests/test_grid.py::TestGaussianGridConstruction::test_t21_default_sizes -v

# Lint and format
uv run ruff check .
uv run ruff format --check .

# Type check
uv run mypy
```

CI runs lint (ruff check, ruff format --check, mypy) and tests (pytest) on push/PR to main.

## Architecture

The model follows a standard spectral-transform GCM pipeline: grid-point physics → spectral-space dynamics → time integration.

**Data flow per timestep:**
1. Transform spectral state → grid point values (`transforms.py`)
2. Compute physics tendencies in grid space (`physics/`)
3. Compute dynamics tendencies in spectral space (`dynamics/`)
4. Solve semi-implicit system for fast waves (`timestepping/semi_implicit_pe.py`)
5. Advance state with IMEX leapfrog + Robert-Asselin filter (`timestepping/imex.py`)

**Key abstractions:**
- `GaussianGrid` — Gaussian latitudes/longitudes/quadrature weights
- `SpectralTransform` — FFT + Legendre transforms with precomputed operator arrays
- `PrimitiveEquationState` / `ShallowWaterState` — immutable JAX pytree dataclasses holding spectral coefficients
- `SigmaLevels` — vertical sigma coordinate definition
- `PlanetaryConstants` — planet parameters (Earth predefined as `EARTH`)
- `Forcing` protocol — physics interface; implementations: `HeldSuarezForcing`, `SimplePhysicsForcing`, etc.
- `build_pe_stepper()` — factory that wires dynamics + physics + semi-implicit solver into a single `step(state, dt)` callable

**Module layout under `src/notus/`:**
- `operators/` — spectral operators (Laplacian, derivatives, filtering, wind reconstruction)
- `dynamics/` — tendency computations for shallow water and primitive equations
- `physics/` — parameterizations (radiation, convection, moisture, boundary layer, surface/slab ocean, solar geometry)
- `timestepping/` — IMEX leapfrog, semi-implicit Helmholtz solvers, spinup utilities
- `vertical/` — sigma coordinate, vertical finite-difference operators

All public API is re-exported from `__init__.py`.

## Code Conventions

- **Python 3.13+** with strict mypy. JAX arrays are the primary numeric type.
- **Unicode math**: Greek letters (φ, λ, μ, σ, etc.) and math symbols (∂, ∇, ×) are used in variable names and comments to match equations. These are configured as allowed confusables in ruff.
- **Line length**: 100 characters.
- **Docstrings**: NumPy convention.
- **Ruff** handles linting, formatting, and import sorting. Auto-fix is enabled. See `pyproject.toml` for the full rule set and per-file ignores (tests allow assert/magic values/private access; examples allow print/complexity). Do not disable rules without good reason explained in a comment.
- **JAX idioms**: pure functions, `@jax.jit`, immutable state via frozen dataclasses registered as JAX pytrees. No in-place mutation.
