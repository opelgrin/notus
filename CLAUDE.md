# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Notus?

Notus is a spectral-transform atmospheric General Circulation Model (GCM) written from scratch in Python/JAX. It solves the primitive equations on a rotating sphere using spherical harmonics, targeting standard benchmarks (Held-Suarez, Jablonowski-Williamson) and aquaplanet configurations with moisture, radiation, slab ocean coupling, bucket land surface, and idealized topography.

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

## Code Conventions

- **Python 3.13+** with strict mypy. JAX arrays are the primary numeric type.
- **Unicode math**: Greek letters (φ, λ, μ, σ, etc.) and math symbols (∂, ∇, ×) are used in variable names and comments to match equations. These are configured as allowed confusables in ruff.
- **Line length**: 100 characters.
- **Docstrings**: NumPy convention.
- **Ruff** handles linting, formatting, and import sorting. Auto-fix is enabled. See `pyproject.toml` for the full rule set and per-file ignores (tests allow assert/magic values/private access; examples allow print/complexity). 
- **JAX idioms**: pure functions, `@jax.jit`, immutable state via frozen dataclasses registered as JAX pytrees. No in-place mutation.
- **Never use `#type: ignore` or `# noqa:` without justification.** If a linter or type checker flags code, fix the underlying issue. Only suppress warnings if the warning is genuinely wrong (e.g., a false positive in mypy or ruff), and in such cases, always include a comment explaining *why* the suppression is needed and *what* the false positive is. The comment should be specific enough that a future reader understands the reason without needing to reproduce the error.
- **Always run ruff format, ruff check, and mypy before committing.** This ensures code is consistently formatted and type-safe.
- **Tests must verify behaviours, not implementations specifics.** Tests should focus on what the code does, not how it does it. This allows for refactoring without breaking tests. Avoid testing implementation details, magic numbers, or internal state unless absolutely necessary.
