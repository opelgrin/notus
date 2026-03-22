"""Shared fixtures for GCM tests."""

from __future__ import annotations

import pytest

from gcm import EARTH, GaussianGrid, SpectralTransform


@pytest.fixture
def t21_grid() -> GaussianGrid:
    """T21 Gaussian grid (standard low-res test configuration)."""
    return GaussianGrid(truncation=21)


@pytest.fixture
def t42_grid() -> GaussianGrid:
    """T42 Gaussian grid (standard Held-Suarez resolution)."""
    return GaussianGrid(truncation=42)


@pytest.fixture
def t21_transform(t21_grid: GaussianGrid) -> SpectralTransform:
    """Spectral transform for T21."""
    return SpectralTransform(t21_grid)


@pytest.fixture
def t42_transform(t42_grid: GaussianGrid) -> SpectralTransform:
    """Spectral transform for T42."""
    return SpectralTransform(t42_grid)


@pytest.fixture
def earth():
    """Earth planetary constants."""
    return EARTH
