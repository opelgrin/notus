"""Tests for slab ocean thermodynamics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.physics.surface import (
    OceanState,
    SlabOceanConfig,
    step_slab_ocean,
)


jax.config.update("jax_enable_x64", True)


class TestSlabOceanConfig:
    """Verify slab ocean configuration."""

    def test_default_heat_capacity(self) -> None:
        """Default 50m mixed layer should have expected heat capacity."""
        cfg = SlabOceanConfig()
        expected = 1025.0 * 3994.0 * 50.0  # ~2.05e8 J/(m²·K)
        np.testing.assert_allclose(cfg.heat_capacity, expected)

    def test_custom_depth(self) -> None:
        """Heat capacity should scale linearly with mixed-layer depth."""
        cfg_50 = SlabOceanConfig(mixed_layer_depth=50.0)
        cfg_100 = SlabOceanConfig(mixed_layer_depth=100.0)
        np.testing.assert_allclose(cfg_100.heat_capacity, 2.0 * cfg_50.heat_capacity)


class TestOceanState:
    """Verify OceanState pytree registration."""

    def test_pytree_roundtrip(self) -> None:
        """OceanState should survive JAX pytree flatten/unflatten."""
        sst = jnp.array([280.0, 290.0, 300.0])
        ocean = OceanState(surface_temperature=sst)
        leaves, treedef = jax.tree_util.tree_flatten(ocean)
        ocean2 = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_allclose(ocean2.surface_temperature, sst)

    def test_tree_map(self) -> None:
        """JAX tree_map should work on OceanState."""
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        scaled = jax.tree.map(lambda x: x * 2.0, ocean)
        np.testing.assert_allclose(scaled.surface_temperature, 600.0)

    def test_jit_compatible(self) -> None:
        """OceanState should pass through JIT."""
        @jax.jit
        def identity(o: OceanState) -> OceanState:
            return o
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        result = identity(ocean)
        np.testing.assert_allclose(result.surface_temperature, 300.0)


class TestStepSlabOcean:
    """Verify slab ocean time stepping."""

    def test_zero_flux_unchanged(self) -> None:
        """Zero net flux should leave SST unchanged."""
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        net_flux = jnp.array([0.0])
        q_flux = jnp.array([0.0])
        ocean_new = step_slab_ocean(ocean, net_flux, q_flux, 2.0e8, 900.0)
        np.testing.assert_allclose(ocean_new.surface_temperature, 300.0)

    def test_positive_flux_warms(self) -> None:
        """Positive net flux should increase SST."""
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        net_flux = jnp.array([100.0])  # 100 W/m²
        q_flux = jnp.array([0.0])
        ocean_new = step_slab_ocean(ocean, net_flux, q_flux, 2.0e8, 900.0)
        assert float(ocean_new.surface_temperature[0]) > 300.0

    def test_negative_flux_cools(self) -> None:
        """Negative net flux should decrease SST."""
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        net_flux = jnp.array([-100.0])
        q_flux = jnp.array([0.0])
        ocean_new = step_slab_ocean(ocean, net_flux, q_flux, 2.0e8, 900.0)
        assert float(ocean_new.surface_temperature[0]) < 300.0

    def test_q_flux_effect(self) -> None:
        """Q-flux should add to the net surface flux."""
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        net_flux = jnp.array([-50.0])
        q_flux = jnp.array([100.0])  # Q-flux exceeds cooling
        ocean_new = step_slab_ocean(ocean, net_flux, q_flux, 2.0e8, 900.0)
        assert float(ocean_new.surface_temperature[0]) > 300.0

    def test_heat_capacity_scaling(self) -> None:
        """Doubling heat capacity should halve the temperature change."""
        ocean = OceanState(surface_temperature=jnp.array([300.0]))
        net_flux = jnp.array([100.0])
        q_flux = jnp.array([0.0])

        o1 = step_slab_ocean(ocean, net_flux, q_flux, 1.0e8, 900.0)
        o2 = step_slab_ocean(ocean, net_flux, q_flux, 2.0e8, 900.0)

        dt1 = float(o1.surface_temperature[0]) - 300.0
        dt2 = float(o2.surface_temperature[0]) - 300.0
        np.testing.assert_allclose(dt1, 2.0 * dt2, rtol=1e-10)

    def test_exact_temperature_change(self) -> None:
        """Verify dT = F * dt / C for a known flux."""
        sst0 = 290.0
        flux = 200.0  # W/m²
        dt = 3600.0   # 1 hour
        c_ocean = 2.0e8  # J/(m²·K)

        ocean = OceanState(surface_temperature=jnp.array([sst0]))
        ocean_new = step_slab_ocean(
            ocean, jnp.array([flux]), jnp.array([0.0]), c_ocean, dt,
        )

        expected = sst0 + flux * dt / c_ocean
        np.testing.assert_allclose(
            float(ocean_new.surface_temperature[0]), expected, rtol=1e-10,
        )
