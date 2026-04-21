"""Tests for sea ice thermodynamics (Phase 11A).

Zero-layer Semtner model: ice thickness, ice fraction, albedo feedback,
freezing/melting coupled to the slab ocean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from notus.physics.surface import (
    OceanState,
    PrescribedSST,
    SeaIceConfig,
    SeaIceState,
    SlabOceanConfig,
    SurfaceState,
    compute_ice_weighted_flux,
    compute_sst,
    conductive_flux,
    diagnose_ice_surface_temperature,
    ice_fraction_from_thickness,
    ice_growth_rate,
    ice_modified_albedo,
    ice_weighted_surface_temperature,
    init_sea_ice_state,
    step_sea_ice,
)


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Category 1 — Config & State Infrastructure
# ---------------------------------------------------------------------------


class TestSeaIceConfig:
    """Verify sea ice configuration and derived properties."""

    def test_default_energy_per_meter(self) -> None:
        """Energy per meter should equal rho_ice * L_fusion."""
        cfg = SeaIceConfig()
        expected = 917.0 * 3.34e5  # ~3.063e8 J/m^3
        np.testing.assert_allclose(cfg.energy_per_meter, expected)

    def test_custom_rho_scales_energy(self) -> None:
        """Energy per meter should scale linearly with ice density."""
        cfg1 = SeaIceConfig(rho_ice=917.0)
        cfg2 = SeaIceConfig(rho_ice=1834.0)
        np.testing.assert_allclose(cfg2.energy_per_meter, 2.0 * cfg1.energy_per_meter)


class TestSeaIceState:
    """Verify SeaIceState pytree registration and operations."""

    def test_pytree_roundtrip(self) -> None:
        """SeaIceState should survive JAX pytree flatten/unflatten."""
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0, 2.0]),
            ice_fraction=jnp.array([0.5, 1.0]),
        )
        leaves, treedef = jax.tree_util.tree_flatten(ice)
        ice2 = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_allclose(ice2.ice_thickness, ice.ice_thickness)
        np.testing.assert_allclose(ice2.ice_fraction, ice.ice_fraction)

    def test_tree_map(self) -> None:
        """JAX tree_map should work on SeaIceState."""
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([0.5]),
        )
        scaled = jax.tree.map(lambda x: x * 2.0, ice)
        np.testing.assert_allclose(scaled.ice_thickness, 2.0)
        np.testing.assert_allclose(scaled.ice_fraction, 1.0)

    def test_jit_compatible(self) -> None:
        """SeaIceState should pass through JIT."""

        @jax.jit
        def identity(s: SeaIceState) -> SeaIceState:
            return s

        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([0.5]),
        )
        result = identity(ice)
        np.testing.assert_allclose(result.ice_thickness, 1.0)
        np.testing.assert_allclose(result.ice_fraction, 0.5)

    def test_replace(self) -> None:
        """replace() should update specified fields, leave others unchanged."""
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([0.5]),
        )
        new = ice.replace(ice_thickness=jnp.array([2.0]))
        np.testing.assert_allclose(new.ice_thickness, 2.0)
        np.testing.assert_allclose(new.ice_fraction, 0.5)


class TestSurfaceStateWithIce:
    """Verify SurfaceState works with the ice field."""

    def test_ocean_ice_no_land(self) -> None:
        """SurfaceState with ocean and ice but no land."""
        ocean = OceanState(surface_temperature=jnp.array([280.0]))
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([0.8]),
        )
        sfc = SurfaceState(ocean=ocean, ice=ice)
        assert sfc.land is None
        assert sfc.ice is not None
        np.testing.assert_allclose(sfc.ice.ice_thickness, 1.0)

    def test_pytree_roundtrip_with_ice(self) -> None:
        """SurfaceState with all three components round-trips through pytree."""
        from notus.physics.surface import LandState

        ocean = OceanState(surface_temperature=jnp.array([280.0]))
        land = LandState(
            soil_temperature=jnp.array([[260.0]]),
            bucket_depth=jnp.array([[0.1]]),
        )
        ice = SeaIceState(
            ice_thickness=jnp.array([1.5]),
            ice_fraction=jnp.array([1.0]),
        )
        sfc = SurfaceState(ocean=ocean, land=land, ice=ice)
        leaves, treedef = jax.tree_util.tree_flatten(sfc)
        sfc2 = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_allclose(sfc2.ocean.surface_temperature, 280.0)
        np.testing.assert_allclose(sfc2.land.soil_temperature, 260.0)
        np.testing.assert_allclose(sfc2.ice.ice_thickness, 1.5)

    def test_backward_compatible_no_ice(self) -> None:
        """SurfaceState without ice should still work (ice=None)."""
        ocean = OceanState(surface_temperature=jnp.array([290.0]))
        sfc = SurfaceState(ocean=ocean)
        assert sfc.ice is None
        assert sfc.land is None
        leaves, treedef = jax.tree_util.tree_flatten(sfc)
        sfc2 = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_allclose(sfc2.ocean.surface_temperature, 290.0)
        assert sfc2.ice is None


# ---------------------------------------------------------------------------
# Category 2 — Analytical Thermodynamic Tests
# ---------------------------------------------------------------------------


class TestConductiveFlux:
    """Verify conductive heat flux through ice: F = k * (T_freeze - T_s) / H."""

    def test_known_value(self) -> None:
        """H=1m, dT=10K, k=2.0 -> F=20 W/m^2."""
        f = conductive_flux(
            ice_thickness=jnp.array([1.0]),
            t_surface_ice=jnp.array([261.35]),
            t_freeze=271.35,
            k_ice=2.0,
        )
        np.testing.assert_allclose(f, 20.0, rtol=1e-10)

    def test_thin_ice_large_flux(self) -> None:
        """Thin ice (0.1m) conducts 10x more than 1m ice."""
        f = conductive_flux(
            ice_thickness=jnp.array([0.1]),
            t_surface_ice=jnp.array([261.35]),
            t_freeze=271.35,
            k_ice=2.0,
        )
        np.testing.assert_allclose(f, 200.0, rtol=1e-10)

    def test_thick_ice_small_flux(self) -> None:
        """Thick ice (5m) conducts much less."""
        f = conductive_flux(
            ice_thickness=jnp.array([5.0]),
            t_surface_ice=jnp.array([261.35]),
            t_freeze=271.35,
            k_ice=2.0,
        )
        np.testing.assert_allclose(f, 4.0, rtol=1e-10)

    def test_surface_at_freezing_zero_flux(self) -> None:
        """No temperature gradient means zero conductive flux."""
        f = conductive_flux(
            ice_thickness=jnp.array([1.0]),
            t_surface_ice=jnp.array([271.35]),
            t_freeze=271.35,
            k_ice=2.0,
        )
        np.testing.assert_allclose(f, 0.0, atol=1e-12)


class TestDiagnoseIceSurfaceTemperature:
    """Verify diagnostic ice surface temperature from linearized flux balance.

    The zero-layer model solves: F_atm(T_s) = F_cond(T_s) where
    F_atm = F0 + dF/dT * (T_s - T_freeze) and F_cond = k*(T_s - T_freeze)/H.
    Solution: T_s = T_freeze + F0 / (k/H - dF/dT), capped at T_freeze.
    """

    def test_cold_atmosphere(self) -> None:
        """Negative atmospheric flux -> T_s below freezing.

        F0=-20 W/m^2, H=1m, k=2.0, dF/dT=-4.0 W/(m^2*K).
        T_s = 271.35 + (-20)/(2.0/1.0 - (-4.0)) = 271.35 - 20/6 = 268.017 K.
        """
        t_s, melt_flux = diagnose_ice_surface_temperature(
            ice_thickness=jnp.array([1.0]),
            atm_flux_at_freeze=jnp.array([-20.0]),
            dflux_dt=jnp.array([-4.0]),
            k_ice=2.0,
            t_freeze=271.35,
        )
        expected = 271.35 + (-20.0) / (2.0 / 1.0 - (-4.0))
        np.testing.assert_allclose(t_s, expected, rtol=1e-10)
        np.testing.assert_allclose(melt_flux, 0.0, atol=1e-12)

    def test_warm_atmosphere_capped(self) -> None:
        """Positive atmospheric flux -> T_s capped at T_freeze.

        Without cap, T_s > T_freeze. The excess goes to surface melt.
        """
        t_s, melt_flux = diagnose_ice_surface_temperature(
            ice_thickness=jnp.array([1.0]),
            atm_flux_at_freeze=jnp.array([50.0]),
            dflux_dt=jnp.array([-4.0]),
            k_ice=2.0,
            t_freeze=271.35,
        )
        np.testing.assert_allclose(t_s, 271.35, rtol=1e-10)
        # When T_s = T_freeze, conductive flux is zero, so all atmospheric
        # flux goes to melting. F_melt = F0 + dF/dT*(T_freeze - T_freeze) = F0.
        np.testing.assert_allclose(melt_flux, 50.0, rtol=1e-10)

    def test_returns_melt_flux(self) -> None:
        """Melt flux should be zero when surface is below freezing."""
        _, melt_flux = diagnose_ice_surface_temperature(
            ice_thickness=jnp.array([2.0]),
            atm_flux_at_freeze=jnp.array([-30.0]),
            dflux_dt=jnp.array([-5.0]),
            k_ice=2.0,
            t_freeze=271.35,
        )
        np.testing.assert_allclose(melt_flux, 0.0, atol=1e-12)


class TestIceGrowthRate:
    """Verify ice growth rate: dH/dt = (F_cond - F_ocean) / (rho * L)."""

    def test_steady_state_zero_growth(self) -> None:
        """F_cond = F_ocean -> no growth."""
        rate = ice_growth_rate(
            conductive_flux=jnp.array([20.0]),
            oceanic_heat_flux=jnp.array([20.0]),
            rho_ice=917.0,
            l_fusion=3.34e5,
        )
        np.testing.assert_allclose(rate, 0.0, atol=1e-15)

    def test_growth_when_cond_exceeds_ocean(self) -> None:
        """F_cond > F_ocean -> positive growth (ice thickens)."""
        rate = ice_growth_rate(
            conductive_flux=jnp.array([30.0]),
            oceanic_heat_flux=jnp.array([10.0]),
            rho_ice=917.0,
            l_fusion=3.34e5,
        )
        assert float(rate[0]) > 0.0

    def test_melt_when_ocean_exceeds_cond(self) -> None:
        """F_ocean > F_cond -> negative rate (bottom melt)."""
        rate = ice_growth_rate(
            conductive_flux=jnp.array([10.0]),
            oceanic_heat_flux=jnp.array([30.0]),
            rho_ice=917.0,
            l_fusion=3.34e5,
        )
        assert float(rate[0]) < 0.0

    def test_exact_value(self) -> None:
        """20 W/m^2 net freezing over 1 day -> dH = 0.00564 m."""
        rate = ice_growth_rate(
            conductive_flux=jnp.array([20.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            rho_ice=917.0,
            l_fusion=3.34e5,
        )
        dh_per_day = float(rate[0]) * 86400.0
        expected = 20.0 * 86400.0 / (917.0 * 3.34e5)
        np.testing.assert_allclose(dh_per_day, expected, rtol=1e-10)


class TestSteadyStateIceThickness:
    """Verify convergence to analytical equilibrium thickness.

    At steady state: H_eq = k * (T_freeze - T_s) / F_ocean.
    """

    def test_equilibrium_convergence(self) -> None:
        """From H=0.1m, iterate toward equilibrium.

        With F0=-20, dF/dT=-4, k=2, F_ocean=10:
        H_eq satisfies F_cond(H) = F_ocean.  F_cond = k·ΔT/H where
        ΔT = -F0/(k/H + |dF/dT|).  Solving gives H_eq ≈ 0.5 m.

        Use jax.lax.fori_loop with daily timesteps for fast convergence.
        """
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([0.1]),
            ice_fraction=jnp.array([0.1]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))
        dt = 86400.0  # 1 day
        atm = jnp.array([-20.0])
        dfl = jnp.array([-4.0])
        ohf = jnp.array([10.0])

        def body(_, carry):  # type: ignore[no-untyped-def]
            i, o = carry
            return step_sea_ice(i, o, atm, dfl, ohf, cfg, dt)

        ice, ocean = jax.lax.fori_loop(0, 2000, body, (ice, ocean))

        h_final = float(ice.ice_thickness[0])
        # Should converge near 0.5 m (analytical equilibrium)
        np.testing.assert_allclose(h_final, 0.5, atol=0.1)

    def test_equilibrium_thin(self) -> None:
        """Starting thick (2m), ice should thin toward equilibrium.

        With F0=-2, dF/dT=-4, k=2, F_ocean=10: the weak atmospheric
        cooling cannot sustain thick ice against the ocean heat flux.
        """
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([2.0]),
            ice_fraction=jnp.array([1.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))
        dt = 86400.0
        atm = jnp.array([-2.0])
        dfl = jnp.array([-4.0])
        ohf = jnp.array([10.0])

        def body(_, carry):  # type: ignore[no-untyped-def]
            i, o = carry
            return step_sea_ice(i, o, atm, dfl, ohf, cfg, dt)

        ice, ocean = jax.lax.fori_loop(0, 2000, body, (ice, ocean))

        h_final = float(ice.ice_thickness[0])
        assert h_final < 1.0, f"Expected thinning from 2.0m, got H={h_final}"


class TestEnergyConservation:
    """Verify energy conservation during ice growth and melt."""

    def test_growth_energy_balance(self) -> None:
        """Energy stored in new ice should match integrated conductive flux.

        With no ocean heat flux and no surface melt (cold atmosphere),
        all conductive flux goes to ice growth:
        rho * L * dH = F_cond * dt
        """
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([1.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))
        dt = 3600.0
        h_before = float(ice.ice_thickness[0])

        ice_new, _ = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-20.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            config=cfg,
            dt=dt,
        )

        h_after = float(ice_new.ice_thickness[0])
        dh = h_after - h_before
        energy_in_ice = cfg.rho_ice * cfg.l_fusion * dh

        # Compute the conductive flux for this step. The surface temperature
        # is diagnosed, so we reconstruct it.
        k_over_h = cfg.k_ice / max(h_before, cfg.h_min)
        f0 = -20.0
        df_dt = -4.0
        t_s = cfg.t_freeze + f0 / (k_over_h - df_dt)
        f_cond = cfg.k_ice * (cfg.t_freeze - t_s) / h_before
        energy_from_flux = f_cond * dt

        np.testing.assert_allclose(energy_in_ice, energy_from_flux, rtol=1e-10)

    def test_melt_energy_balance(self) -> None:
        """Energy released by melting should match integrated ocean heat flux excess."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([2.0]),
            ice_fraction=jnp.array([1.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))
        dt = 3600.0
        h_before = float(ice.ice_thickness[0])

        # Large ocean heat flux, cold atmosphere -> bottom melt dominates
        ice_new, _ = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-10.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([50.0]),
            config=cfg,
            dt=dt,
        )

        h_after = float(ice_new.ice_thickness[0])
        dh = h_before - h_after  # positive = ice lost
        energy_released = cfg.rho_ice * cfg.l_fusion * dh

        # Reconstruct conductive flux
        k_over_h = cfg.k_ice / h_before
        f0 = -10.0
        df_dt = -4.0
        t_s = cfg.t_freeze + f0 / (k_over_h - df_dt)
        f_cond = cfg.k_ice * (cfg.t_freeze - t_s) / h_before
        # Net energy into melting = F_ocean - F_cond (bottom) + melt_flux (top)
        # Top melt is zero here because T_s < T_freeze
        net_melt_energy = (50.0 - f_cond) * dt

        np.testing.assert_allclose(energy_released, net_melt_energy, rtol=1e-10)


# ---------------------------------------------------------------------------
# Category 3 — Freezing/Melting Regime Tests
# ---------------------------------------------------------------------------


class TestFreezingOnset:
    """Verify ice formation criteria."""

    def test_no_ice_warm_ocean(self) -> None:
        """Warm ocean (280 K) should not form ice."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([0.0]),
            ice_fraction=jnp.array([0.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([280.0]))

        ice_new, _ocean_new = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-10.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            config=cfg,
            dt=3600.0,
        )
        np.testing.assert_allclose(ice_new.ice_thickness, 0.0, atol=1e-15)
        np.testing.assert_allclose(ice_new.ice_fraction, 0.0, atol=1e-15)

    def test_ice_forms_at_freezing(self) -> None:
        """SST at freezing with negative flux should create ice."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([0.0]),
            ice_fraction=jnp.array([0.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))

        ice_new, _ = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-20.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            config=cfg,
            dt=3600.0,
        )
        assert float(ice_new.ice_thickness[0]) > 0.0
        assert float(ice_new.ice_fraction[0]) > 0.0

    def test_ice_forms_below_freezing(self) -> None:
        """SST below freezing should form ice and clamp SST to T_freeze."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([0.0]),
            ice_fraction=jnp.array([0.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([270.0]))

        ice_new, ocean_new = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-10.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            config=cfg,
            dt=3600.0,
        )
        assert float(ice_new.ice_thickness[0]) > 0.0
        np.testing.assert_allclose(ocean_new.surface_temperature, cfg.t_freeze, rtol=1e-10)

    def test_sst_clamped_at_freezing(self) -> None:
        """With existing ice, SST should remain at T_freeze after cooling."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([1.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))

        _, ocean_new = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-30.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            config=cfg,
            dt=3600.0,
        )
        np.testing.assert_allclose(ocean_new.surface_temperature, cfg.t_freeze, rtol=1e-10)


class TestMeltingRegimes:
    """Verify ice melting from top and bottom."""

    def test_top_melt(self) -> None:
        """Warm atmosphere melts ice from the top (surface melt)."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([1.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))

        # Strong positive atmospheric flux at T_freeze -> surface melt
        ice_new, _ = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([100.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([0.0]),
            config=cfg,
            dt=3600.0,
        )
        assert float(ice_new.ice_thickness[0]) < 1.0

    def test_bottom_melt(self) -> None:
        """Large ocean heat flux melts ice from below."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([1.0]),
            ice_fraction=jnp.array([1.0]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))

        ice_new, _ = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([-20.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([100.0]),
            config=cfg,
            dt=3600.0,
        )
        assert float(ice_new.ice_thickness[0]) < 1.0

    def test_ice_disappears_completely(self) -> None:
        """Very thin ice under strong melt should vanish entirely."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([0.02]),
            ice_fraction=jnp.array([0.02]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))
        atm = jnp.array([200.0])
        dfl = jnp.array([-4.0])
        ohf = jnp.array([200.0])

        # Several steps with massive melt flux
        def body(_, carry):  # type: ignore[no-untyped-def]
            i, o = carry
            return step_sea_ice(i, o, atm, dfl, ohf, cfg, 3600.0)

        ice_new, _ = jax.lax.fori_loop(0, 10, body, (ice, ocean))
        np.testing.assert_allclose(ice_new.ice_thickness, 0.0, atol=1e-15)
        np.testing.assert_allclose(ice_new.ice_fraction, 0.0, atol=1e-15)

    def test_sst_resumes_after_melt(self) -> None:
        """When all ice melts, excess heat warms ocean above T_freeze."""
        cfg = SeaIceConfig()
        ice = SeaIceState(
            ice_thickness=jnp.array([0.005]),
            ice_fraction=jnp.array([0.005]),
        )
        ocean = OceanState(surface_temperature=jnp.array([cfg.t_freeze]))

        # Huge ocean heat flux on very thin ice -> ice melts, ocean warms
        ice_new, ocean_new = step_sea_ice(
            ice=ice,
            ocean=ocean,
            atm_flux_at_freeze=jnp.array([0.0]),
            dflux_dt=jnp.array([-4.0]),
            oceanic_heat_flux=jnp.array([500.0]),
            config=cfg,
            dt=3600.0,
        )
        np.testing.assert_allclose(ice_new.ice_thickness, 0.0, atol=1e-15)
        assert float(ocean_new.surface_temperature[0]) > cfg.t_freeze


class TestIceFractionLogic:
    """Verify ice fraction parameterization: f = min(1, H / H_crit)."""

    def test_fraction_one_when_thick(self) -> None:
        """H >= H_crit -> f = 1."""
        f = ice_fraction_from_thickness(jnp.array([1.0, 2.0, 5.0]), h_crit=1.0)
        np.testing.assert_allclose(f, jnp.array([1.0, 1.0, 1.0]))

    def test_fraction_scales_linearly(self) -> None:
        """H = 0.5 * H_crit -> f = 0.5."""
        f = ice_fraction_from_thickness(jnp.array([0.5]), h_crit=1.0)
        np.testing.assert_allclose(f, 0.5, rtol=1e-10)

    def test_fraction_zero_when_no_ice(self) -> None:
        """H = 0 -> f = 0."""
        f = ice_fraction_from_thickness(jnp.array([0.0]), h_crit=1.0)
        np.testing.assert_allclose(f, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Category 4 — Albedo and Insulation Tests
# ---------------------------------------------------------------------------


class TestIceAlbedo:
    """Verify albedo blending: alpha = f * alpha_ice + (1-f) * alpha_ocean."""

    def test_full_ice(self) -> None:
        """Full ice coverage -> ice albedo."""
        a = ice_modified_albedo(jnp.array([1.0]), albedo_ice=0.65, albedo_ocean=0.06)
        np.testing.assert_allclose(a, 0.65, rtol=1e-10)

    def test_no_ice(self) -> None:
        """No ice -> ocean albedo."""
        a = ice_modified_albedo(jnp.array([0.0]), albedo_ice=0.65, albedo_ocean=0.06)
        np.testing.assert_allclose(a, 0.06, rtol=1e-10)

    def test_partial(self) -> None:
        """50% ice coverage -> average of ice and ocean albedo."""
        a = ice_modified_albedo(jnp.array([0.5]), albedo_ice=0.65, albedo_ocean=0.06)
        expected = 0.5 * 0.65 + 0.5 * 0.06
        np.testing.assert_allclose(a, expected, rtol=1e-10)

    def test_array_blending(self) -> None:
        """Vectorized blending across multiple ice fractions."""
        fracs = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        a = ice_modified_albedo(fracs, albedo_ice=0.65, albedo_ocean=0.06)
        expected = fracs * 0.65 + (1.0 - fracs) * 0.06
        np.testing.assert_allclose(a, expected, rtol=1e-10)


class TestIceInsulation:
    """Verify effective surface temperature: T_eff = f*T_ice + (1-f)*SST."""

    def test_full_ice_sees_ice_temp(self) -> None:
        """Full ice -> atmosphere sees ice surface temperature."""
        t = ice_weighted_surface_temperature(
            jnp.array([1.0]), t_ice_surface=jnp.array([250.0]), sst=jnp.array([271.35])
        )
        np.testing.assert_allclose(t, 250.0, rtol=1e-10)

    def test_no_ice_sees_sst(self) -> None:
        """No ice -> atmosphere sees SST."""
        t = ice_weighted_surface_temperature(
            jnp.array([0.0]), t_ice_surface=jnp.array([250.0]), sst=jnp.array([271.35])
        )
        np.testing.assert_allclose(t, 271.35, rtol=1e-10)

    def test_partial_blending(self) -> None:
        """50% ice -> weighted average."""
        t = ice_weighted_surface_temperature(
            jnp.array([0.5]), t_ice_surface=jnp.array([260.0]), sst=jnp.array([271.35])
        )
        expected = 0.5 * 260.0 + 0.5 * 271.35
        np.testing.assert_allclose(t, expected, rtol=1e-10)

    def test_insulation_reduces_heat_loss(self) -> None:
        """Ice-covered surface should have lower effective temperature than open ocean.

        Lower effective T -> less LW emission -> less heat loss to space.
        """
        sst = jnp.array([271.35])
        t_ice = jnp.array([255.0])

        t_with_ice = ice_weighted_surface_temperature(
            jnp.array([0.8]), t_ice_surface=t_ice, sst=sst
        )
        t_no_ice = ice_weighted_surface_temperature(jnp.array([0.0]), t_ice_surface=t_ice, sst=sst)
        assert float(t_with_ice[0]) < float(t_no_ice[0])


class TestIceWeightedFlux:
    """Verify area-weighted flux averaging over ice and ocean."""

    def test_area_weighted_average(self) -> None:
        """f=0.3, F_ice=-10, F_ocean=50 -> 0.3*(-10) + 0.7*50 = 32.0."""
        f = compute_ice_weighted_flux(
            ice_fraction=jnp.array([0.3]),
            flux_over_ice=jnp.array([-10.0]),
            flux_over_ocean=jnp.array([50.0]),
        )
        np.testing.assert_allclose(f, 32.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Category 5 — Integration Tests
# ---------------------------------------------------------------------------


class TestSeaIceIntegration:
    """Integration tests for the coupled sea-ice-ocean stepper."""

    @pytest.fixture(scope="class")
    def setup(self) -> dict:
        """Shared spinup state for integration tests."""
        from notus.constants import EARTH
        from notus.grid import GaussianGrid
        from notus.initial_conditions import moist_aquaplanet_initial_state
        from notus.physics.physics_suite import PhysicsSuite, PhysicsSuiteConfig
        from notus.physics.radiation import ByrneRadiation
        from notus.physics.solar import EARTH_ORBIT
        from notus.timestepping.spinup import spinup_prescribed_sst
        from notus.transforms import SpectralTransform
        from notus.vertical.sigma import standard_sigma_levels

        grid = GaussianGrid(truncation=21)
        transform = SpectralTransform(grid, EARTH.radius)
        levels = standard_sigma_levels(20)
        state, ref, sphi = moist_aquaplanet_initial_state(
            transform,
            EARTH,
            levels,
            initial_rh=0.7,
            seed=42,
        )
        config = PhysicsSuiteConfig(
            radiation=ByrneRadiation(sw_tau_0=0.22),
            orbital=EARTH_ORBIT,
        )
        forcing_spinup = PhysicsSuite(transform, EARTH, levels, config=config)
        result = spinup_prescribed_sst(
            state,
            forcing_spinup,
            transform,
            EARTH,
            levels,
            ref,
            sphi,
            dt=900.0,
            spinup_days=50,
            averaging_days=50,
            verbose=False,
        )
        return {
            "grid": grid,
            "transform": transform,
            "levels": levels,
            "config": config,
            "ref": ref,
            "sphi": sphi,
            "result": result,
        }

    def _run_coupled_ice(
        self,
        setup: dict,
        n_days: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run coupled integration with sea ice, return (SST, H_ice, f_ice)."""
        from notus.constants import EARTH
        from notus.operators import exponential_filter
        from notus.physics.physics_suite import PhysicsSuite
        from notus.timestepping.coupled import build_coupled_pe_stepper

        s = setup
        forcing = PhysicsSuite(s["transform"], EARTH, s["levels"], config=s["config"])
        sst_init = compute_sst(PrescribedSST(), s["grid"].latitudes)
        ocean = OceanState(surface_temperature=sst_init)
        ice_cfg = SeaIceConfig()
        ice = init_sea_ice_state(sst_init, ice_cfg)
        surface = SurfaceState(ocean=ocean, ice=ice)

        init_fn, step_fn = build_coupled_pe_stepper(
            transform=s["transform"],
            planet=EARTH,
            levels=s["levels"],
            reference_temperature=s["ref"],
            surface_geopotential=s["sphi"],
            dt=900.0,
            forcing=forcing,
            ocean_config=SlabOceanConfig(mixed_layer_depth=50.0),
            q_flux=s["result"].q_flux,
            ice_config=ice_cfg,
            spectral_filter=exponential_filter(s["transform"].arrays, 900.0),
        )
        steps_per_day = 96
        day_of_year = jnp.float64(0.0)
        forcing.prescribed_sst = surface.ocean.surface_temperature
        prev, curr, surface, _diags = init_fn(s["result"].state, surface, day_of_year)

        for day in range(1, n_days + 1):
            day_of_year = jnp.float64(day)
            forcing.prescribed_sst = surface.ocean.surface_temperature

            def scan_body(carry, _, day_of_year=day_of_year):  # type: ignore[no-untyped-def]
                p, c, sfc = carry
                p, c, sfc, _d = step_fn(p, c, sfc, day_of_year)
                return (p, c, sfc), None

            (prev, curr, surface), _ = jax.lax.scan(
                scan_body,
                (prev, curr, surface),
                None,
                length=steps_per_day,
            )

        sst = np.asarray(surface.ocean.surface_temperature)
        assert surface.ice is not None
        h_ice = np.asarray(surface.ice.ice_thickness)
        f_ice = np.asarray(surface.ice.ice_fraction)
        return sst, h_ice, f_ice

    @pytest.mark.slow
    def test_ice_ocean_stable_10_days(self, setup: dict) -> None:
        """Coupled ice-ocean run stays finite for 10 days."""
        sst, h_ice, f_ice = self._run_coupled_ice(setup)
        assert np.all(np.isfinite(sst))
        assert np.all(np.isfinite(h_ice))
        assert np.all(np.isfinite(f_ice))

    @pytest.mark.slow
    def test_ice_thickness_physical_range(self, setup: dict) -> None:
        """Ice thickness stays in a physically plausible range."""
        _, h_ice, _ = self._run_coupled_ice(setup)
        assert np.min(h_ice) >= 0.0
        assert np.max(h_ice) < 10.0

    @pytest.mark.slow
    def test_ice_fraction_in_range(self, setup: dict) -> None:
        """Ice fraction stays in [0, 1]."""
        _, _, f_ice = self._run_coupled_ice(setup)
        assert np.min(f_ice) >= 0.0
        assert np.max(f_ice) <= 1.0

    @pytest.mark.slow
    def test_sst_not_below_freezing_when_ice(self, setup: dict) -> None:
        """Where ice is present, SST should be at or above T_freeze."""
        sst, _, f_ice = self._run_coupled_ice(setup)
        ice_mask = f_ice > 0.0
        if np.any(ice_mask):
            np.testing.assert_array_less(271.34, sst[ice_mask])

    @pytest.mark.slow
    def test_ice_at_high_latitudes_only(self, setup: dict) -> None:
        """Ice should only exist at high latitudes, not near the equator."""
        _sst, h_ice, _ = self._run_coupled_ice(setup)
        lats = np.asarray(setup["grid"].latitudes)
        equatorial = np.abs(lats) < np.radians(30.0)
        np.testing.assert_allclose(h_ice[equatorial], 0.0, atol=1e-10)
