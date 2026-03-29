"""Diagnostic cloud scheme for SPEEDY-style radiation.

Computes cloud cover, cloud-top level, and stratiform clouds from
relative humidity, precipitation rate, and boundary-layer stability.
Cloud fields feed into the SPEEDY multi-band radiation to provide
SW reflection (planetary albedo) and LW absorption (cloud greenhouse).

Based on the SPEEDY cloud scheme (Molteni 2003).

All functions are JIT-compatible.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, slots=True)
class CloudConfig:
    """Parameters for the diagnostic cloud scheme.

    Parameters
    ----------
    rhcl1 : float
        Relative humidity threshold for cloud cover = 0.
    rhcl2 : float
        Relative humidity threshold for cloud cover = 1.
    qacl : float
        Minimum specific humidity for cloud formation [kg/kg].
    wpcl : float
        Cloud cover weight for sqrt(precipitation).
    pmaxcl : float
        Maximum precipitation contributing to cloud cover [mm/day].
    clsmax : float
        Maximum stratiform cloud cover (ocean).
    clsminl : float
        Minimum stratiform cloud cover over land (at RH=1).
    gse_s0 : float
        DSE gradient for stratiform cloud cover = 0.
    gse_s1 : float
        DSE gradient for stratiform cloud cover = 1.
    albcl : float
        Cloud albedo (convective/deep clouds).
    albcls : float
        Stratiform cloud albedo.
    abscl1 : float
        SW cloud absorptivity (visible band, max).
    abscl2 : float
        SW cloud absorptivity (general).
    ablcl1 : float
        LW absorptivity of thick clouds (window band, below cloud top).
    ablcl2 : float
        LW absorptivity of thin clouds (window + H₂O bands).
    """

    rhcl1: float = 0.30
    rhcl2: float = 1.00
    qacl: float = 0.20e-3
    wpcl: float = 0.2
    pmaxcl: float = 10.0
    clsmax: float = 0.60
    clsminl: float = 0.15
    gse_s0: float = 0.25
    gse_s1: float = 0.40
    albcl: float = 0.43
    albcls: float = 0.50
    abscl1: float = 0.015
    abscl2: float = 0.15
    ablcl1: float = 12.0
    ablcl2: float = 0.6


@dataclasses.dataclass(frozen=True, slots=True)
class CloudDiagnostic:
    """Diagnostic cloud fields for radiation coupling.

    All fields have shape ``(n_lat, n_lon)`` except where noted.

    Parameters
    ----------
    cloud_cover : jnp.ndarray
        Total cloud fraction [0, 1].
    cloud_top : jnp.ndarray
        Cloud-top level index (0 = TOA, n_levels = surface).
        Set to ``n_levels`` where no cloud exists.
    stratiform_cover : jnp.ndarray
        Stratiform cloud fraction [0, 1] at PBL top.
    cloud_humidity : jnp.ndarray
        Equivalent cloud specific humidity [kg/kg] (from near-surface).
    """

    cloud_cover: jnp.ndarray
    cloud_top: jnp.ndarray
    stratiform_cover: jnp.ndarray
    cloud_humidity: jnp.ndarray


def diagnose_clouds(
    relative_humidity: jnp.ndarray,
    humidity: jnp.ndarray,
    temperature: jnp.ndarray,
    geopotential: jnp.ndarray,
    precipitation_rate: jnp.ndarray,
    convective_mask: jnp.ndarray,
    *,
    gravity: float,
    specific_heat_cp: float,
    land_fraction: jnp.ndarray | None = None,
    config: CloudConfig | None = None,
) -> CloudDiagnostic:
    """Diagnose cloud cover from atmospheric state and precipitation.

    Parameters
    ----------
    relative_humidity : jnp.ndarray
        Relative humidity at full levels, shape ``(n_levels, n_lat, n_lon)``.
    humidity : jnp.ndarray
        Specific humidity [kg/kg], shape ``(n_levels, n_lat, n_lon)``.
    temperature : jnp.ndarray
        Temperature [K] at full levels, shape ``(n_levels, n_lat, n_lon)``.
    geopotential : jnp.ndarray
        Geopotential [m²/s²] at full levels, shape ``(n_levels, n_lat, n_lon)``.
    precipitation_rate : jnp.ndarray
        Total precipitation rate [kg/(m²·s)], shape ``(n_lat, n_lon)``.
    convective_mask : jnp.ndarray
        Boolean mask of convective column, shape ``(n_levels, n_lat, n_lon)``.
        True where convection is active (e.g. from BM tendencies != 0).
    gravity : float
        Gravitational acceleration [m/s²].
    specific_heat_cp : float
        Specific heat at constant pressure [J/(kg·K)].
    land_fraction : jnp.ndarray or None
        Land fraction [0, 1], shape ``(n_lat, n_lon)``. If None, pure ocean.
    config : CloudConfig or None
        Cloud scheme parameters. Uses SPEEDY defaults if None.

    Returns
    -------
    CloudDiagnostic
        Diagnosed cloud fields.
    """
    if config is None:
        config = CloudConfig()

    n_levels, _n_lat, _n_lon = relative_humidity.shape
    nl1 = n_levels - 2  # level above surface (PBL top)
    rrcl = 1.0 / (config.rhcl2 - config.rhcl1)

    # --- Step 1: Cloud cover from max RH in free troposphere ---
    # Mask: only levels where q > qacl, excluding top 2 (strat) and bottom (PBL)
    level_idx = jnp.arange(n_levels)[:, None, None]
    n_strat = 2  # number of stratospheric levels to skip
    free_tropo = (level_idx >= n_strat) & (level_idx <= nl1) & (humidity > config.qacl)

    # RH excess above threshold, masked to free troposphere
    drh = jnp.where(free_tropo, relative_humidity - config.rhcl1, -1.0)

    # Max RH excess in the column and its level index
    max_drh_level = jnp.argmax(drh, axis=0)  # (n_lat, n_lon)
    max_drh = jnp.max(drh, axis=0)  # (n_lat, n_lon)
    cloudc_rh = jnp.maximum(max_drh, 0.0)

    # Also check near-surface RH
    rh_sfc = relative_humidity[nl1]
    sfc_drh = jnp.maximum(rh_sfc - config.rhcl1, 0.0)

    # Use whichever is larger (free tropo or near-surface)
    use_sfc = sfc_drh > cloudc_rh
    cloudc_rh = jnp.where(use_sfc, sfc_drh, cloudc_rh)
    rh_cloud_top = jnp.where(use_sfc, nl1, max_drh_level)

    # --- Step 2: Add precipitation contribution ---
    # Convert from kg/(m²·s) to mm/day: 1 kg/(m²·s) = 86400 mm/day
    precip_mmday = jnp.minimum(config.pmaxcl, 86400.0 * precipitation_rate)
    precip_term = config.wpcl * jnp.sqrt(jnp.maximum(precip_mmday, 0.0))

    cloudc = jnp.minimum(1.0, precip_term + jnp.minimum(1.0, cloudc_rh * rrcl) ** 2)

    # --- Step 3: Cloud-top level ---
    # From convection: highest active level
    conv_any = jnp.any(convective_mask, axis=0)  # (n_lat, n_lon)
    # argmax on reversed mask gives first True from top
    conv_top = n_levels - 1 - jnp.argmax(convective_mask[::-1], axis=0)
    conv_top = jnp.where(conv_any, conv_top, n_levels)

    # Cloud top = min of RH-based and convection-based (higher = smaller index)
    icltop = jnp.minimum(rh_cloud_top, conv_top)
    # Where no clouds, set to n_levels
    icltop = jnp.where(cloudc > 0.0, icltop, n_levels)

    # --- Step 4: Stratiform clouds from boundary-layer stability ---
    # Dry static energy: se = cp*T + g*z
    se = specific_heat_cp * temperature + geopotential
    # Gradient at PBL top
    dse = se[nl1] - se[n_levels - 1]
    dphi = geopotential[nl1] - geopotential[n_levels - 1]
    dphi_safe = jnp.where(jnp.abs(dphi) > 1.0, dphi, 1.0)
    gse = dse / dphi_safe

    rgse = 1.0 / (config.gse_s1 - config.gse_s0)
    fstab = jnp.clip(rgse * (gse - config.gse_s0), 0.0, 1.0)

    clfact = 1.2
    clstr = fstab * jnp.maximum(config.clsmax - clfact * cloudc, 0.0)

    # Land adjustment: minimum stratiform cover over land
    if land_fraction is not None:
        clstrl = jnp.maximum(clstr, config.clsminl) * relative_humidity[n_levels - 1]
        clstr += land_fraction * (clstrl - clstr)

    # --- Step 5: Cloud humidity (proxy for cloud water content) ---
    cloud_q = humidity[nl1]  # specific humidity at level above surface

    return CloudDiagnostic(
        cloud_cover=cloudc,
        cloud_top=icltop,
        stratiform_cover=clstr,
        cloud_humidity=cloud_q,
    )
