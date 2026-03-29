# Radiation

Notus includes three radiation schemes of increasing complexity. All use the two-stream
approximation (upward and downward fluxes) with layer-by-layer transmissivity computed from
optical depth. The heating rate at each level is proportional to the net flux divergence
across the layer.

## General framework

The radiative heating rate at level $k$ is

$$
Q_k = -\frac{g}{c_p} \frac{F^\uparrow_{k+1/2} - F^\uparrow_{k-1/2}
- F^\downarrow_{k+1/2} + F^\downarrow_{k-1/2}}{\Delta p_k}
$$

where $F^\uparrow$ and $F^\downarrow$ are the upward and downward radiative fluxes at
half-level interfaces, and $\Delta p_k = p_s \Delta\sigma_k$ is the pressure thickness of
layer $k$.

## Frierson gray radiation

The simplest scheme (Frierson et al., 2006) uses a single gray longwave band with
prescribed optical depth and an optional shortwave Beer-Lambert absorption.

### Longwave optical depth

The optical depth from the top of the atmosphere to level $\sigma$ varies with latitude
$\varphi$ as

$$
\tau(\sigma, \varphi) = \tau_0(\varphi) \left[
f_l \, \sigma + (1 - f_l) \, \sigma^\alpha \right]
$$

where the surface optical depth has a meridional gradient:

$$
\tau_0(\varphi) = \tau_e + (\tau_p - \tau_e) \sin^2\varphi
$$

The linear term ($f_l \sigma$) represents well-mixed absorbers and the power-law term
($(1-f_l)\sigma^\alpha$) represents absorbers concentrated near the surface (primarily water
vapor).

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Equatorial optical depth | $\tau_e$ | 6.0 | Surface $\tau$ at equator |
| Polar optical depth | $\tau_p$ | 1.5 | Surface $\tau$ at poles |
| Linear fraction | $f_l$ | 0.1 | Well-mixed absorber fraction |
| Pressure exponent | $\alpha$ | 4.0 | Surface concentration strength |

### Longwave fluxes

The longwave two-stream equations are integrated layer by layer. The upward flux at each
interface is the sum of surface emission (attenuated by the layers above) and emission from
each intervening layer. The downward flux is computed similarly, starting from zero at the
model top.

The layer transmissivity between half levels $k-1/2$ and $k+1/2$ is

$$
\mathcal{T}_k = \exp(-\Delta\tau_k)
$$

where $\Delta\tau_k = \tau_{k+1/2} - \tau_{k-1/2}$. Each layer emits as a gray body at
the local temperature, with emissivity $1 - \mathcal{T}_k$.

### Shortwave absorption

When enabled, shortwave radiation is absorbed following Beer-Lambert:

$$
F_\text{SW}(\sigma) = S_\text{TOA} \, \exp(-\tau_\text{SW}(\sigma))
$$

where the shortwave optical depth is

$$
\tau_\text{SW}(\sigma) = \tau_{\text{SW},0} \, \sigma^{\alpha_\text{SW}}
$$

The top-of-atmosphere shortwave flux $S_\text{TOA}$ is either a fixed function of latitude
(with an insolation distribution parameter $\delta_s = 1.4$) or computed from the seasonal
solar geometry.

## Byrne humidity-dependent radiation

The Byrne scheme (Isca-style) makes the optical depth explicitly dependent on the local
specific humidity, providing a physically motivated water vapor feedback.

### Optical depth

The longwave optical depth gradient is

$$
\frac{d\tau}{d(p/p_0)} = a_\text{LW} + b_\text{LW} \, q
$$

where $a_\text{LW}$ represents well-mixed greenhouse gases and $b_\text{LW} q$ represents
the humidity-dependent water vapor absorption. The optical depth is accumulated from the
model top by summing over layers:

$$
\tau_{k+1/2} = \tau_{k-1/2} + (a_\text{LW} + b_\text{LW} q_k) \,
\frac{\Delta p_k}{p_0}
$$

The shortwave optical depth uses the same structure:

$$
\tau_{\text{SW},k+1/2} = \tau_{\text{SW},k-1/2}
+ (\tau_{\text{SW},0} + a_\text{SW} + b_\text{SW} q_k) \,
\frac{\Delta p_k}{p_0}
$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| $a_\text{LW}$ | 0.8678 | Well-mixed gas LW coefficient |
| $b_\text{LW}$ | 1997.9 | Water vapor LW coefficient |
| $\tau_{\text{SW},0}$ | 0.22 | Base SW optical depth |
| $a_\text{SW}$ | 0.0 | Well-mixed gas SW coefficient |
| $b_\text{SW}$ | 0.2 | Water vapor SW coefficient |

This scheme produces a water vapor feedback: as the atmosphere warms and moistens, the
optical depth increases, trapping more longwave radiation and amplifying the warming.

## SPEEDY multi-band radiation

The most comprehensive scheme, based on Molteni (2003), uses four longwave bands and two
shortwave bands to resolve the wavelength-dependent absorption of different atmospheric
constituents.

### Longwave bands

The four longwave bands target distinct absorption features:

| Band | Absorber | Absorptivity | Description |
|------|----------|-------------|-------------|
| 0 | Dry air | $\alpha_\text{win} = 0.3$ | Atmospheric window |
| 1 | CO$_2$ | $\alpha_\text{CO_2} = 6.0$ | Well-mixed greenhouse gas |
| 2 | H$_2$O (weak) | $\alpha_\text{wv1} = 0.7$ | Humidity-dependent |
| 3 | H$_2$O (strong) | $\alpha_\text{wv2} = 50.0$ | Humidity-dependent |

The fraction of blackbody emission in each band depends on temperature:

$$
f_1 = 0.148 - 3.0 \times 10^{-6}(T - 247)^2
$$

$$
f_2 = 0.356 - 5.2 \times 10^{-6}(T - 282)^2
$$

$$
f_3 = 0.314 + 1.0 \times 10^{-5}(T - 315)^2
$$

$$
f_0 = (1 - \epsilon_\text{LW})(1 - f_1 - f_2 - f_3)
$$

where $\epsilon_\text{LW} = 0.05$ is a small fraction reserved for boundary-layer-only
emission.

For bands 0 and 1 (dry absorbers), the layer transmissivity is

$$
\mathcal{T}_k = \exp\left(-\frac{p_s}{p_0} \, \Delta\sigma_k \, \alpha\right)
$$

For bands 2 and 3 (water vapor), the absorptivity is humidity-dependent:

$$
\mathcal{T}_k = \exp\left(-\frac{p_s}{p_0} \, \Delta\sigma_k \, \alpha \, q_k\right)
$$

The total longwave flux is the sum over all four bands, each computed with its own
transmissivities and emission fractions.

### Cloud interaction with longwave

When clouds are present, additional absorption is applied below the cloud-top level. The
window band receives enhanced absorption ($\alpha_\text{cl1} = 12.0$) for thick clouds, and
a thinner cloud absorption ($\alpha_\text{cl2} = 0.6$) is applied to all bands above the
cloud top.

### Shortwave bands

Two shortwave bands partition the incoming solar flux:

| Band | Fraction | Absorbers | Description |
|------|----------|-----------|-------------|
| 1 (visible) | 0.95 | Dry air, aerosol, weak H$_2$O | Reaches surface |
| 2 (near-IR) | 0.05 | Strong H$_2$O | Absorbed in atmosphere |

**Visible band** absorptivity per layer:

$$
\alpha_\text{vis} = \alpha_\text{dry} + \alpha_\text{aer} \, \sigma_k^2
+ \alpha_\text{wv1} \, q_k
$$

where the aerosol absorption is weighted by $\sigma^2$ (concentrated near the surface) and
the weak water vapor absorption scales linearly with humidity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\alpha_\text{dry}$ | 0.033 | Dry-air SW absorption |
| $\alpha_\text{aer}$ | 0.033 | Aerosol SW absorption |
| $\alpha_\text{wv1}$ | 0.022 | Weak H$_2$O SW absorption |
| $\alpha_\text{wv2}$ | 15.0 | Strong H$_2$O SW absorption (near-IR) |

**Near-IR band** absorptivity per layer:

$$
\alpha_\text{nir} = \alpha_\text{wv2} \, q_k
$$

### Cloud interaction with shortwave

Clouds reflect a fraction of the incoming shortwave before it reaches the surface:

$$
\text{Cloud reflection} = 1 - \alpha_\text{cl} \cdot C_\text{cloud}
$$

where $\alpha_\text{cl} = 0.43$ for convective clouds and $\alpha_\text{cls} = 0.50$ for
stratiform clouds. The reflected fraction is subtracted from the downward shortwave flux at
the cloud-top level.

### Surface emission

The surface emits longwave radiation as a near-blackbody:

$$
F^\uparrow_\text{sfc} = \epsilon_s \, \sigma_\text{SB} \, T_s^4
$$

where $\epsilon_s = 0.98$ is the surface emissivity and $\sigma_\text{SB} = 5.670374419
\times 10^{-8}$ W/(m$^2$K$^4$) is the Stefan-Boltzmann constant.

## References

- Frierson, D. M. W., Held, I. M., and Zurita-Gotor, P. (2006). A gray-radiation
  aquaplanet moist GCM. Part I: Static stability and eddy scale. *J. Atmos. Sci.*, 63,
  2548-2566.
- Byrne, M. P. and O'Gorman, P. A. (2013). Land-ocean warming contrast over a wide range
  of climates: Convective quasi-equilibrium theory and idealized simulations. *J. Climate*,
  26, 4000-4016.
- Molteni, F. (2003). Atmospheric simulations using a GCM with simplified physical
  parameterizations. I: Model climatology and variability in multi-decadal experiments.
  *Climate Dyn.*, 20, 175-191.
