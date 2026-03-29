# Standard Test Cases

Notus is validated against standard idealized benchmarks from the atmospheric modeling
community. These test cases progress from dry dynamics to full moist physics, each targeting
specific aspects of the model.

## Held-Suarez (1994)

The Held-Suarez benchmark is the standard intercomparison test for dry atmospheric dynamical
cores. It replaces all physical parameterizations with two simple forcings: Newtonian
relaxation of temperature toward a prescribed equilibrium profile, and Rayleigh friction in
the boundary layer.

### Equilibrium temperature

$$
T_\text{eq}(\varphi, p) = \max\left(T_\text{min}, \;
\left[315 - \Delta T_y \sin^2\varphi
- \Delta\theta_z \ln\frac{p}{p_0} \cos^2\varphi\right]
\left(\frac{p}{p_0}\right)^\kappa \right)
$$

This produces a warm equator, cold poles, and a statically stable stratosphere (where
$T_\text{eq}$ is clamped to $T_\text{min}$).

### Newtonian relaxation

The temperature is relaxed toward $T_\text{eq}$ with a latitude- and height-dependent rate:

$$
k_T = k_a + (k_s - k_a) \, \max\left(0, \frac{\sigma - \sigma_b}{1 - \sigma_b}\right)
\cos^4\varphi
$$

The relaxation is fast near the surface at low latitudes ($\sim$4-day timescale) and slow
in the free atmosphere ($\sim$40-day timescale).

### Rayleigh friction

Boundary-layer drag acts on vorticity and divergence with rate

$$
k_v = k_f \, \max\left(0, \frac{\sigma - \sigma_b}{1 - \sigma_b}\right)
$$

ramping from zero above $\sigma_b = 0.7$ to $k_f = 1/(1 \text{ day})$ at the surface.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\Delta T_y$ | 60 K | Equator-to-pole temperature contrast |
| $\Delta\theta_z$ | 10 K | Vertical stability parameter |
| $T_\text{min}$ | 200 K | Stratospheric temperature floor |
| $k_a$ | $1/(40 \text{ days})$ | Free-atmosphere relaxation rate |
| $k_s$ | $1/(4 \text{ days})$ | Surface relaxation rate |
| $k_f$ | $1/(1 \text{ day})$ | Surface friction rate |
| $\sigma_b$ | 0.7 | Boundary-layer top |

### Standard configuration

The canonical setup uses T42 resolution with 20 vertical levels, a 600 s time step, and is
integrated for 1200 days (200 days spinup, 1000 days averaging). A reduced-resolution
regression test uses T21L20 for 300 days.

### Initial conditions

The model is initialized with an isothermal rest state at 264 K, uniform surface pressure
at $10^5$ Pa, and a small random temperature perturbation (1 K amplitude) to break the
hemispheric symmetry and trigger baroclinic instability.

### Expected climatology

The Held-Suarez climatology should show:

- Subtropical jets at 25--40 m/s in the upper troposphere (around 200 hPa)
- Midlatitude surface westerlies of 5--15 m/s
- An equator-to-pole surface temperature gradient of 30--50 K
- Eddy kinetic energy peaking in midlatitudes at upper levels
- Approximate hemispheric symmetry

## Jablonowski-Williamson (2006)

The Jablonowski-Williamson test case provides an analytically balanced initial condition for
testing baroclinic wave development. Unlike Held-Suarez, the initial state is a known exact
solution, allowing verification of the dynamical core's ability to maintain balance and
correctly develop baroclinic instability.

### Balanced initial state

The initial wind field is a midlatitude jet:

$$
u(\varphi, \eta) = u_0 \cos^{3/2}\eta_v \cos^2\varphi
$$

where $\eta_v = (\eta - \eta_0)\pi/2$ and $\eta = \sigma$ is the vertical coordinate. The
jet peaks at about 35 m/s in the upper troposphere.

The initial temperature is analytically derived from thermal wind balance:

$$
T'(\varphi, \eta) = \frac{3\eta\pi u_0}{4R} \sin\eta_v \sqrt{\cos\eta_v}
\left[A(\varphi) \cdot 2u_0 \cos^{3/2}\eta_v + B(\varphi) \cdot a\Omega\right]
$$

where

$$
A(\varphi) = -2\sin^6\varphi\left(\cos^2\varphi + \tfrac{1}{3}\right) + \tfrac{10}{63}
$$

$$
B(\varphi) = \tfrac{8}{5}\cos^3\varphi\left(\sin^2\varphi + \tfrac{2}{3}\right)
- \frac{\pi}{4}
$$

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $u_0$ | 35 m/s | Jet strength |
| $T_0$ | 288 K | Reference surface temperature |
| $\Delta T$ | $4.8 \times 10^5$ K | Stratospheric temperature parameter |
| $\gamma$ | 0.005 K/m | Temperature lapse rate |
| $\eta_0$ | 0.252 | Reference eta level |
| $\eta_\text{tropo}$ | 0.2 | Tropopause level |

### Steady-state test

Without perturbation, the balanced state should remain stationary. This tests the
dynamical core's ability to preserve a non-trivial balanced flow. With double precision
arithmetic, the state remains steady for at least 30 days.

### Baroclinic wave test

A localized velocity perturbation triggers baroclinic instability:

$$
u_\text{pert} = u_p \exp\left(-\frac{r^2}{r_p^2}\right)
$$

centered at (20°E, 40°N) with $u_p = 1$ m/s and $r_p = 0.1a$ (where $a$ is the planet
radius). The perturbation triggers a growing baroclinic wave that develops into a
cyclone-anticyclone dipole over the following 10--15 days.

## Aquaplanet configurations

Beyond the dry benchmarks, Notus supports several aquaplanet configurations with increasing
physical complexity.

### Dry aquaplanet

Uses Frierson gray longwave radiation, dry convective adjustment, and prescribed SST. This
provides a step between the idealized Held-Suarez forcing and full moist physics: the
radiation and surface fluxes are physically based, but moisture is absent.

### Moist aquaplanet

Adds specific humidity as a prognostic variable, large-scale condensation, simplified
Betts-Miller convection, and surface evaporation. The standard setup uses T21L20, a 900 s
time step, and initial relative humidity of 70% in the troposphere (decaying above $\sigma
= 0.3$). This is the core moist benchmark for validating the interaction between dynamics,
moisture, and convection.

### Seasonal aquaplanet

Extends the moist aquaplanet with a time-varying solar insolation from the full orbital
geometry (obliquity, eccentricity, perihelion longitude). This produces a seasonal cycle
with migrating ITCZ and solsticial Hadley cell asymmetry. Typical runs use 2 years
(365 days spinup + 365 days averaging with monthly output).

### Coupled slab-ocean aquaplanet

Replaces the prescribed SST with an interactive slab ocean, allowing the SST to respond to
atmospheric changes. A q-flux diagnosed from a prescribed-SST spinup maintains the desired
mean SST pattern. This configuration tests the atmosphere-ocean coupling and the stability
of the coupled system.

### Topography experiments

Idealized topography (Gaussian mountain, zonal ridge, sinusoidal chain) can be added to any
configuration. The surface geopotential modifies the divergence tendency and the initial
surface pressure field. Spectral smoothing prevents Gibbs artifacts.

## Planetary constants

All test cases use the Earth planetary constants:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a$ | $6.371 \times 10^6$ m | Planetary radius |
| $\Omega$ | $7.292 \times 10^{-5}$ rad/s | Rotation rate |
| $g$ | 9.80616 m/s$^2$ | Gravitational acceleration |
| $R_d$ | 287.04 J/(kg K) | Gas constant (dry air) |
| $c_p$ | 1004.64 J/(kg K) | Specific heat at constant pressure |
| $\kappa$ | 0.2857 | $R_d / c_p$ |
| $p_0$ | $10^5$ Pa | Reference surface pressure |
| $S_0$ | 1360 W/m$^2$ | Solar constant |
| $\alpha_s$ | 0.06 | Surface albedo |

## References

- Held, I. M. and Suarez, M. J. (1994). A proposal for the intercomparison of the
  dynamical cores of atmospheric general circulation models. *Bull. Amer. Meteor. Soc.*,
  75, 1825-1830.
- Jablonowski, C. and Williamson, D. L. (2006). A baroclinic instability test case for
  atmospheric model dynamical cores. *Quart. J. Roy. Meteor. Soc.*, 132, 2943-2975.
- Frierson, D. M. W., Held, I. M., and Zurita-Gotor, P. (2006). A gray-radiation
  aquaplanet moist GCM. Part I: Static stability and eddy scale. *J. Atmos. Sci.*, 63,
  2548-2566.
