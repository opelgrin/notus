# Surface Models and Coupled Integration

This section describes the surface boundary conditions (slab ocean, bucket land model),
the solar geometry that drives the shortwave forcing, the treatment of topography, and how
these components are coupled to the atmospheric model.

## Prescribed sea surface temperature

The simplest surface boundary condition is a fixed, zonally symmetric SST profile following
Frierson et al. (2006):

$$
T_s(\varphi) = \max\left(T_\text{min}, \; T_\text{min} + \Delta T \, \exp\left(-\frac{\varphi^2}{2\varphi_w^2}\right)\right)
$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| $T_\text{min}$ | 271 K | Temperature floor (prevents sea ice) |
| $\Delta T$ | 29 K | Equator-to-pole temperature difference |
| $\varphi_w$ | 26° | Latitude width of the warm pool |

This produces a Gaussian-shaped warm pool centered on the equator with a meridional
$e$-folding scale of about 26° latitude. The maximum SST is $T_\text{min} + \Delta T = 300$
K at the equator.

## Slab ocean

The slab ocean model represents the upper ocean as a thermodynamic mixed layer with no
horizontal heat transport. The SST evolves according to the net surface energy balance:

$$
C_\text{ocean} \frac{dT_s}{dt} = F_\text{net} + Q_\text{flux}
$$

where $C_\text{ocean} = \rho_w c_w h$ is the mixed-layer heat capacity, $F_\text{net}$ is
the net downward energy flux at the surface, and $Q_\text{flux}$ is a prescribed
convergence of oceanic heat transport (q-flux).

### Net surface energy balance

$$
F_\text{net} = S_\text{abs} + F^\downarrow_\text{LW} - \sigma_\text{SB} T_s^4 - H - L_v E
$$

where $S_\text{abs}$ is the absorbed shortwave radiation (after surface albedo),
$F^\downarrow_\text{LW}$ is the downwelling longwave, $\sigma_\text{SB} T_s^4$ is the
surface longwave emission, $H$ is the sensible heat flux, and $L_v E$ is the latent heat
flux.

### Implicit time stepping

The SST tendency depends on $T_s$ through the longwave emission and surface fluxes. To
maintain stability, the update uses a linearized implicit scheme:

$$
T_s^{n+1} = T_s^n + \Delta t \, \frac{F_\text{net} + Q_\text{flux}}{C_\text{ocean} - \Delta t \, \partial F / \partial T_s}
$$

The flux derivative is

$$
\frac{\partial F}{\partial T_s} = -4\sigma_\text{SB} T_s^3 - \rho \, c_p C_H |V| - \rho \, L_v C_H |V| \frac{dq_\text{sat}}{dT_s}
$$

Since all three terms are negative, the denominator is always larger than
$C_\text{ocean}$, guaranteeing unconditional stability.

### Default parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Mixed-layer depth | 50 m | Thermodynamic depth |
| $\rho_w$ | 1025 kg/m$^3$ | Seawater density |
| $c_w$ | 3994 J/(kg K) | Seawater specific heat |
| $C_\text{ocean}$ | $2.05 \times 10^8$ J/(m$^2$ K) | Heat capacity |

### Q-flux diagnosis

For experiments that require a specific SST climatology (e.g., to match an observed
pattern), the q-flux is diagnosed from a prescribed-SST spinup run. The q-flux is the
negative of the time-mean net surface energy flux:

$$
Q_\text{flux} = -\langle F_\text{net} \rangle
$$

where $\langle \cdot \rangle$ denotes a temporal average over the spinup period (typically
100 days spinup followed by 200 days of averaging). When this q-flux is applied, the slab
ocean reproduces the prescribed SST in equilibrium while allowing SST perturbations in
response to atmospheric changes.

## Bucket land surface model

The bucket model (Manabe, 1969) represents the land surface with two prognostic variables:
soil temperature and a bucket water reservoir.

### Soil temperature

The soil temperature evolves as

$$
C_\text{soil} \frac{dT_\text{soil}}{dt} = F_\text{net,land}
$$

where $C_\text{soil}$ is the soil heat capacity and $F_\text{net,land}$ is the net surface
energy flux over land (same form as the ocean balance, but with modified albedo and
evaporation). The same implicit linearization as the slab ocean is used.

### Bucket hydrology

The bucket depth $W$ (in meters of water equivalent) evolves as

$$
\frac{dW}{dt} = P - \beta E_\text{pot} - R
$$

where $P$ is precipitation, $E_\text{pot}$ is the potential evaporation rate, and $R$ is
runoff (excess above the bucket capacity). The bucket depth is bounded between 0 and
$W_\text{max}$:

$$
W^{n+1} = \text{clip}\left(W^n + \Delta t \, \frac{P - \beta E_\text{pot}}{\rho_w}, \; 0, \; W_\text{max}\right)
$$

### Evaporation reduction

The evaporation efficiency factor $\beta$ depends on the bucket fill fraction:

$$
\beta = \min\left(1, \; \frac{W}{W_\text{crit}}\right)
$$

where $W_\text{crit} = f_\text{crit} \cdot W_\text{max}$ is the critical depth below which
evaporation is limited. When the bucket is above the critical depth, potential evaporation
occurs; below it, evaporation decreases linearly to zero.

### Moisture-dependent albedo

The land surface albedo varies with soil moisture:

$$
\alpha_\text{land} = \alpha_\text{dry} + (\alpha_\text{wet} - \alpha_\text{dry}) \min\left(1, \frac{W}{W_\text{max}}\right)
$$

Dry soils are more reflective than wet soils. The effective surface albedo blends ocean and
land values weighted by the land fraction.

### Default parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $C_\text{soil}$ | $4 \times 10^6$ J/(m$^2$ K) | Soil heat capacity (~2 m soil) |
| $W_\text{max}$ | 0.15 m | Bucket capacity |
| $f_\text{crit}$ | 0.75 | Critical fraction |
| $\alpha_\text{dry}$ | 0.35 | Dry-soil albedo |
| $\alpha_\text{wet}$ | 0.20 | Wet-soil albedo |

## Diagnostic clouds

A diagnostic cloud scheme based on the SPEEDY model (Molteni, 2003) provides cloud cover
for the radiation calculation. Cloud cover is diagnosed from relative humidity and
precipitation.

### Convective and humidity-based clouds

Cloud cover in the free troposphere is

$$
C = \min\left(1, \; w_p \sqrt{P_\text{mm/day}} + \min\left(1, \; (r \cdot \Delta\text{RH})^2 \right)\right)
$$

where $\Delta\text{RH} = \max(0, \text{RH} - \text{RH}_1)$ is the relative humidity excess
above the threshold, $r = 1/(\text{RH}_2 - \text{RH}_1)$ is the normalization, and $w_p$
weights the precipitation contribution.

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\text{RH}_1$ | 0.30 | Cloud-free RH threshold |
| $\text{RH}_2$ | 1.00 | Overcast RH threshold |
| $w_p$ | 0.20 | Precipitation weight |

### Stratiform clouds

Low-level stratiform clouds are diagnosed from the dry static energy (DSE) gradient across
the boundary-layer top:

$$
\text{DSE} = c_p T + g z
$$

$$
C_\text{strat} = f_\text{stab} \, \max(0, \; C_\text{strat,max} - 1.2 \, C)
$$

where $f_\text{stab}$ is a stability function that increases with the DSE inversion
strength. Stratiform clouds form when the boundary layer is capped by a temperature
inversion (large DSE gradient).

### Cloud-radiation coupling

Cloud cover modifies both shortwave and longwave radiation:

- **Shortwave**: clouds reflect incoming solar radiation with albedos of 0.43 (convective)
  and 0.50 (stratiform).
- **Longwave**: clouds enhance atmospheric absorption in the window band below the
  cloud-top level.

## Solar geometry and seasonal cycle

When seasonal forcing is enabled, the incoming solar radiation varies with day of year
through changes in solar declination and Earth-Sun distance.

### Orbital parameters

| Parameter | Earth default | Description |
|-----------|--------------|-------------|
| Eccentricity | 0.0167086 | Orbital eccentricity |
| Obliquity | 23.44° | Axial tilt |
| Longitude of perihelion | 282.9° | Orientation of orbit |
| Days per year | 365.25 | Orbital period |

### Solar declination

The declination angle (latitude of the sub-solar point) varies as

$$
\delta = \arcsin(\sin\epsilon \, \sin\lambda_s)
$$

where $\epsilon$ is the obliquity and $\lambda_s = 2\pi \, d / N_\text{year}$ is the true
solar longitude measured from the vernal equinox ($d$ is the day of year).

### Earth-Sun distance

The Earth-Sun distance varies with orbital position. To first order in eccentricity:

$$
\left(\frac{a}{r}\right)^2 \approx 1 + 2e\cos(\lambda_s - \omega)
$$

where $e$ is the eccentricity and $\omega$ is the longitude of perihelion.

### Daily-mean insolation

The daily-mean insolation at latitude $\varphi$ is obtained by analytically integrating
over the diurnal cycle (Hartmann, 2016):

$$
\bar{S}(\varphi, d) = \frac{S_0}{\pi} \left(\frac{a}{r}\right)^2 \left[h_0 \sin\varphi \sin\delta + \cos\varphi \cos\delta \sin h_0 \right]
$$

where $h_0$ is the sunrise hour angle:

$$
h_0 = \arccos(-\tan\varphi \, \tan\delta)
$$

clamped to $[0, \pi]$ to handle polar day ($h_0 = \pi$) and polar night ($h_0 = 0$).

### Fixed insolation

When seasonal forcing is not used, a fixed insolation profile distributes the solar
constant across latitudes:

$$
\bar{S}(\varphi) = \frac{S_0}{4}\left[1 + \frac{\delta_s}{4}(1 - 3\sin^2\varphi)\right]
$$

where $\delta_s = 1.4$ is an insolation distribution parameter that concentrates more
energy in the tropics than a uniform $S_0/4$ distribution.

## Topography

Notus supports idealized topography through analytic orography generators and spectral
smoothing.

### Orography generators

**Gaussian mountain**: an isolated peak at $(\varphi_0, \lambda_0)$:

$$
z(\varphi, \lambda) = h_0 \exp\left(-\frac{(\varphi - \varphi_0)^2 + (\lambda - \lambda_0)^2}{\sigma^2}\right)
$$

**Zonal ridge**: a latitude-only ridge:

$$
z(\varphi) = h_0 \exp\left(-\frac{(\varphi - \varphi_0)^2}{\sigma^2}\right)
$$

**Sinusoidal mountains**: a wavenumber-$k$ chain for stationary Rossby wave tests:

$$
z(\varphi, \lambda) = h_0 \cos^2\varphi \cos(k\lambda)
$$

The surface geopotential is $\Phi_s = g z$.

### Spectral smoothing

To avoid Gibbs ringing when sharp topographic features are represented spectrally, two
smoothing filters are available:

**Lanczos $\sigma$-factor**:

$$
\sigma(n) = \frac{\sin(\pi n / (T+1))}{\pi n / (T+1)}
$$

**Exponential smoothing**:

$$
w(n) = \exp\left(-\kappa \left(\frac{n}{T+1}\right)^2\right)
$$

where $\kappa = 2s$ and $s$ is the filter order.

### Surface pressure initialization

Given the surface geopotential, the initial surface pressure is set from hydrostatic
balance:

$$
\ln\frac{p_s}{p_0} = -\frac{\Phi_s}{R \, T_\text{ref}}
$$

This ensures the initial state is hydrostatically consistent with the topography.

## Coupled atmosphere-surface integration

The coupled stepper integrates the atmosphere, slab ocean, and optional bucket land surface
as a single system.

### Time step sequence

Each coupled time step proceeds as follows:

1. **Atmospheric IMEX step**: advance vorticity, divergence, temperature, humidity, and
   log surface pressure using the leapfrog scheme with semi-implicit gravity waves.
2. **Spectral filtering and humidity clipping**: post-process the atmospheric state.
3. **Rayleigh friction**: apply boundary-layer drag implicitly (exponential decay).
4. **Extract surface conditions**: transform lowest-level winds, temperature, and humidity
   to grid space; compute transfer coefficients.
5. **Radiation**: compute shortwave and longwave surface fluxes.
6. **Ocean SST update**: advance the slab ocean with the implicit scheme.
7. **Land update** (if present): advance soil temperature and bucket hydrology.
8. **Implicit surface fluxes**: relax lowest-level atmospheric temperature and humidity
   toward the surface target values using exponential decay.

### Surface target values

The surface temperature and humidity targets blend ocean and land contributions:

$$
T_\text{target} = (1 - f_\text{land}) \, T_\text{SST}
+ f_\text{land} \, T_\text{soil}
$$

$$
q_\text{target} = (1 - f_\text{land}) \, q_\text{sat}(T_\text{SST}) + f_\text{land} \, \beta \, q_\text{sat}(T_\text{soil})
$$

where $f_\text{land}$ is the land fraction and $\beta$ is the bucket evaporation efficiency.

### Prescribed-SST spinup

For experiments that need a diagnosed q-flux, a spinup procedure is available. The model
runs with prescribed SST for a specified period (typically 100 days discarded + 200 days
averaged), accumulating the time-mean net surface flux. The negative of this flux provides
the q-flux that, when applied in a slab-ocean run, reproduces the prescribed SST in
equilibrium.

## References

- Manabe, S. (1969). Climate and the ocean circulation: I. The atmospheric circulation and
  the hydrology of the Earth's surface. *Mon. Wea. Rev.*, 97, 739-774.
- Frierson, D. M. W., Held, I. M., and Zurita-Gotor, P. (2006). A gray-radiation
  aquaplanet moist GCM. Part I: Static stability and eddy scale. *J. Atmos. Sci.*, 63,
  2548-2566.
- Molteni, F. (2003). Atmospheric simulations using a GCM with simplified physical
  parameterizations. I: Model climatology and variability in multi-decadal experiments.
  *Climate Dyn.*, 20, 175-191.
- Hartmann, D. L. (2016). *Global Physical Climatology*. 2nd ed. Elsevier.
