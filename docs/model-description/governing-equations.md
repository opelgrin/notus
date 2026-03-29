# Governing Equations

Notus solves the hydrostatic primitive equations on a rotating sphere. The prognostic
variables are vorticity, divergence, temperature, the logarithm of surface pressure, and
(optionally) specific humidity. This vorticity-divergence formulation is natural for
spectral models because vorticity and divergence are scalar fields, whereas the wind
components $u$ and $v$ are not scalars on the sphere and cannot be directly expanded in
spherical harmonics.

## Coordinate system

The horizontal coordinates are longitude $\lambda$ and latitude $\varphi$. The vertical
coordinate is $\sigma = p / p_s$, where $p$ is pressure and $p_s$ is surface pressure. The
coordinate $\sigma$ ranges from 0 at the top of the atmosphere to 1 at the surface.

## Prognostic variables

| Variable | Symbol | Description |
|----------|--------|-------------|
| Relative vorticity | $\zeta$ | Vertical component of the curl of the horizontal wind |
| Horizontal divergence | $\delta$ | Divergence of the horizontal wind |
| Temperature | $T$ | Atmospheric temperature |
| Log surface pressure | $\ln p_s$ | Logarithm of surface pressure |
| Specific humidity | $q$ | Mass of water vapor per unit mass of moist air (optional) |

## Continuous equations

### Vorticity equation

$$
\frac{\partial \zeta}{\partial t} = -\frac{1}{a\cos\varphi}\frac{\partial}{\partial\lambda}\left[\frac{(\zeta + f)\,v\cos\varphi + \dot{\sigma}\,\partial u/\partial\sigma}{\cos\varphi}\right] + \frac{1}{a\cos\varphi}\frac{\partial}{\partial\varphi}\left[\frac{(\zeta + f)\,u\cos\varphi + \dot{\sigma}\,\partial v/\partial\sigma}{\cos\varphi}\right] + F_\zeta
$$

or, more compactly, using the spectral curl operator on the momentum flux vector
$\mathbf{F}$:

$$
\frac{\partial \zeta}{\partial t} = \text{curl}_z(\mathbf{F}) + F_\zeta
$$

where $f = 2\Omega\sin\varphi$ is the Coriolis parameter, $\Omega$ is the planetary
rotation rate, $\dot{\sigma}$ is the vertical velocity in sigma coordinates, and $F_\zeta$
represents any forcing (friction, diffusion).

### Divergence equation

$$
\frac{\partial \delta}{\partial t} = -\text{div}(\mathbf{F}) - \nabla^2 E - \nabla^2(\Phi + R T_v \ln p_s) + F_\delta
$$

where

$$
E = \frac{u^2 + v^2}{2}
$$

is the kinetic energy, $\Phi$ is the geopotential, $R$ is the gas constant for dry air,
$T_v$ is the virtual temperature, and $F_\delta$ represents forcing. The term $-\nabla^2
\Phi_s$ (where $\Phi_s = g z_s$ is the surface geopotential) is included when topography is
present.

The pressure gradient term $-\nabla^2(\Phi + R T_v \ln p_s)$ couples divergence to
temperature and surface pressure. This coupling supports fast gravity waves and is treated
semi-implicitly (see [Time Integration](time-integration.md)).

### Thermodynamic equation

$$
\frac{\partial T}{\partial t} = -\frac{1}{a\cos\varphi}\left(u\frac{\partial T}{\partial\lambda} + v\cos\varphi\frac{\partial T}{\partial\varphi}\right) - \dot{\sigma}\frac{\partial T}{\partial\sigma} + \kappa T \frac{\omega}{p} + Q
$$

where $\kappa = R/c_p$ is the Poisson constant and $\omega/p$ is the pressure velocity
divided by pressure. The adiabatic heating term $\kappa T (\omega/p)$ represents the
conversion between kinetic and thermal energy during vertical motion. $Q$ represents
diabatic heating (radiation, latent heat release, etc.).

### Continuity equation (surface pressure tendency)

$$
\frac{\partial \ln p_s}{\partial t} = -\sum_{k=1}^{L} D_k^* \, \Delta\sigma_k
$$

where the sum is over all $L$ vertical levels and

$$
D_k^* = \delta_k + \mathbf{v}_k \cdot \nabla \ln p_s
$$

is the mass-flux divergence at level $k$. This equation expresses mass conservation
integrated over the full atmospheric column.

### Moisture equation

$$
\frac{\partial q}{\partial t} = -\frac{1}{a\cos\varphi}\left(u\frac{\partial q}{\partial\lambda} + v\cos\varphi\frac{\partial q}{\partial\varphi}\right) - \dot{\sigma}\frac{\partial q}{\partial\sigma} + E - C
$$

where $E$ is the evaporation source and $C$ is the condensation sink. Moisture is advected
spectrally, with grid-space clipping of negative values after each time step to prevent
unphysical states arising from Gibbs oscillations near sharp gradients.

## Virtual temperature

When moisture is active, the equation of state uses the virtual temperature

$$
T_v = T(1 + \varepsilon_v \, q)
$$

where $\varepsilon_v = R_v/R_d - 1 \approx 0.608$ accounts for the lower molecular weight
of water vapor compared to dry air. The virtual temperature appears in the pressure gradient
and geopotential calculations. The hydrostatic equation becomes $\partial\Phi/\partial\ln\sigma = -R T_v$.

## Momentum flux vector

The combined momentum flux that appears in the vorticity and divergence tendencies is

$$
\mathbf{F} = (\zeta + f)(\hat{k} \times \mathbf{v}) + \dot{\sigma}\frac{\partial\mathbf{v}}{\partial\sigma} + R T_v' \nabla\ln p_s
$$

where $T_v'$ is the virtual temperature anomaly relative to a reference profile (see
[Spectral Dynamics](spectral-dynamics.md) for the semi-implicit splitting). The first term
is the advection of absolute vorticity. The second is the vertical advection of momentum.
The third is the pressure gradient force associated with surface pressure variations.

## Hydrostatic balance

The geopotential at each level is obtained by integrating the hydrostatic equation downward
from the top of the atmosphere:

$$
\Phi_k = \Phi_s + R \sum_{j=1}^{L} G_{kj} \, T_j
$$

where $\Phi_s = g z_s$ is the surface geopotential and $G$ is the geopotential weight
matrix (see [Vertical Discretization](vertical-discretization.md)).

## References

- Bourke, W. (1972). An efficient, one-level, primitive-equation spectral model. *Mon. Wea.
  Rev.*, 100, 683-689.
- Hoskins, B. J. and Simmons, A. J. (1975). A multi-layer spectral model and the
  semi-implicit method. *Quart. J. Roy. Meteor. Soc.*, 101, 637-655.
- Simmons, A. J. and Burridge, D. M. (1981). An energy and angular-momentum conserving
  vertical finite-difference scheme and hybrid vertical coordinates. *Mon. Wea. Rev.*, 109,
  758-766.
