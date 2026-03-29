# Vertical Discretization

Notus uses a sigma coordinate with Lorenz staggering. This section describes the vertical
grid structure, the discrete hydrostatic equation, and the computation of vertical transport
quantities.

## Sigma coordinate

The vertical coordinate is defined as

$$
\sigma = \frac{p}{p_s}
$$

where $p$ is pressure and $p_s$ is surface pressure. The coordinate ranges from $\sigma =
0$ at the model top to $\sigma = 1$ at the surface. This terrain-following coordinate
ensures that the lowest model level follows the topography.

## Lorenz staggering

Prognostic variables (temperature, vorticity, divergence, humidity) are defined at **full
levels** (layer midpoints), while the vertical velocity $\dot{\sigma}$ is defined at
**half levels** (layer interfaces). The vertical grid is indexed as follows:

```
half level 0      sigma = 0         (model top)
full level 0      sigma_full[0]     (topmost layer midpoint)
half level 1      sigma_half[1]
full level 1      sigma_full[1]
    ...
full level L-1    sigma_full[L-1]   (lowest layer midpoint)
half level L      sigma = 1         (surface)
```

The layer thickness for level $k$ is

$$
\Delta\sigma_k = \sigma_{k+1/2} - \sigma_{k-1/2}
$$

and the thicknesses sum to unity: $\sum_{k=0}^{L-1} \Delta\sigma_k = 1$.

## Level placement

Two strategies for placing sigma levels are available.

**Uniform levels** distribute layers equally in $\sigma$, giving constant
$\Delta\sigma_k$.

**Standard levels** concentrate resolution near the surface (to resolve the boundary layer),
near the tropopause (to capture the temperature inversion and jet structure), and near the
model top (to provide a sponge layer). The placement uses a smooth weighting function that
blends three Gaussian bumps centered on these regions of interest.

## Sigma ratios

The geopotential integration requires logarithmic sigma ratios defined at half levels:

$$
\alpha_j = \frac{1}{2} \ln\frac{\sigma_\text{full}[j+1]}{\sigma_\text{full}[j]} \qquad \text{for } j < L-1
$$

$$
\alpha_{L-1} = -\ln \sigma_\text{full}[L-1] \qquad \text{(bottom level)}
$$

These arise from the analytical integration of the hydrostatic equation between adjacent
full levels, assuming $T$ is constant within each layer.

## Geopotential integration

The geopotential at each full level is obtained from the hydrostatic relation
$\partial\Phi / \partial\ln\sigma = -R T$ (or $-R T_v$ when moisture is active), integrated
upward from the surface:

$$
\Phi_k = \Phi_s + R \sum_{j=0}^{L-1} G_{kj} \, T_j
$$

where $\Phi_s = g z_s$ is the surface geopotential. The weight matrix $G$ is upper
triangular with entries

$$
G_{kk} = R \, \alpha_k
$$

$$
G_{kj} = R \, (\alpha_j + \alpha_{j-1}) \qquad \text{for } j > k
$$

$$
G_{kj} = 0 \qquad \text{for } j < k
$$

The structure reflects the fact that the geopotential at level $k$ depends on the
temperature in all layers between $k$ and the surface. The diagonal entry $R\alpha_k$
accounts for the half-layer between level $k$ and the interface below it.

## Surface pressure tendency

The continuity equation, integrated over the full atmospheric column, gives the surface
pressure tendency:

$$
\frac{\partial \ln p_s}{\partial t} = -\sum_{k=0}^{L-1} D_k^* \, \Delta\sigma_k
$$

where

$$
D_k^* = \delta_k + \mathbf{v}_k \cdot \nabla\ln p_s
$$

is the mass-flux divergence at level $k$, combining the horizontal divergence with the
advection of the surface pressure field.

## Vertical velocity

The vertical velocity in sigma coordinates, $\dot{\sigma}$, is computed at half levels from
the continuity equation. Defining the cumulative mass flux from the model top:

$$
C_k = \sum_{j=0}^{k-1} D_j^* \, \Delta\sigma_j
$$

and the column-integrated mass flux $C_L = \sum_{j=0}^{L-1} D_j^* \, \Delta\sigma_j$, the
vertical velocity at half level $k + 1/2$ is

$$
\dot{\sigma}_{k+1/2} = \sigma_{k+1/2} \, C_L - C_k
$$

This construction automatically satisfies the boundary conditions $\dot{\sigma} = 0$ at
$\sigma = 0$ (model top) and $\sigma = 1$ (surface).

## Vertical advection

Vertical advection of any field $f$ defined at full levels uses centered second-order finite
differences. First, the vertical gradient at half levels is computed:

$$
\left.\frac{\partial f}{\partial\sigma}\right|_{k+1/2} = \frac{f_{k+1} - f_k}{\sigma_\text{full}[k+1] - \sigma_\text{full}[k]}
$$

with zero-gradient boundary conditions at the top and bottom. The vertical advection at full
levels is then

$$
-\dot{\sigma}\frac{\partial f}{\partial\sigma}\bigg|_k = -\frac{1}{2}\left(\dot{\sigma}_{k-1/2} \left.\frac{\partial f}{\partial\sigma}\right|_{k-1/2} + \dot{\sigma}_{k+1/2} \left.\frac{\partial f}{\partial\sigma}\right|_{k+1/2}\right)
$$

The averaging of the two half-level products ensures second-order accuracy and prevents
spurious energy generation at the grid scale.

## Pressure velocity

The quantity $\omega/p$ (pressure velocity divided by pressure) appears in the adiabatic
heating term of the thermodynamic equation. Following Durran, it is computed at full levels
as

$$
\left(\frac{\omega}{p}\right)_k = \mathbf{v}_k \cdot \nabla\ln p_s - \frac{1}{\Delta\sigma_k}\left(\alpha_k \, F_k + \alpha_{k-1} \, F_{k-1}\right)
$$

where

$$
F_k = \sum_{j=0}^{k-1} D_j^* \, \Delta\sigma_j
$$

is the cumulative mass flux divergence from the top. The first term represents the
horizontal contribution and the second the vertical redistribution.

## References

- Simmons, A. J. and Burridge, D. M. (1981). An energy and angular-momentum conserving
  vertical finite-difference scheme and hybrid vertical coordinates. *Mon. Wea. Rev.*, 109,
  758-766.
- Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics*. 2nd ed. Springer.
