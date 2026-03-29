# Boundary Layer and Surface Fluxes

The boundary layer parameterization controls the exchange of momentum, heat, and moisture
between the atmosphere and the surface. Notus uses bulk aerodynamic formulas for the surface
fluxes, with optional stability-dependent transfer coefficients following Louis (1979).

## Bulk aerodynamic surface fluxes

### Sensible heat flux

The surface sensible heat flux (upward positive) is

$$
H = \rho \, c_p \, C_H \, |V| \, (T_s - T_a)
$$

where $\rho$ is the surface air density, $C_H$ is the heat transfer coefficient, $|V|$ is
the near-surface wind speed, $T_s$ is the surface temperature, and $T_a$ is the
atmospheric temperature at the lowest model level.

The corresponding temperature tendency for the lowest atmospheric layer is

$$
\frac{\partial T}{\partial t}\bigg|_\text{sfc} = \frac{g \, H}{\Delta p \, c_p}
$$

where $\Delta p = p_s \Delta\sigma_L$ is the pressure thickness of the lowest layer.

### Latent heat flux

The surface evaporation rate is

$$
E = \rho \, C_H \, |V| \, (q_\text{sat}(T_s) - q_a)
$$

where $q_\text{sat}(T_s)$ is the saturation specific humidity at the surface temperature
and $q_a$ is the specific humidity at the lowest model level. The latent heat flux is $L_v
E$. Over land, the evaporation is modified by a soil wetness factor (see [Surface
Models](surface-models.md)).

### Surface density

The surface air density is estimated from the lowest model level:

$$
\rho = \frac{p_s \, \sigma_\text{lowest}}{R_d \, T_a}
$$

where $\sigma_\text{lowest} = 1 - \Delta\sigma_L / 2$ is the sigma value of the lowest
full level.

## Rayleigh boundary-layer friction

In simplified configurations (Held-Suarez, basic aquaplanet), boundary-layer drag on the
winds is parameterized as Rayleigh friction applied to vorticity and divergence:

$$
\frac{\partial \zeta}{\partial t}\bigg|_\text{fric} = -k_v \, \zeta, \qquad
\frac{\partial \delta}{\partial t}\bigg|_\text{fric} = -k_v \, \delta
$$

where the friction coefficient varies vertically:

$$
k_v = k_f \, \max\left(0, \frac{\sigma - \sigma_b}{1 - \sigma_b}\right)
$$

This ramps linearly from zero above $\sigma_b = 0.7$ to a maximum of $k_f =
1/(1 \text{ day})$ at the surface. When applied implicitly, the damping uses the exact
exponential decay:

$$
\zeta \leftarrow \zeta \, e^{-k_v \Delta t}
$$

which is unconditionally stable regardless of the time step.

## Louis (1979) stability functions

For more realistic simulations, the constant drag coefficient is replaced by
stability-dependent transfer coefficients following Louis (1979). The scheme modifies the
neutral exchange coefficients based on the bulk Richardson number.

### Neutral exchange coefficients

The neutral drag coefficient for momentum is

$$
C_{DN} = \left(\frac{\kappa}{\ln(z_\text{ref} / z_{0m})}\right)^2
$$

and for heat:

$$
C_{HN} = \frac{\kappa^2}{\ln(z_\text{ref} / z_{0m}) \, \ln(z_\text{ref} / z_{0h})}
$$

where $\kappa = 0.4$ is the von Karman constant, $z_\text{ref}$ is the reference height
(estimated from the lowest model level), and $z_{0m}$, $z_{0h}$ are the roughness lengths
for momentum and heat respectively.

The reference height is estimated from the hydrostatic relation:

$$
z_\text{ref} = \frac{R_d T_a}{g} \, \frac{\Delta\sigma_L / 2}{\sigma_\text{lowest}}
$$

### Bulk Richardson number

The surface-layer stability is characterized by the bulk Richardson number:

$$
Ri_b = \frac{g \, z_\text{ref} \, (T_a - T_s)}{T_a \, |V|^2}
$$

Negative values indicate unstable conditions (surface warmer than air), positive values
indicate stable conditions.

### Stability correction functions

**Unstable conditions** ($Ri_b < 0$):

$$
f_m = 1 - \frac{2b \, Ri_b}
{1 + 3b \, c_m \sqrt{C_{DN}} \sqrt{z_\text{ref}/z_{0m}} \sqrt{|Ri_b|}}
$$

$$
f_h = 1 - \frac{3b \, Ri_b}
{1 + 3b \, c_h \sqrt{C_{HN}} \sqrt{z_\text{ref}/z_{0h}} \sqrt{|Ri_b|}}
$$

**Stable conditions** ($Ri_b \ge 0$):

$$
f_m = f_h = \frac{1}{1 + 2b \, Ri_b / \sqrt{1 + d \, Ri_b}}
$$

The effective transfer coefficients are $C_D = C_{DN} f_m$ and $C_H = C_{HN} f_h$.

### Default parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $b$ | 5.0 | Stability parameter |
| $c_m$ | 7.5 | Unstable momentum enhancement |
| $c_h$ | 5.0 | Unstable heat enhancement |
| $d$ | 5.0 | Stable denominator parameter |
| $z_{0m}$ | $10^{-4}$ m | Momentum roughness length |
| $z_{0h}$ | $z_{0m}/10$ | Heat roughness length |
| $Ri_\text{max}$ | 1.0 | Maximum Richardson number (stability clamp) |

## Spatially varying surface properties

The transfer coefficients can vary in space through the surface properties fields:

- **Land fraction**: controls the blending of ocean and land surface temperatures
- **Surface albedo**: may depend on soil moisture over land
- **Roughness lengths** ($z_{0m}$, $z_{0h}$): can differ between ocean and land

## Implicit treatment

For numerical stability, the surface fluxes can be applied implicitly. The atmospheric
temperature and humidity at the lowest level are relaxed toward their surface targets using
the exact exponential decay:

$$
T_a \leftarrow T_\text{target} + (T_a - T_\text{target}) \, e^{-k_\text{sfc} \Delta t}
$$

where the decay rate $k_\text{sfc}$ is derived from the surface flux:

$$
k_\text{sfc} = \frac{g \, \rho \, C_H \, |V|}{\Delta p}
$$

This implicit treatment is unconditionally stable, allowing large time steps without the
surface flux terms driving oscillations.

## References

- Louis, J. F. (1979). A parametric model of vertical eddy fluxes in the atmosphere.
  *Bound.-Layer Meteor.*, 17, 187-202.
- Held, I. M. and Suarez, M. J. (1994). A proposal for the intercomparison of the
  dynamical cores of atmospheric general circulation models. *Bull. Amer. Meteor. Soc.*,
  75, 1825-1830.
