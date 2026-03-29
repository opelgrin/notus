# Moist Physics

This section describes the moisture thermodynamics, large-scale condensation, and convective
parameterizations. When moisture is active, specific humidity $q$ is carried as a prognostic
variable, advected spectrally alongside the other fields. Grid-space clipping of negative
values is applied after each time step to correct Gibbs oscillations.

## Saturation thermodynamics

### Saturation vapor pressure

The saturation vapor pressure over liquid water follows the Bolton (1980) formula:

$$
e_\text{sat}(T) = 611.2 \, \exp\left(\frac{17.67\,(T - 273.15)}{T - 29.65}\right)
\quad [\text{Pa}]
$$

This approximation is accurate to within 0.3% for temperatures between $-35$°C and $+35$°C.

### Saturation specific humidity

The saturation specific humidity at pressure $p$ is

$$
q_\text{sat}(T, p) = \frac{\varepsilon \, e_\text{sat}(T)}
{p - (1 - \varepsilon) \, e_\text{sat}(T)}
$$

where $\varepsilon = R_d / R_v \approx 0.622$ is the ratio of the gas constants for dry air
and water vapor.

### Clausius-Clapeyron derivative

Several schemes require the derivative of $q_\text{sat}$ with respect to temperature. From
the Clausius-Clapeyron relation:

$$
\frac{dq_\text{sat}}{dT} \approx \frac{L_v \, \varepsilon \, q_\text{sat}}{R_d \, T^2}
$$

This appears in the implicit condensation scheme and in the linearized surface flux
calculations.

### Moist pseudoadiabat

The moist pseudoadiabatic lapse rate is used to construct reference profiles for the
Betts-Miller convection scheme. A parcel lifted from the surface follows

$$
\frac{dT}{dp} = \frac{R_d T + L_v q_\text{sat}}
{c_p p + L_v^2 \varepsilon \, q_\text{sat} / (R_d T^2)}
$$

This is integrated upward in log-pressure from the surface to each model level using 200
small steps for accuracy. The numerator is the total energy available for expansion; the
denominator accounts for the latent heating feedback on the saturation point.

## Dry convective adjustment

Dry convective adjustment (Manabe and Strickler, 1964) removes gravitational instability by
mixing adjacent layers until the potential temperature no longer decreases with height.

The potential temperature at level $k$ is

$$
\theta_k = T_k \, \sigma_k^{-\kappa}
$$

where $\kappa = R/c_p$. When $\theta_k > \theta_{k+1}$ (the layer above is colder than it
should be in a stable atmosphere), the pair is mixed to a common potential temperature that
conserves the column-integrated enthalpy:

$$
\theta_\text{new} = \frac{\Delta\sigma_k \, T_k + \Delta\sigma_{k+1} \, T_{k+1}}
{\Delta\sigma_k \, \sigma_k^\kappa + \Delta\sigma_{k+1} \, \sigma_{k+1}^\kappa}
$$

The adjustment sweeps bottom-to-top and is repeated for a configurable number of iterations
(default 3) to handle multi-layer instabilities.

## Large-scale condensation

Large-scale condensation removes supersaturation using an implicit scheme (Frierson et al.,
2006, Eq. 21) that accounts for the latent heating feedback. When $q > q_\text{sat}$,
simply setting $q = q_\text{sat}$ would underestimate the condensation because the latent
heat release raises the temperature and hence $q_\text{sat}$ itself.

The moisture increment is

$$
\Delta q = -\frac{q - \text{rh}_\text{crit} \, q_\text{sat}}
{1 + \dfrac{L_v^2 \varepsilon \, q_\text{sat}}{c_p R_d T^2}}
$$

The denominator is the Clausius-Clapeyron correction: it is always $> 1$, reducing the
condensation relative to the naive estimate. The temperature is adjusted by the corresponding
latent heating:

$$
\Delta T = -\frac{L_v}{c_p} \Delta q
$$

The scheme iterates (default 3 times) to converge when the supersaturation is large. The
threshold relative humidity $\text{rh}_\text{crit}$ is set to 1.0 (saturation) by default.

The resulting precipitation rate is

$$
P = -\frac{\Delta q \, \Delta p}{g \, \Delta t}
$$

accumulated over all levels.

## Simplified Betts-Miller convection

The simplified Betts-Miller scheme (Frierson, 2007) parameterizes deep and shallow moist
convection by relaxing the atmospheric column toward a moist-adiabatic reference profile
over a specified timescale.

### Algorithm

1. **Construct the reference profile.** Starting from the lowest-level temperature and
   humidity, integrate the moist pseudoadiabat upward to obtain a reference temperature
   profile $T_\text{ref}(k)$. The reference humidity is

   $$
   q_\text{ref}(k) = \text{rh}_\text{ref} \, q_\text{sat}(T_\text{ref}(k), p_k)
   $$

   where $\text{rh}_\text{ref} = 0.7$ (70% relative humidity) by default.

2. **Find the level of zero buoyancy (LZB).** The LZB is the highest level where the
   parcel virtual temperature exceeds the environment virtual temperature:

   $$
   T_{v,\text{parcel}} = T_\text{ref}(1 + (1/\varepsilon - 1) q_\text{ref})
   > T_{v,\text{env}} = T(1 + (1/\varepsilon - 1) q)
   $$

   Convection requires at least two buoyant levels between the surface and the LZB.

3. **Classify as deep or shallow.** Two column integrals determine the regime:

   $$
   P_q = \int_\text{sfc}^\text{LZB} (q - q_\text{ref}) \, d\sigma \qquad
   \text{(moisture excess)}
   $$

   $$
   P_t = -\int_\text{sfc}^\text{LZB} (T - T_\text{ref}) \, d\sigma \qquad
   \text{(heat excess)}
   $$

   - **Deep convection** ($P_q > 0$ and $P_t > 0$): the column has excess moisture and
     excess convective available potential energy. The reference temperature is shifted by
     an enthalpy-conserving offset:

     $$
     \Delta T = \frac{P_t - P_q L_v / c_p}{\Delta\sigma_\text{LZB}}
     $$

   - **Shallow convection** ($P_q \le 0$ and $P_t > 0$): the column has excess heat but
     insufficient moisture for deep convection. The reference humidity is rescaled to
     preserve column moisture:

     $$
     f_q = 1 - \frac{P_q}{\sum q_\text{ref} \, \Delta\sigma}, \qquad
     q_\text{ref} \leftarrow q_\text{ref} \cdot f_q
     $$

     and the temperature offset uses only the heat excess:

     $$
     \Delta T = \frac{P_t}{\Delta\sigma_\text{LZB}}
     $$

4. **Relax toward the reference.** The convective tendencies are

   $$
   \frac{\partial T}{\partial t}\bigg|_\text{conv}
   = -\frac{T - (T_\text{ref} + \Delta T)}{\tau_\text{BM}}
   $$

   $$
   \frac{\partial q}{\partial t}\bigg|_\text{conv}
   = -\frac{q - q_\text{ref}}{\tau_\text{BM}}
   $$

   where $\tau_\text{BM} = 7200$ s (2 hours) is the relaxation timescale.

### Default parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\tau_\text{BM}$ | 7200 s | Relaxation timescale |
| $\text{rh}_\text{ref}$ | 0.7 | Reference relative humidity |
| Min buoyant levels | 2 | Activation threshold |

## Interaction between schemes

When all moisture physics is active, the schemes are applied in sequence each time step:

1. **Betts-Miller convection** (deep or shallow)
2. **Large-scale condensation** (removes any remaining supersaturation)
3. **Dry convective adjustment** (removes gravitational instability)

Precipitation from both condensation and convection is accumulated into the total surface
precipitation diagnostic.

## References

- Bolton, D. (1980). The computation of equivalent potential temperature. *Mon. Wea. Rev.*,
  108, 1046-1053.
- Manabe, S. and Strickler, R. F. (1964). Thermal equilibrium of the atmosphere with a
  convective adjustment. *J. Atmos. Sci.*, 21, 361-385.
- Frierson, D. M. W., Held, I. M., and Zurita-Gotor, P. (2006). A gray-radiation
  aquaplanet moist GCM. Part I: Static stability and eddy scale. *J. Atmos. Sci.*, 63,
  2548-2566.
- Frierson, D. M. W. (2007). The dynamics of idealized convection schemes and their effect
  on the zonally averaged tropical circulation. *J. Atmos. Sci.*, 64, 1959-1976.
