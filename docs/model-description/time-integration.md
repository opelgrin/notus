# Time Integration

Notus uses an implicit-explicit (IMEX) leapfrog scheme with a Robert-Asselin-Williams
filter. The explicit part handles advection, Coriolis, and nonlinear terms; the implicit
part handles the gravity wave terms that would otherwise impose a severe CFL restriction on
the time step.

## Leapfrog scheme

The leapfrog (centered-in-time) scheme advances the state from two time levels:

$$
\mathbf{x}^{n+1} = \mathbf{x}^{n-1} + 2\Delta t \, \mathbf{F}(\mathbf{x}^n)
$$

where $\mathbf{F}$ contains all right-hand-side tendencies. The scheme is second-order
accurate and non-dissipative, but it admits a computational mode (alternating-timestep
oscillation) that must be filtered.

## IMEX splitting

The tendency is split into explicit and implicit parts:

$$
\mathbf{F}(\mathbf{x}) = \mathbf{F}_\text{E}(\mathbf{x}) + \mathbf{L}(\mathbf{x})
$$

where $\mathbf{F}_\text{E}$ contains the nonlinear advection, Coriolis, and diabatic terms,
and $\mathbf{L}$ contains the linear gravity wave coupling (pressure gradient + divergence
in the thermodynamic and continuity equations).

### Initialization (forward Euler)

The first time step uses a backward-forward Euler step:

$$
\mathbf{x}^* = \mathbf{x}^0 + \Delta t \, \mathbf{F}_\text{E}(\mathbf{x}^0)
$$

$$
\mathbf{x}^1 = (\mathbf{I} - \Delta t \, \mathbf{L})^{-1} \, \mathbf{x}^*
$$

### Main step

Subsequent time steps use the IMEX leapfrog with implicit weighting parameter $\alpha$:

$$
\mathbf{x}^* = \mathbf{x}^{n-1} + 2\Delta t \left[\mathbf{F}_\text{E}(\mathbf{x}^n) + (1 - \alpha)\,\mathbf{L}(\mathbf{x}^{n-1})\right]
$$

$$
\mathbf{x}^{n+1} = (\mathbf{I} - 2\Delta t \, \alpha \, \mathbf{L})^{-1} \, \mathbf{x}^*
$$

The parameter $\alpha = 0.5$ gives centered (Crank-Nicolson) implicit weighting, which is
second-order accurate in time. The implicit inverse is the Helmholtz solve described in
[Spectral Dynamics](spectral-dynamics.md).

## Robert-Asselin-Williams filter

The computational mode of the leapfrog scheme is damped by the Robert-Asselin-Williams (RAW)
filter (Williams, 2009). After computing $\mathbf{x}^{n+1}$, the filter modifies both the
current and future states:

$$
D = \mathbf{x}^{n-1} - 2\mathbf{x}^n + \mathbf{x}^{n+1}
$$

$$
\mathbf{x}^n_\text{filtered} = \mathbf{x}^n + \frac{\nu\alpha_\text{RAW}}{2} D
$$

$$
\mathbf{x}^{n+1}_\text{filtered} = \mathbf{x}^{n+1} - \frac{\nu(1 - \alpha_\text{RAW})}{2} D
$$

where $\nu$ is the Robert coefficient (default 0.05) controlling the filter strength and
$\alpha_\text{RAW}$ is the Williams coefficient (default 0.53) that distributes the
correction between the current and future time levels. Setting $\alpha_\text{RAW} = 1$
recovers the original Robert-Asselin filter. The RAW modification improves accuracy by
partially compensating the damping applied to the physical mode.

## Semi-implicit solve for primitive equations

The implicit system couples divergence, temperature, and log surface pressure. As derived in
[Spectral Dynamics](spectral-dynamics.md), the problem reduces to solving for each spectral
mode $n$:

$$
\left(\mathbf{I} - s^2 \lambda_n \mathbf{M}\right) \hat{\mathbf{\delta}}^{n+1} = \hat{\mathbf{\delta}}^* - s \lambda_n \hat{\mathbf{\Phi}}^*
$$

where $s = 2\Delta t \, \alpha$ is the implicit step size, $\lambda_n = -n(n+1)/a^2$, and
the superscript $*$ denotes the intermediate state after the explicit step. Once
$\hat{\mathbf{\delta}}^{n+1}$ is known, the other variables follow by
back-substitution:

$$
\hat{\mathbf{T}}^{n+1} = \hat{\mathbf{T}}^* - s \, \mathbf{H} \, \hat{\mathbf{\delta}}^{n+1}
$$

$$
\widehat{\ln p_s}^{n+1} = \widehat{\ln p_s}^* - s \, \mathbf{\Delta\sigma}^T \hat{\mathbf{\delta}}^{n+1}
$$

The $L \times L$ system is solved efficiently via eigendecomposition of $\mathbf{M}$,
reducing the problem to a diagonal solve for each vertical eigenmode.

## Semi-implicit solve for shallow water

The shallow water equations use a simpler semi-implicit system coupling divergence and
geopotential. For each spectral mode, the Schur complement gives:

$$
\hat{\delta}^{n+1} = \frac{\hat{\delta}^* - s \lambda_n \hat{\Phi}^*}{1 - s^2 \Phi_0 \lambda_n}
$$

$$
\hat{\Phi}^{n+1} = \frac{-s \Phi_0 \hat{\delta}^* + \hat{\Phi}^*}{1 - s^2 \Phi_0 \lambda_n}
$$

where $\Phi_0 = gH_0$ is the mean geopotential. This is a scalar (diagonal) solve for each
wavenumber.

## Post-step operations

After the implicit solve and RAW filter, several post-processing steps are applied:

1. **Spectral filtering** (if enabled): the exponential filter described in
   [Spectral Transform Method](spectral-transform.md) is applied to all prognostic fields.
2. **Humidity clipping**: specific humidity is transformed to grid space, negative values
   are set to zero, and the result is transformed back to spectral space. This corrects
   Gibbs oscillations near sharp moisture gradients.
3. **Implicit physics** (if enabled): boundary-layer drag and surface flux terms may be
   applied implicitly for numerical stability (see
   [Boundary Layer and Surface Fluxes](boundary-layer.md)).

## Stability considerations

The semi-implicit treatment removes the gravity wave CFL constraint, which would otherwise
limit the time step to

$$
\Delta t < \frac{\Delta x}{c_g}
$$

where $c_g \approx 300$ m/s is the external gravity wave speed and $\Delta x$ is the grid
spacing. With semi-implicit treatment, the time step is limited by the slower advective CFL:

$$
\Delta t < \frac{\Delta x}{u_\text{max}}
$$

where $u_\text{max}$ is the maximum wind speed (typically 50--80 m/s for Earth). The
explicit physics parameterizations (convection, radiation) may impose additional stability
constraints.

## Default parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Robert coefficient | $\nu$ | 0.05 | Leapfrog filter strength |
| Williams coefficient | $\alpha_\text{RAW}$ | 0.53 | RAW filter distribution |
| Implicit weighting | $\alpha$ | 0.5 | Centered (Crank-Nicolson) |
| Diffusion order | $2p$ | 8 ($p=4$) | $\nabla^8$ hyperdiffusion |
| Diffusion timescale | $\tau$ | 7200 s | 2-hour $e$-folding at truncation |

## References

- Robert, A. J. (1966). The integration of a low order spectral form of the primitive
  meteorological equations. *J. Meteor. Soc. Japan*, 44, 237-245.
- Asselin, R. (1972). Frequency filter for time integrations. *Mon. Wea. Rev.*, 100,
  487-490.
- Williams, P. D. (2009). A proposed modification to the Robert-Asselin time filter.
  *Mon. Wea. Rev.*, 137, 2538-2546.
