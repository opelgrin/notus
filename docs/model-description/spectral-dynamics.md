# Spectral Dynamics

This section describes how the explicit tendency terms of the primitive equations are
computed in spectral space, including the transform method for nonlinear products and the
semi-implicit splitting that separates fast gravity wave terms from the rest.

## Transform method for nonlinear terms

The right-hand sides of the vorticity, divergence, and thermodynamic equations contain
nonlinear products (e.g., $(\zeta + f) \cdot u$, $T \cdot u$). These cannot be computed
directly in spectral space without convolution sums that are prohibitively expensive. The
**transform method** evaluates them on the Gaussian grid and transforms the result back to
spectral space:

1. Reconstruct $u\cos\varphi$ and $v\cos\varphi$ from $\hat{\zeta}$ and $\hat{\delta}$
   using the spectral wind reconstruction (see
   [Spectral Transform Method](spectral-transform.md)).
2. Transform all needed fields ($u\cos\varphi$, $v\cos\varphi$, $\zeta$, $T$, $q$,
   $\partial\ln p_s / \partial\lambda$, $\cos\varphi \, \partial\ln p_s / \partial\varphi$)
   to grid space.
3. Form the nonlinear products on the grid (see below).
4. Transform the products back to spectral space.
5. Apply spectral operators (curl, divergence, Laplacian) to obtain the tendencies.

The alias-free (quadratic) Gaussian grid ensures that the transform of the quadratic
products is exact up to the truncation wavenumber.

## Grid-point products

### Absolute vorticity fluxes

The vorticity tendency requires the momentum flux:

$$
A_\zeta = \frac{(\zeta + f) \, u\cos\varphi}{\cos^2\varphi}, \qquad
B_\zeta = \frac{(\zeta + f) \, v\cos\varphi}{\cos^2\varphi}
$$

The $\cos^{-2}\varphi$ weighting is required by the spectral divergence and curl operators,
which expect fluxes in this form. The vorticity and divergence tendencies from these fluxes
are

$$
\frac{\partial\hat{\zeta}}{\partial t}\bigg|_\text{adv} = \text{curl}_z(\hat{A}_\zeta, \hat{B}_\zeta)
$$

$$
\frac{\partial\hat{\delta}}{\partial t}\bigg|_\text{adv} = -\text{div}(\hat{A}_\zeta, \hat{B}_\zeta)
$$

### Kinetic energy

The kinetic energy contribution to the divergence tendency is

$$
E = \frac{u^2 + v^2}{2\cos^2\varphi}
$$

contributing $-\nabla^2 \hat{E}$ to the divergence tendency.

### Temperature fluxes

The temperature advection uses flux form in spectral space:

$$
A_T = \frac{T' \, u\cos\varphi}{\cos^2\varphi}, \qquad
B_T = \frac{T' \, v\cos\varphi}{\cos^2\varphi}
$$

where $T' = T - T_\text{ref}$ is the temperature anomaly relative to the reference profile.
The spectral divergence of these fluxes gives $-\nabla \cdot (T' \mathbf{v})$, but the
semi-implicit scheme assumes the advective form $-\mathbf{v} \cdot \nabla T'$. Since these
differ by $T' \delta$, a grid-point correction is added:

$$
\frac{\partial T}{\partial t}\bigg|_\text{nodal} = \dot{\sigma}\frac{\partial T}{\partial\sigma}\bigg|_\text{explicit} + \kappa T \frac{\omega}{p} + T' \delta
$$

This nodal correction, transformed to spectral space and added to the flux divergence,
recovers the advective form needed for consistency with the semi-implicit treatment.

### Pressure gradient

The explicit part of the pressure gradient force is

$$
R T_v' \, \nabla\ln p_s
$$

where $T_v'$ is the virtual temperature anomaly relative to the reference virtual
temperature. When moisture is active, $T_v' = T(1 + \varepsilon_v q) - T_{v,\text{ref}}$;
otherwise $T_v' = T - T_\text{ref}$. This term is added to the momentum flux components
before computing the vorticity and divergence tendencies.

### Vertical advection of momentum

The vertical advection of momentum is computed on the grid:

$$
-\dot{\sigma}\frac{\partial(u\cos\varphi)}{\partial\sigma}, \qquad
-\dot{\sigma}\frac{\partial(v\cos\varphi)}{\partial\sigma}
$$

These are added to the momentum flux, divided by $\cos^2\varphi$, before the spectral curl
and divergence operations.

## Semi-implicit splitting

The semi-implicit method separates the right-hand side into explicit nonlinear terms and
implicit linear terms that support gravity waves. The linearization is performed around a
reference state consisting of a horizontally uniform reference temperature $T_\text{ref}$
and a resting atmosphere.

### Reference temperature

The reference temperature $T_\text{ref}$ is taken as the global mean temperature (the
$(0,0)$ spectral mode at the start of the simulation). When moisture is active, a reference
humidity profile $q_\text{ref}$ defines the reference virtual temperature:

$$
T_{v,\text{ref}} = T_\text{ref}(1 + \varepsilon_v \, q_\text{ref})
$$

This reference virtual temperature is used in the implicit pressure gradient term.

### Implicit tendency terms

The linear terms treated implicitly are:

$$
L_\delta = -\nabla^2(\Phi' + R \, T_{v,\text{ref}} \, \ln p_s)
$$

$$
L_T = -\mathbf{H} \, \mathbf{\delta}
$$

$$
L_{\ln p_s} = -\mathbf{\Delta\sigma}^T \mathbf{\delta}
$$

where $\Phi' = \mathbf{G}_v \, \mathbf{T}'$ is the geopotential anomaly computed from
the temperature anomaly via the (possibly virtual-temperature-scaled) geopotential weight
matrix, and $\mathbf{H}$ is the temperature implicit weight matrix. The vorticity equation
has no implicit terms ($L_\zeta = 0$).

### Temperature implicit weight matrix

The matrix $\mathbf{H}$ encodes how divergence affects temperature through two mechanisms:

1. **Divergence-dependent adiabatic heating**: the part of $\kappa T_\text{ref} (\omega/p)$
   that depends linearly on $\delta$.
2. **Divergence-dependent vertical advection**: the part of
   $-\dot{\sigma} \, \partial T_\text{ref} / \partial\sigma$ that depends linearly on
   $\delta$ (through $\dot{\sigma}$).

The matrix entries follow Durran's formulation and involve the sigma ratios $\alpha_j$, the
step function, and vertical differences of $T_\text{ref}$. When $T_\text{ref}$ is vertically
uniform, the vertical advection contribution vanishes and $\mathbf{H}$ reduces to the
adiabatic heating term alone.

### Vertical coupling matrix

The implicit system couples the vertical levels through the matrix

$$
\mathbf{M} = \mathbf{G}_v \mathbf{H} + R \, T_{v,\text{ref}} \otimes \mathbf{\Delta\sigma}
$$

The Helmholtz problem to be solved at each time step (see
[Time Integration](time-integration.md)) takes the form

$$
\left(\mathbf{I} - s^2 \lambda_n \mathbf{M}\right) \hat{\mathbf{\delta}}_n^m = \text{rhs}
$$

where $s$ is the implicit step size and $\lambda_n = -n(n+1)/a^2$ is the Laplacian
eigenvalue. Because $\lambda_n$ depends only on total wavenumber $n$, each spectral mode
decouples horizontally --- the solve reduces to an independent $L \times L$ linear system
for each $n$.

### Eigendecomposition solve

Rather than solving the $L \times L$ system directly for each wavenumber,
$\mathbf{M}$ is eigendecomposed once:

$$
\mathbf{M} = \mathbf{P} \, \text{diag}(\mu_j) \, \mathbf{P}^{-1}
$$

The system then becomes diagonal:

$$
(1 - s^2 \lambda_n \mu_j) \, (\mathbf{P}^{-1} \hat{\mathbf{\delta}})_j = (\mathbf{P}^{-1} \, \text{rhs})_j
$$

This requires only $O(L^2)$ work per spectral mode (for the matrix-vector products with
$\mathbf{P}$ and $\mathbf{P}^{-1}$) rather than $O(L^3)$ for a dense solve.

## Orography

When topography is present, the surface geopotential $\Phi_s = g z_s$ contributes an
additional term to the divergence tendency:

$$
\frac{\partial\hat{\delta}}{\partial t}\bigg|_\text{orog} = -\nabla^2 \hat{\Phi}_s
$$

This term is pre-computed in spectral space and added to the explicit divergence tendency at
every time step.

## Hyperdiffusion

Scale-selective diffusion is applied to vorticity, divergence, and temperature to prevent
the buildup of enstrophy at the smallest resolved scales. The default is $\nabla^8$
diffusion ($p = 4$) with a 2-hour $e$-folding time at the truncation wavenumber. See
[Spectral Transform Method](spectral-transform.md) for the formula.

## References

- Hoskins, B. J. and Simmons, A. J. (1975). A multi-layer spectral model and the
  semi-implicit method. *Quart. J. Roy. Meteor. Soc.*, 101, 637-655.
- Simmons, A. J. and Burridge, D. M. (1981). An energy and angular-momentum conserving
  vertical finite-difference scheme and hybrid vertical coordinates. *Mon. Wea. Rev.*, 109,
  758-766.
- Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics*. 2nd ed. Springer.
