# Spectral Transform Method

Notus uses the spectral transform method to solve the primitive equations on the sphere.
Prognostic variables are stored as coefficients of a spherical harmonic expansion (spectral
space), while nonlinear products are evaluated on a Gaussian grid (grid-point space).
Transforms between the two representations are performed every time step.

This approach exploits the fact that horizontal derivatives and the Laplacian are diagonal
in spectral space, making them exact and efficient, while nonlinear terms are most naturally
computed as point-wise products on the grid.

## Spherical harmonics

A scalar field on the sphere is expanded as

$$
f(\lambda, \varphi) = \sum_{m=0}^{T} \sum_{n=m}^{T} \hat{f}_n^m \,
\bar{P}_n^m(\sin\varphi) \, e^{im\lambda}
$$

where $\lambda$ is longitude, $\varphi$ is latitude, $m$ is the zonal wavenumber, $n$ is
the total wavenumber, and $T$ is the truncation wavenumber. The expansion uses **triangular
truncation**: for each $m$, the total wavenumber $n$ ranges from $m$ to $T$.

The $\bar{P}_n^m$ are fully normalized associated Legendre polynomials, satisfying the
orthonormality condition

$$
\int_{-1}^{1} \bar{P}_n^m(\mu) \, \bar{P}_{n'}^m(\mu) \, d\mu
= \frac{2}{2n+1} \, \delta_{nn'}
$$

where $\mu = \sin\varphi$. With this normalization the spherical harmonics
$Y_n^m = \bar{P}_n^m(\sin\varphi) \, e^{im\lambda}$ are orthonormal over the unit sphere:

$$
\int_0^{2\pi} \int_{-\pi/2}^{\pi/2}
Y_n^m \, {Y_{n'}^{m'}}^* \, \cos\varphi \, d\varphi \, d\lambda = \delta_{nn'}\delta_{mm'}
$$

### Legendre polynomial computation

The normalized associated Legendre polynomials are computed via three-term recurrence.
Starting from the sectoral value ($n = m$), computed iteratively to avoid overflow:

$$
\bar{P}_m^m(\mu) = (-1)^m \prod_{i=1}^{m}
\sqrt{\frac{2i+1}{2i}} \; (1 - \mu^2)^{m/2}
$$

The first tesseral value is

$$
\bar{P}_{m+1}^m(\mu) = a_{m+1}^m \, \mu \, \bar{P}_m^m(\mu)
$$

and subsequent values follow from

$$
\bar{P}_n^m(\mu) = a_n^m \, \mu \, \bar{P}_{n-1}^m(\mu)
- b_n^m \, \bar{P}_{n-2}^m(\mu)
$$

with recurrence coefficients

$$
a_n^m = \sqrt{\frac{(2n-1)(2n+1)}{(n-m)(n+m)}}, \qquad
b_n^m = \sqrt{\frac{(2n+1)(n+m-1)(n-m-1)}{(n-m)(n+m)(2n-3)}}
$$

### Spectral indexing

Spectral coefficients are stored in a one-dimensional array using lower-triangular
column-major ordering. The flat index for wavenumber pair $(m, n)$ is

$$
\text{index}(m, n) = m(T+1) - \frac{m(m-1)}{2} + (n - m)
$$

The total number of complex spectral coefficients is $(T+1)(T+2)/2$.

## Gaussian grid

The physical-space grid is a **Gaussian grid**: latitudes are placed at the roots of the
Legendre polynomial $P_{N_\text{lat}}(\sin\varphi)$, and longitudes are equally spaced.

### Grid dimensions

For a truncation $T$, the default grid uses the **quadratic** (alias-free) rule:

$$
N_\text{lat} = \left\lceil \frac{3T + 1}{2} \right\rceil
\quad \text{(rounded up to the next even number)}
$$

$$
N_\text{lon} = 2 \, N_\text{lat}
$$

The quadratic grid ensures exact Gaussian quadrature for products of three spherical
harmonics of degree up to $T$. Since the nonlinear advection terms involve products of two
fields (each truncated at degree $T$), the result has degree up to $2T$; integrating this
against a test function of degree $T$ requires exactness up to degree $3T$. Gaussian
quadrature with $N_\text{lat}$ points is exact for polynomials of degree $2N_\text{lat} -
1$, so the requirement $2N_\text{lat} - 1 \ge 3T$ gives the formula above.

### Gaussian quadrature

The latitudes $\varphi_j$ and quadrature weights $w_j$ are obtained from the roots and
weights of $P_{N_\text{lat}}(\mu)$ via standard Gauss-Legendre quadrature. Latitudes are
ordered north to south. The weights satisfy

$$
\sum_{j=1}^{N_\text{lat}} w_j \, P_n(\mu_j) \, P_{n'}(\mu_j)
= \frac{2}{2n+1} \, \delta_{nn'}
\qquad \text{for } n + n' \le 2N_\text{lat} - 1
$$

Longitudes are uniformly spaced:

$$
\lambda_k = \frac{2\pi k}{N_\text{lon}}, \qquad k = 0, 1, \ldots, N_\text{lon} - 1
$$

## Forward transform (grid to spectral)

The transform from grid-point values to spectral coefficients is factored into two steps.

**Step 1: FFT along longitude.** For each latitude $\varphi_j$, compute the Fourier
coefficients

$$
\tilde{f}_m(\varphi_j) = \sum_{k=0}^{N_\text{lon}-1}
f(\lambda_k, \varphi_j) \, e^{-im\lambda_k}
$$

via a real-to-complex FFT, retaining wavenumbers $m = 0, 1, \ldots, T$.

**Step 2: Legendre transform with Gaussian quadrature.** For each zonal wavenumber $m$,
project onto the associated Legendre polynomials:

$$
\hat{f}_n^m = \frac{1}{2N_\text{lon}} \sum_{j=1}^{N_\text{lat}}
w_j \, \tilde{f}_m(\varphi_j) \, \bar{P}_n^m(\sin\varphi_j)
$$

The prefactor $1/(2N_\text{lon})$ combines the longitude trapezoidal rule ($2\pi /
N_\text{lon}$) with the spherical harmonic normalization ($1/4\pi$).

## Inverse transform (spectral to grid)

The inverse transform reverses the two steps.

**Step 1: Inverse Legendre transform.** For each zonal wavenumber $m$ and latitude
$\varphi_j$, sum over total wavenumber:

$$
\tilde{f}_m(\varphi_j) = N_\text{lon} \sum_{n=m}^{T}
\hat{f}_n^m \, \bar{P}_n^m(\sin\varphi_j)
$$

The factor $N_\text{lon}$ pre-multiplies the Fourier coefficients to compensate for the
inverse FFT normalization.

**Step 2: Inverse FFT along longitude.** Reconstruct the grid-point field via a
complex-to-real inverse FFT.

## Spectral operators

Because spherical harmonics are eigenfunctions of the Laplacian on the sphere, many
differential operators reduce to algebraic operations in spectral space.

### Laplacian

$$
\widehat{\nabla^2 f}_n^m = -\frac{n(n+1)}{a^2} \, \hat{f}_n^m
$$

where $a$ is the planetary radius. The Laplacian eigenvalue for wavenumber $n$ is
$\lambda_n = -n(n+1)/a^2$.

### Inverse Laplacian

$$
\widehat{\nabla^{-2} f}_n^m = \frac{1}{\lambda_n} \, \hat{f}_n^m
\qquad (n \ge 1)
$$

The global mean ($n = 0$) is set to zero, since the inverse Laplacian is defined only up to
a constant.

### Zonal derivative

$$
\widehat{\frac{\partial f}{\partial \lambda}}\bigg|_n^m = im \, \hat{f}_n^m
$$

### Meridional derivative

The meridional derivative is not diagonal in spectral space; instead it couples adjacent
total wavenumbers via the recurrence

$$
\widehat{\cos\varphi \frac{\partial f}{\partial \varphi}}\bigg|_n^m
= (n+2) \, \varepsilon_{n+1}^m \, \hat{f}_{n+1}^m
- (n-1) \, \varepsilon_n^m \, \hat{f}_{n-1}^m
$$

with coupling coefficient

$$
\varepsilon_n^m = \sqrt{\frac{n^2 - m^2}{4n^2 - 1}}
$$

### Wind reconstruction from vorticity and divergence

Winds are recovered from the streamfunction $\psi$ and velocity potential $\chi$, obtained
from the vorticity $\zeta$ and divergence $\delta$ via the inverse Laplacian:

$$
\hat{\psi}_n^m = \nabla^{-2} \hat{\zeta}_n^m, \qquad
\hat{\chi}_n^m = \nabla^{-2} \hat{\delta}_n^m
$$

The wind components (scaled by $\cos\varphi$) are then

$$
\widehat{u\cos\varphi} = \frac{1}{a}\left(
-\widehat{\cos\varphi\,\frac{\partial\psi}{\partial\varphi}}
+ \frac{\partial\hat{\chi}}{\partial\lambda}\right)
$$

$$
\widehat{v\cos\varphi} = \frac{1}{a}\left(
\frac{\partial\hat{\psi}}{\partial\lambda}
+ \widehat{\cos\varphi\,\frac{\partial\chi}{\partial\varphi}}\right)
$$

### Spectral divergence and curl

Given spectral flux components $\hat{A}$ and $\hat{B}$ (representing zonal and meridional
fluxes divided by $\cos^2\varphi$), the spectral divergence and curl of the flux vector are

$$
\widehat{\nabla \cdot \mathbf{F}}\bigg|_n^m
= \frac{1}{a}\left(im\,\hat{A}_n^m + \frac{d\hat{B}_n^m}{d\mu}\right)
$$

$$
\widehat{(\nabla \times \mathbf{F})_z}\bigg|_n^m
= \frac{1}{a}\left(-im\,\hat{B}_n^m + \frac{d\hat{A}_n^m}{d\mu}\right)
$$

where $\mu = \sin\varphi$ and $d/d\mu$ is computed via a three-term recurrence analogous to
the meridional derivative.

### Hyperdiffusion

Scale-selective diffusion of order $2p$ acts on spectral coefficients as

$$
\frac{\partial \hat{f}_n^m}{\partial t}\bigg|_\text{diff}
= (-1)^{p+1} \, \nu_{2p} \, \nabla^{2p} \hat{f}_n^m
= -\nu_{2p} \left(\frac{n(n+1)}{a^2}\right)^p \hat{f}_n^m
$$

The coefficient $\nu_{2p}$ is chosen so that the smallest retained scale ($n = T$) is
damped with an $e$-folding time $\tau$:

$$
\nu_{2p} = \frac{1}{\tau} \left(\frac{a^2}{T(T+1)}\right)^p
$$

The default configuration uses $\nabla^8$ diffusion ($p = 4$) with a 2-hour damping
timescale.

### Exponential spectral filter

An additional multiplicative filter (Hou and Li, 2007) may be applied once per time step to
suppress noise at the smallest scales:

$$
\hat{f}_n^m \leftarrow \hat{f}_n^m \,
\exp\left(-\alpha \left(\frac{n}{T}\right)^{2s}\right)
$$

where $s$ is the filter order (default 18) and $\alpha = \Delta t \cdot 2\Omega / \tau_f$
with $\tau_f$ a non-dimensional timescale (default 0.010938). The high polynomial order
ensures that only the highest wavenumbers are affected.

## References

- Bourke, W. (1972). An efficient, one-level, primitive-equation spectral model. *Mon. Wea.
  Rev.*, 100, 683-689.
- Hack, J. J. and Jakob, R. (1992). Description of a global shallow water model based on
  the spectral transform method. *NCAR Technical Note*, NCAR/TN-343+STR.
- Hou, T. Y. and Li, R. (2007). Computing nearly singular solutions using pseudo-spectral
  methods. *J. Comput. Phys.*, 226, 379-397.
