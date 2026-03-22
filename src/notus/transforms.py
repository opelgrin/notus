"""Spherical harmonic transforms between grid-point and spectral space.

The forward transform (grid -> spectral) decomposes a scalar field f(lambda, phi)
into spherical harmonic coefficients:

    f(lambda, phi) = sum_m sum_n  f_hat_n^m * P_n^m(sin phi) * exp(i*m*lambda)

The transform is factored into two steps:

1. **FFT in longitude** -- for each latitude row, compute:
       f_m(phi_j) = sum_k f(lambda_k, phi_j) * exp(-i*m*lambda_k)      (via FFT)

2. **Legendre transform in latitude** -- for each m, compute:
       f_hat_n^m = sum_j w_j * f_m(phi_j) * P_n^m(sin phi_j)   (Gaussian quadrature)

The inverse transform reverses these steps.

Implementation note: to avoid dynamic shapes inside JAX's fori_loop, the
Legendre polynomials are stored in a 3D array of shape (T+1, n_lat, T+1)
indexed as [m, :, n-m], zero-padded for n > T.  This allows fixed-shape
slicing inside JIT-compiled loops.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from notus.grid import GaussianGrid
from notus.spherical_harmonics import compute_legendre_polynomials


class SpectralTransform:
    """Pre-computed transform state for a given grid.

    Caches the Legendre polynomials reorganized for efficient JIT-compiled
    transforms.  All heavy computation happens at construction time.

    Parameters
    ----------
    grid : GaussianGrid
        The Gaussian grid defining resolution and truncation.
    """

    def __init__(self, grid: GaussianGrid) -> None:
        self.grid = grid
        t = grid.truncation

        # Flat Legendre: (n_lat, n_spectral) — used for operators/diagnostics
        self.legendre_flat = compute_legendre_polynomials(t, grid.sin_lat)

        # Reorganize into 3D array for JIT-friendly transforms.
        # legendre_3d[m, j, k] = P_n^m(sin phi_j) where k = n - m.
        # Shape: (T+1, n_lat, T+1), zero-padded when k > T - m.
        n_lat = grid.n_lat
        max_len = t + 1  # max number of n-values for any m (occurs at m=0)

        leg3d = jnp.zeros((max_len, n_lat, max_len))
        wleg3d = jnp.zeros((max_len, n_lat, max_len))

        for m in range(t + 1):
            for k in range(t - m + 1):
                n = m + k
                flat_idx = grid.spectral_index(m, n)
                leg3d = leg3d.at[m, :, k].set(self.legendre_flat[:, flat_idx])
                wleg3d = wleg3d.at[m, :, k].set(
                    self.legendre_flat[:, flat_idx] * grid.lat_weights
                )

        self._legendre_3d = leg3d          # (T+1, n_lat, T+1)
        self._weighted_legendre_3d = wleg3d  # (T+1, n_lat, T+1)

        # Mask: mask_3d[m, k] = 1.0 if k <= T - m, else 0.0
        mask = jnp.zeros((max_len, max_len))
        for m in range(t + 1):
            for k in range(t - m + 1):
                mask = mask.at[m, k].set(1.0)
        self._mask = mask  # (T+1, T+1)

    def grid_to_spectral(self, field: jnp.ndarray) -> jnp.ndarray:
        """Forward transform: grid-point field -> spectral coefficients.

        Parameters
        ----------
        field : jnp.ndarray
            Real-valued grid field, shape ``(n_lat, n_lon)``.

        Returns
        -------
        jnp.ndarray
            Complex spectral coefficients, shape ``(n_spectral,)``.
        """
        return _grid_to_spectral(
            field,
            self._weighted_legendre_3d,
            self._mask,
            self.grid.truncation,
            self.grid.n_lon,
        )

    def spectral_to_grid(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Inverse transform: spectral coefficients -> grid-point field.

        Parameters
        ----------
        coeffs : jnp.ndarray
            Complex spectral coefficients, shape ``(n_spectral,)``.

        Returns
        -------
        jnp.ndarray
            Real-valued grid field, shape ``(n_lat, n_lon)``.
        """
        return _spectral_to_grid(
            coeffs,
            self._legendre_3d,
            self.grid.truncation,
            self.grid.n_lon,
        )


@functools.partial(jax.jit, static_argnums=(3, 4))
def _grid_to_spectral(
    field: jnp.ndarray,
    weighted_legendre_3d: jnp.ndarray,
    mask: jnp.ndarray,
    truncation: int,
    n_lon: int,
) -> jnp.ndarray:
    """JIT-compiled forward transform.

    Steps:
    1. FFT along longitude -> Fourier coefficients for m = 0..T
    2. For each m, Gaussian quadrature with weighted Legendre polynomials.
    """
    max_len = truncation + 1

    # Step 1: FFT
    fourier = jnp.fft.rfft(field, axis=1)[:, :max_len]  # (n_lat, T+1)

    # Normalization: f_hat = 1/(2*N_lon) * sum_j w_j * P_n^m * FFT[f]
    # The 1/(2*N_lon) combines the longitude trapezoidal rule (2*pi/N_lon)
    # with the 1/(4*pi) spherical harmonic normalization.
    fourier /= 2.0 * n_lon

    # Legendre transform — vectorized over all m at once via einsum.
    # result[m, k] = sum_j fourier[j, m] * weighted_legendre[m, j, k]
    result_3d = jnp.einsum("jm,mjk->mk", fourier, weighted_legendre_3d)

    # Zero out invalid entries (k > T - m)
    result_3d *= mask

    # Pack back to flat spectral array.
    return _pack_spectral(result_3d, truncation)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _spectral_to_grid(
    coeffs: jnp.ndarray,
    legendre_3d: jnp.ndarray,
    truncation: int,
    n_lon: int,
) -> jnp.ndarray:
    """JIT-compiled inverse transform.

    Steps:
    1. Inverse Legendre: for each m, sum spectral coefficients * P_n^m.
    2. Inverse FFT along longitude.
    """
    max_len = truncation + 1

    # Unpack flat spectral array to 3D: (T+1, T+1) indexed [m, k=n-m]
    coeffs_3d = _unpack_spectral(coeffs, truncation)  # (T+1, T+1)

    # Inverse Legendre: fourier[j, m] = sum_k coeffs[m, k] * P[m, j, k]
    fourier = jnp.einsum("mk,mjk->jm", coeffs_3d, legendre_3d)  # (n_lat, T+1)

    # Inverse FFT: pad Fourier coefficients to n_lon//2 + 1 and apply irfft.
    n_rfft = n_lon // 2 + 1
    fourier_full = jnp.zeros((fourier.shape[0], n_rfft), dtype=jnp.complex128)
    fourier_full = fourier_full.at[:, :max_len].set(fourier)

    # The forward transform divided by (2*N_lon), so inverse multiplies by (2*N_lon).
    # jnp.fft.irfft computes: (1/N) * sum c_m * exp(i*m*lambda) * N = sum c_m * exp(...)
    # Wait — jnp.fft.irfft expects unnormalized FFT output.
    # rfft(x)[m] = sum_k x[k] * exp(-2*pi*i*m*k/N)  (unnormalized)
    # irfft(X)[k] = (1/N) * sum_m X[m] * exp(2*pi*i*m*k/N)
    # Our forward gave: coeff = FFT / (2*N) * (Legendre stuff)
    # So spectral -> fourier gives: fourier[j,m] = sum_n coeffs_n^m * P_n^m
    # We want: f(lambda_k, phi_j) = sum_m fourier[j,m] * exp(i*m*lambda_k)
    # Using irfft: f = irfft(X, N) = (1/N) * sum X[m] * exp(i*m*...)
    # ... no, irfft actually does: sum X[m] * exp(2*pi*i*m*k/N) / N
    # So we need X[m] = fourier[j,m] * N ... wait.
    # Let's just think about it end-to-end:
    # Forward: fft gives sum f*exp(-i*m*lam) with lam = 2*pi*k/N.
    # We then divide by (2*N_lon).
    # Inverse: we reconstruct fourier[j,m] from spectral coefficients.
    # Then irfft(X, N) = (1/N) * sum_{m=0}^{N/2} X[m] * exp(i*2*pi*m*k/N) * weight
    # where weight accounts for conjugate symmetry.
    # To recover the original field: irfft(rfft(field)) = field * N ... no.
    # Actually jnp.fft.irfft(jnp.fft.rfft(x)) = x (they are mutual inverses).
    # So rfft(x)[m] / (2*N) was stored. To invert: multiply by (2*N) and irfft.
    # The synthesis is: f(λ_k, φ_j) = Σ_m g_m(φ_j) · e^{imλ_k}
    # irfft computes: x[k] = (1/N) Σ_m X[m] · e^{2πi·m·k/N}
    # So we need to multiply by N to cancel the 1/N in irfft.
    # (The forward divided by 2N; the factor of 2 accounts for the
    # longitude integral normalization 2π/(4π) = 1/2, which is already
    # absorbed into the spectral coefficients and doesn't need undoing.)
    return jnp.fft.irfft(fourier_full * n_lon, n=n_lon, axis=1)


def _pack_spectral(arr_3d: jnp.ndarray, truncation: int) -> jnp.ndarray:
    """Pack (T+1, T+1) array [m, k=n-m] into flat spectral (n_spectral,).

    We use a padded flat buffer of size (T+1)^2 to ensure dynamic_update_slice
    never goes out of bounds (JAX silently shifts start indices otherwise).
    """
    max_len = truncation + 1
    n_spectral = max_len * (max_len + 1) // 2
    # Padded buffer: large enough that writing max_len at any valid start is safe.
    padded_size = n_spectral + max_len
    flat = jnp.zeros(padded_size, dtype=arr_3d.dtype)

    def body(m: int, flat: jnp.ndarray) -> jnp.ndarray:
        start = m * max_len - m * (m - 1) // 2
        row = arr_3d[m, :]  # (T+1,) — zero-padded beyond valid k
        return jax.lax.dynamic_update_slice(flat, row, (start,))

    return jax.lax.fori_loop(0, max_len, body, flat)[:n_spectral]


def _unpack_spectral(flat: jnp.ndarray, truncation: int) -> jnp.ndarray:
    """Unpack flat spectral (n_spectral,) into (T+1, T+1) array [m, k=n-m].

    Pads the input so dynamic_slice never goes out of bounds.
    """
    max_len = truncation + 1
    # Pad the flat array so slicing max_len from any valid start is safe.
    padded = jnp.pad(flat, (0, max_len), mode="constant")
    arr = jnp.zeros((max_len, max_len), dtype=flat.dtype)

    def body(m: int, arr: jnp.ndarray) -> jnp.ndarray:
        start = m * max_len - m * (m - 1) // 2
        row = jax.lax.dynamic_slice(padded, (start,), (max_len,))
        return arr.at[m, :].set(row)

    return jax.lax.fori_loop(0, max_len, body, arr)
