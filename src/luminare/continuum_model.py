"""Equinox module for continuum forward modeling."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional


class ContinuumModel(eqx.Module):
    """
    Module for modeling stellar continuum using Fourier basis functions.
    """

    n_parameters: int
    design_matrix: jax.Array

    def __init__(
        self,
        λ: jax.Array,
        continuum_regions: Tuple[Tuple[float, float], ...],
        continuum_n_modes: int,
    ):
        """Initialize the continuum model.

        Args:
            λ: Wavelength grid.
            continuum_regions: Tuple of (start, end) λ continuum_regions
            continuum_n_modes: Number of Fourier modes per region
        """
        self.n_parameters = len(continuum_regions) * continuum_n_modes
        self.design_matrix = create_design_matrix(λ, continuum_regions, continuum_n_modes)

    def __call__(self, θ) -> jnp.ndarray:
        """Forward model the continuum.

        Args:
            θ: Model parameters

        Returns:
            Continuum values at the given λs
        """
        return self.design_matrix @ θ


class NoContinuumModel(eqx.Module):
    
    n_parameters: int = 0

    def __call__(self, θ: jnp.ndarray) -> jnp.ndarray:
        return 1
    

def get_fourier_mode_indices(P):
    return jnp.arange(P) - P // 2

def fourier_modes(*n_modes):
    indices = list(map(get_fourier_mode_indices, n_modes))
    return -1j * jnp.array(jnp.meshgrid(*indices, indexing="ij")).T.astype(
        jnp.complex64
    )

def fourier_design_matrix(x, n_modes):
    x = jnp.atleast_2d(x)
    n_dim, n_data = x.shape
    n_modes = jnp.atleast_1d(n_modes)
    assert n_dim == n_modes.size

    A = jnp.exp(fourier_modes(*n_modes) @ x).T.reshape(n_data, -1)
    m = A.shape[1]

    # make a dense matrix by taking the matrix-matrix product with an identity matrix
    # that has been made hermitian
    cols = [0.5 * (A[:, i].real + A[:, m - i - 1].real) for i in range(m // 2 + 1)]
    for i in range(m // 2 + 1, m):
        cols.append((0.5j * A[:, i] - 0.5j * A[:, m - i - 1]).real)
    return jnp.vstack(cols).T

def create_design_matrix(
    λ: jnp.ndarray, 
    continuum_regions: Tuple[Tuple[float, float], ...],
    continuum_n_modes: int
) -> jnp.ndarray:
    """Create the design matrix for the given λ array.

    Args:
        λ: λ array
        continuum_regions: Tuple of (start, end) λ continuum_regions
        continuum_n_modes: Number of Fourier modes per region

    Returns:
        Design matrix A such that continuum = coefficients @ A.T
    """
    A = jnp.zeros((λ.size, len(continuum_regions) * continuum_n_modes))

    for i, (start_wave, end_wave) in enumerate(continuum_regions):
        # Find λ indices for this region
        start_idx = jnp.searchsorted(λ, start_wave, side="left")
        end_idx = jnp.searchsorted(λ, end_wave, side="right")

        # Normalize λ to [0, 1] within the region
        wave_region = λ[start_idx:end_idx]
        if wave_region.size > 0:
            x_norm = (wave_region - wave_region[0]) / (
                wave_region[-1] - wave_region[0] + 1e-10
            )

            # Create Fourier design matrix for this region
            start_coeff = i * continuum_n_modes
            end_coeff = (i + 1) * continuum_n_modes
            A = A.at[start_idx:end_idx, start_coeff:end_coeff].set(
                fourier_design_matrix(x_norm.reshape(1, -1), continuum_n_modes)
            )
    return A
