"""Forward model continuum-normalized spectra."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
# TODO: exojax has a lot of dependencies (mat plotlib, plotly, pandas, etc.)
#       Consider using only the parts we need.
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid as _velocity_grid

from luminare.fourier import eval_at_point

class FluxModel(eqx.Module):
    pass

class RectifiedFluxModel(FluxModel):
    """
    Rectified flux model for continuum-normalized spectra.
    """

    H: jax.Array
    X: jax.Array
    n_modes: Tuple[int, ...]
    n_parameters: int

    def __init__(
        self,
        H: jax.Array,
        X: jax.Array,
        n_modes: Tuple[int, ...],
        n_parameters: int
    ):
        self.H = H
        self.X = X
        self.n_modes = n_modes
        self.n_parameters = n_parameters

    def basis_weights(self, ϴ: jnp.ndarray, epsilon: float) -> jnp.ndarray:
        W = jax.vmap(eval_at_point, in_axes=(None, None, 1))(ϴ, self.n_modes, self.X)
        return jnp.clip(W, epsilon, None)

    def __call__(self, θ: jnp.ndarray, epsilon: float = 0.0) -> jnp.ndarray:
        W = self.basis_weights(θ, epsilon)
        return 1 - W @ self.H


class ConvolvedFluxModel(eqx.Module):

    """Convolved flux model for continuum-normalized spectra."""

    rectified_flux_model: RectifiedFluxModel
    velocity_grid: jnp.ndarray
    max_vsini: float
    n_parameters: int

    def __init__(
        self,
        rectified_flux_model: RectifiedFluxModel,
        max_vsini: float,
    ):
        self.rectified_flux_model = rectified_flux_model
        self.velocity_grid = _velocity_grid(spectral_resolution, max_vsini)
        self.max_vsini = max_vsini
        self.n_parameters = rectified_flux_model.n_parameters + 1
    
    def __call__(self, θ: jnp.ndarray, epsilon: float = 0.0) -> jnp.ndarray:
        rectified_flux = self.rectified_flux_model(θ[:-1], epsilon=epsilon)
        vsini = jnp.clip(θ[-1], 0, self.max_vsini)
        return jax.lax.cond(
            vsini > 0,
            lambda: self.convolve_flux(rectified_flux, vsini),
            lambda: rectified_flux
        ) + jax.lax.stop_gradient(self.velocity_grid)

    def convolve_flux(self, rectified_flux, vsini):
        return (
            convolve_rigid_rotation(
                rectified_flux.flatten(), 
                self.velocity_grid, 
                vsini
            )
            .reshape((1, -1))
        )

