"""Forward model stellar spectra."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional

from luminare.scalers import BaseScaler, NoScaler, PeriodicScaler
from luminare.flux_model import FluxModel, RectifiedFluxModel, ConvolvedFluxModel
from luminare.continuum_model import ContinuumModel, NoContinuumModel

class StellarSpectrumModel(eqx.Module):

    """Model for stellar spectra."""
    
    λ: jax.Array
    flux_model: FluxModel
    continuum_model: ContinuumModel
    n_parameters: int
    scaler: BaseScaler
    stellar_label_names: Tuple[str, ...] = None
    
    def __init__(
        self,
        λ: jax.Array,
        flux_model: FluxModel,
        continuum_model: ContinuumModel,
        stellar_label_names: Optional[Tuple[str, ...]] = None,
        scaler: BaseScaler = NoScaler(),
    ):
        # TODO: have λ here? Or create FluxModel and ContinuumModel here?
        # λ exists implicitly in flux_model and continuum_model, but is not stored
        self.λ = λ
        self.scaler = scaler
        self.flux_model = flux_model
        self.continuum_model = continuum_model
        self.stellar_label_names = stellar_label_names
        self.n_parameters = flux_model.n_parameters + continuum_model.n_parameters

    def __call__(self, θ: jnp.array) -> jnp.ndarray:
        """Predict the stellar spectrum given parameters."""
        eqx.partition((self.flux_model, self.continuum_model), eqx.is_array)

        n = self.flux_model.n_parameters
        rectified_flux = self.flux_model(θ[:n])
        continuum = self.continuum_model(θ[n:])
        return rectified_flux * continuum + jax.lax.stop_gradient(self.λ)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        ndim = self.scaler.ndim
        return jnp.hstack([self.scaler.transform(x[:ndim]), x[ndim:]])

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        ndim = self.scaler.ndim
        return jnp.hstack([self.scaler.inverse_transform(x[:ndim]), x[ndim:]])


def create_stellar_spectrum_model(
    λ: jnp.array,
    H: jnp.ndarray,
    X: jnp.ndarray,
    n_modes: Tuple[int, ...],
    stellar_label_names: Tuple[str, ...] = None,
    min_stellar_labels: Optional[jnp.array] = None,
    max_stellar_labels: Optional[jnp.array] = None,
    n_stellar_label_points: Optional[jnp.array] = None,
    spectral_resolution: Optional[float] = None,
    max_vsini: Optional[float] = None,
    continuum_regions: Optional[Tuple[float, float]] = None,
    continuum_n_modes: Optional[int] = None,
    **kwargs
):  
    λ = jnp.array(λ)

    if (
        min_stellar_labels is None 
    or  max_stellar_labels is None 
    or  n_stellar_label_points is None
    ):
        scaler = NoScaler()
    else:
        scaler = PeriodicScaler(
            n=jnp.array(n_stellar_label_points),
            minimum=jnp.array(min_stellar_labels),
            maximum=jnp.array(max_stellar_labels),
        )

    if continuum_regions is None and continuum_n_modes is None:
        continuum_model = NoContinuumModel()
    else:
        continuum_model = ContinuumModel(
            λ=λ,
            continuum_regions=continuum_regions,
            continuum_n_modes=continuum_n_modes,
        )
    
    flux_model = RectifiedFluxModel(
        H=jnp.array(H), 
        X=jnp.array(X), 
        n_modes=n_modes, 
        n_parameters=len(n_modes)
    )

    if spectral_resolution is not None and max_vsini is not None:
        if stellar_label_names is not None:
            stellar_label_names = (*stellar_label_names, "vsini")

        flux_model = ConvolvedFluxModel(
            rectified_flux_model=flux_model,
            spectral_resolution=spectral_resolution,
            max_vsini=max_vsini,
        )
    
    return StellarSpectrumModel(
        λ=λ,
        stellar_label_names=stellar_label_names,
        flux_model=flux_model,
        continuum_model=continuum_model,
        scaler=scaler
    )

