import sys
import h5py as h5
import numpy as np
import optimistix
import pickle
import jax
import jax.numpy as jnp
from time import time
from functools import partial

sys.path.insert(0, "../src/")


from luminare.emulator import create_stellar_spectrum_model
from luminare.continuum import create_design_matrix, initial_theta



with open("hot_stars_apogee.model", "rb") as fp:
    serialised_model = pickle.load(fp)

λ = 10**(4.179 + 6e-6 * jnp.arange(8575))
λ_model = jnp.linspace(15000, 17000, 100001)
continuum_n_modes = 7
continuum_regions = (
    (15120.0, 15820.0),
    (15840.0, 16440.0),
    (16450.0, 16960.0),
)

model, n_parameters, label_names, transform, inverse_transform = create_stellar_spectrum_model(
    λ=λ,
    H=serialised_model["H"],
    X=serialised_model["X"],
    stellar_label_names=serialised_model["parameter_names"],
    min_stellar_labels=jnp.array(serialised_model["min_parameters"]),
    max_stellar_labels=jnp.array(serialised_model["max_parameters"]),
    n_stellar_label_points=(31, 21, 17),
    λ_model=λ_model,
    n_modes=(31, 21, 9),
    spectral_resolution=800_000,
    max_vsini=400.0,
    continuum_n_modes=continuum_n_modes,
    continuum_regions=continuum_regions,
)

data = h5.File("../data/2025-09-09-spectrum-pack.h5", "r")

@jax.jit
def residual(θ, args):
    flux, inv_sigma = args
    return (model(θ) - flux) * inv_sigma

default_initial_parameters_dict = dict(Teff=12_000, logg=4.0, vsini=0.0, m_H=0.0)

#@partial(jax.jit, static_argnames=("atol", "rtol"))
def fit_spectrum(flux, ivar, A, initial_parameters_dict, atol=1e-3, rtol=1e-3):
    θ_initial = transform(jnp.hstack([
        jnp.array([
            initial_parameters_dict.get(k, default_initial_parameters_dict[k])
            for k in label_names
        ]),
        initial_theta(len(continuum_regions), continuum_n_modes)
    ]))

    initial_rectified_flux = model(θ_initial)
    pseudo_rectified_flux = flux / initial_rectified_flux
    pseudo_rectified_ivar = ivar * initial_rectified_flux**2

    use = (ivar > 0) * jnp.any(A != 0, axis=1)
    Y = jnp.where(use, pseudo_rectified_flux, 0.0)
    C_inv = jnp.diag(jnp.where(use, pseudo_rectified_ivar, 0.0))
    inv_sigma = jnp.sqrt(ivar)
    X, *extras = jnp.linalg.lstsq(A.T @ C_inv @ A, A.T @ C_inv @ Y, rcond=None)

    θ_initial = jnp.hstack([θ_initial[:θ_initial.size - X.size], X])

    t = -time()
    r = optimistix.least_squares(
        residual,
        args=(flux, inv_sigma),
        solver=optimistix.LevenbergMarquardt(rtol=rtol, atol=atol),
        y0=θ_initial,
    )
    t += time()
    print(f"Optimisation took {t:.2f} seconds")

    return r.value

A = create_design_matrix(λ, continuum_regions, continuum_n_modes)

is_hot_star = np.isin(data["spectra/apogee/sdss_id"][:], data["selections/hot_stars"][:])

flux = data["spectra/apogee/flux"][is_hot_star]
ivar = data["spectra/apogee/ivar"][is_hot_star]
sdss_ids = data["spectra/apogee/sdss_id"][is_hot_star]

bad = (~np.isfinite(flux)) | (~np.isfinite(ivar))
flux[bad] = 0.0
ivar[bad] = 0.0

flux, ivar = (jnp.array(flux), jnp.array(ivar))

import matplotlib.pyplot as plt
from tqdm import tqdm

for index in tqdm(range(flux.shape[0])):
    
    r = fit_spectrum(flux[index], ivar[index], A, initial_parameters_dict={})

    continuum = A @ r[-A.shape[1]:]

    p = dict(zip(label_names, inverse_transform(r)))
    model_label = f"Teff={p['Teff']:.0f}, logg={p['logg']:.2f}, vsini={p['vsini']:.0f}, m_H={p['m_H']:.2f}"

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(λ, flux[index] / continuum, label=f"sdss_id={sdss_ids[index]}", c="k")
    ax.plot(λ, model(r) / continuum, label=model_label, c="tab:red")
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.set_xlim(λ[0], λ[-1])
    fig.savefig(f"2025-09-09_hot-star_{sdss_ids[index]}.png", dpi=300)
    plt.close("all")