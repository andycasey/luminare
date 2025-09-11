import sys
import h5py as h5
import numpy as np
import optimistix
import pickle
import jax
import jax.numpy as jnp
from scipy import optimize as op
from time import time
from functools import partial

sys.path.insert(0, "../src/")


from luminare.emulator import create_stellar_spectrum_model
from luminare.continuum import create_design_matrix, initial_theta
from luminare.utils import air_to_vacuum



with open("hot_stars_apogee.model", "rb") as fp:
    serialised_model = pickle.load(fp)

with open("hot_stars_apogee.model.rf", "rb") as fp:
    rf = pickle.load(fp)

λ = 10**(4.179 + 6e-6 * jnp.arange(8575))

continuum_n_modes = 9
continuum_regions = (
    (15120.0, 15820.0),
    (15840.0, 16440.0),
    (16450.0, 16960.0),
)

model = create_stellar_spectrum_model(
    λ=λ,
    H=serialised_model["H"],
    X=serialised_model["X"],
    stellar_label_names=serialised_model["parameter_names"],
    min_stellar_labels=jnp.array(serialised_model["min_parameters"]),
    max_stellar_labels=jnp.array(serialised_model["max_parameters"]),
    n_stellar_label_points=serialised_model["n_points_per_parameter"],
    λ_model=serialised_model["λ"],
    n_modes=serialised_model["n_modes"],
    spectral_resolution=serialised_model["spectral_resolution"],
    max_vsini=400.0,
    continuum_n_modes=continuum_n_modes,
    continuum_regions=continuum_regions,
    model_in_air_wavelengths=(serialised_model["medium"] == "air"),
)

data = h5.File("../data/2025-09-09-spectrum-pack.h5", "r")

@jax.jit
def residual(θ, args):
    flux, inv_sigma = args
    return (model(θ) - flux) * inv_sigma

default_initial_parameters_dict = dict(Teff=12_000, logg=4.0, vsini=1.0, m_H=0.0)




#@partial(jax.jit, static_argnames=("atol", "rtol"))
def fit_spectrum(flux, ivar, A, θ_initial=None, atol=1e-3, rtol=1e-3):
    use = (ivar > 0) * jnp.any(A != 0, axis=1)

    if θ_initial is None:
        θ_initial = transform(jnp.hstack([
            jnp.array([
                default_initial_parameters_dict[k]
                for k in label_names
            ]),
            initial_theta(len(continuum_regions), continuum_n_modes)
        ]))

        initial_rectified_flux = model(θ_initial)
        pseudo_rectified_flux = flux / initial_rectified_flux
        pseudo_rectified_ivar = ivar * initial_rectified_flux**2

        Y = jnp.where(use, pseudo_rectified_flux, 0.0)
        C_inv = jnp.diag(jnp.where(use, pseudo_rectified_ivar, 0.0))
        X, *extras = jnp.linalg.lstsq(A.T @ C_inv @ A, A.T @ C_inv @ Y, rcond=None)

        θ_initial = jnp.hstack([θ_initial[:θ_initial.size - X.size], X])

    inv_sigma = jnp.sqrt(ivar)

    t = -time()
    r = optimistix.least_squares(
        residual,
        args=(flux, inv_sigma),
        solver=optimistix.LevenbergMarquardt(rtol=rtol, atol=atol),
        y0=θ_initial,
        throw=True,
        max_steps=4096
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
import corner
from sklearn.decomposition._nmf import _fit_coordinate_descent

H_i = np.array([np.interp(λ, air_to_vacuum(serialised_model["λ"]), h) for h in serialised_model["H"]])
A_i = np.hstack([H_i.T, A])
from scipy import optimize as op

from luminare.initial import create_initial_estimator, estimate_vsini

# TODO: Should we convolve H with different vsinis to get an estimate of best vsini too?
f = create_initial_estimator(H_i.T, A)

results = []
for index in tqdm(range(flux.shape[0])):
    index = 50
    θ_w, θ_c, rectified_flux, continuum = f(flux[index], ivar[index])
    θ_initial = jnp.hstack([
        model.transform(np.hstack([rf.predict(np.atleast_2d(θ_w))[0], 0])),
        θ_c
    ])
    vsini_index = label_names.index("vsini")
    vsini, vsinis, chi2 = estimate_vsini(
        model, flux[index], ivar[index], θ_initial, vsini_index
    )
    fig, ax = plt.subplots()
    ax.plot(vsinis, chi2)
    ax.axvline(vsini, c="tab:orange")
    fig.savefig("temp-2.png", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(λ, flux[index] / continuum, c='k')
    θ_initial = θ_initial.at[vsini_index].set(0)
    ax.plot(λ, model(θ_initial) / continuum, c="tab:red", label="Initial")
    θ_initial = θ_initial.at[vsini_index].set(vsini)
    ax.plot(λ, model(θ_initial) / continuum, c="tab:blue", label="vsini={:.0f}".format(vsini))
    fig.savefig("temp-3.png", dpi=300)
    raise a

    print(dict(zip(label_names, rf.predict(np.atleast_2d(θ_w))[0])))

    r = fit_spectrum(flux[index], ivar[index], A, θ_initial=θ_initial)

    chi2 = jnp.sum(residual(r, (flux[index], jnp.sqrt(ivar[index])))**2)
    
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(λ, flux[index], label=f"sdss_id={sdss_ids[index]}", c="k")
    ax.plot(λ, (1 - W @ H_i)[0] * continuum, c="tab:blue")
    ax.plot(λ, model(θ_initial), label="Initial", c="tab:red")
    ax.plot(λ, model(r), label="Fitted", c="tab:orange")
    fig.savefig("temp2.png", dpi=300)

    raise a
    r = fit_spectrum(flux[index], ivar[index], A)
    """

    p = dict(zip(label_names, inverse_transform(r)))
    model_label = f"Teff={p['Teff']:.0f}, logg={p['logg']:.2f}, vsini={p['vsini']:.0f}, m_H={p['m_H']:.2f}"

    continuum = A @ r[-A.shape[1]:]

    p.update(sdss_id=sdss_ids[index], chi2=chi2, vsini=r[3])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(λ, flux[index] / continuum, label=f"sdss_id={sdss_ids[index]}", c="k")
    ax.plot(λ, model(r) / continuum, label=model_label, c="tab:red")
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.set_xlim(λ[0], λ[-1])
    fig.savefig(f"2025-09-09_hot-star_{sdss_ids[index]}-vac.png", dpi=300)
    plt.close("all")
    raise a

t = Table(rows=results)
t.write("2025-09-09_hot-stars-apogee-results.fits")