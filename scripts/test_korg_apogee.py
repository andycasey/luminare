import sys
import pickle
import jax
import jax.numpy as jnp
import optimistix
from astropy.io import fits
from time import time

sys.path.insert(0, "../src/")

from luminare.emulator import create_stellar_spectrum_model
from luminare.continuum import create_design_matrix, initial_theta
from luminare.initial import create_initial_estimator

with open("2025-08-04-model.pkl", "rb") as fp:
    serialised_model = pickle.load(fp)

serialised_model.setdefault("λ", 10**(4.179 + 6e-6 * jnp.arange(8575)))


continuum_regions = (
    (15120.0, 15820.0),
    (15840.0, 16440.0),
    (16450.0, 16960.0),
)
continuum_n_modes = 9
λ = 10**(4.179 + 6e-6 * jnp.arange(8575))

model, n_parameters, label_names, transform, inverse_transform = create_stellar_spectrum_model(
    λ=λ,
    H=serialised_model["H"],
    X=serialised_model["X"],
    n_modes=serialised_model["n_modes"],
    stellar_label_names=serialised_model["stellar_label_names"],
    min_stellar_labels=jnp.array(serialised_model["min_stellar_labels"]),
    max_stellar_labels=jnp.array(serialised_model["max_stellar_labels"]),
    n_stellar_label_points=serialised_model["n_stellar_label_points"],
    spectral_resolution=22_500,
    continuum_regions=continuum_regions,
    continuum_n_modes=continuum_n_modes,
)

A = create_design_matrix(λ, continuum_regions, continuum_n_modes)

H = serialised_model["H"].T

f = create_initial_estimator(H, A)





with fits.open("mwmStar-0.6.0-0.fits") as image:
    hdu = 3
    λ = image[hdu].data["wavelength"][0]
    flux = image[hdu].data["flux"][0]
    ivar = image[hdu].data["ivar"][0]

λ, flux, ivar = map(jnp.array, (λ, flux, ivar))

θ_w, θ_c, rectified_flux, continuum = f(flux, ivar)

# Now estimate stellar parameters from θ_w given a RF.



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(λ, flux, label="Flux")
ax.plot(λ, rectified_flux * continuum, label="Initial model")
ax.legend()
fig.savefig("temp.png")
raise a

# Let us assume we have a reasonable initial guess for the stellar parameters.
initial_parameters_dict = {
    "Teff": 5750,
    "logg": 4.5,
    "m_H": 0.0
}

default_initial_parameters_dict = {
    "Teff": 5000,
    "logg": 4.0,
    "m_H": 0.0,
    "vsini": 20.0,
    "vmic": 1.0,
    "C_m": 0.0,
    "N_m": 0.0,
    "alpha_m": 0.0,
}

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

A = create_design_matrix(λ, continuum_regions, continuum_n_modes)
use = (ivar > 0) * jnp.any(A != 0, axis=1)
Y = jnp.where(use, pseudo_rectified_flux, 0.0)
C_inv = jnp.diag(jnp.where(use, pseudo_rectified_ivar, 0.0))

X, *extras = jnp.linalg.lstsq(A.T @ C_inv @ A, A.T @ C_inv @ Y, rcond=None)

θ_initial = jnp.hstack([θ_initial[:θ_initial.size - X.size], X])

atol = rtol = 1e-3

def residual(θ, args):
    flux, inv_sigma = args
    return (model(θ) - flux) * inv_sigma

residual(θ_initial, (flux, inv_sigma)) # jit compile

t = -time()
r = optimistix.least_squares(
    residual,
    args=(flux, inv_sigma),
    solver=optimistix.LevenbergMarquardt(rtol=rtol, atol=atol),
    y0=θ_initial,
)
t += time()
print(f"Optimisation took {t:.2f} seconds")
model_flux = model(r.value)
continuum = A @ r.value[-A.shape[1]:]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), layout="compressed")
for i, ax in enumerate(axes.flat):
    ax.plot(λ, flux / continuum, c='k')
    ax.plot(λ, model_flux / continuum, c="tab:red")
    ax.plot(λ, jnp.clip(inv_sigma, 0, 1), c="tab:blue", alpha=0.5)
    ax.set_xlim(*continuum_regions[i])
    ax.set_ylim(0.5, 1.1)

fig.tight_layout()
fig.savefig("korg_apogee_solar.png", dpi=300, bbox_inches="tight")

for k, v in dict(zip(label_names, inverse_transform(r.value))).items():
    print(f"{k}: {v:.2f}")
