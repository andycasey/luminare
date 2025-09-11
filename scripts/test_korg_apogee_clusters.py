import pickle
from astropy.io import fits
from tqdm import tqdm

import pickle
import jax
import jax.numpy as jnp
import optimistix
from astropy.io import fits
from time import time

from luminare.emulator import create_stellar_spectrum_model
from luminare.continuum import create_design_matrix, initial_theta


with open("../../sparrow/20240517_spectra.pkl", "rb") as fp:
    flux, ivar, all_meta = pickle.load(fp)
    λ = 10**(4.179 + 6e-6 * np.arange(flux.shape[1]))

flux, ivar = jnp.array(flux), jnp.array(ivar)

image = fits.open("../../sparrow/astraAllStarApogeeNet-0.6.0.fits")

def initial_C_m(logg):
    return jnp.clip(0.43 * logg -2.1, -1.5, 0.5)

def approximate_vmic(logg):
    coeffs = jnp.array([0.372160, -0.090531, -0.000802, 0.001263, -0.027321])
    DM = jnp.array([1, logg, logg**2, logg**3, 0])
    return jnp.clip(10**(DM @ coeffs), 0.32, 5.0)

initial_parameters = []
for row in tqdm(all_meta):
    match = np.where(image[2].data["sdss_id"] == row["sdss_id"])[0]    
    match = match[np.argmax(image[2].data["snr"][match])]

    d = dict(
        Teff=np.clip(image[2].data["teff"][match], 3000, 8000),
        logg=np.clip(image[2].data["logg"][match], 0, 5.5),
        m_H=np.clip(image[2].data["fe_h"][match], -2.5, 1.0),
        vsini=20.0,
        C_m=0.0,
        alpha_m=0.0,
    )
    d["vmic"] = approximate_vmic(d["logg"])
    d["C_m"] = initial_C_m(d["logg"])
    initial_parameters.append(d)



with open("2025-08-04-model.pkl", "rb") as fp:
    serialised_model = pickle.load(fp)

serialised_model.setdefault("λ", 10**(4.179 + 6e-6 * jnp.arange(8575)))


continuum_regions = (
    (15120.0, 15820.0),
    (15840.0, 16440.0),
    (16450.0, 16960.0),
)
continuum_n_modes = 9

model, n_parameters, label_names, transform, inverse_transform = create_stellar_spectrum_model(
    spectral_resolution=22_500,
    continuum_regions=continuum_regions,
    continuum_n_modes=continuum_n_modes,
    **serialised_model
)

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


@jax.jit
def residual(θ, args):
    flux, inv_sigma = args
    return (model(θ) - flux) * inv_sigma

@partial(jax.jit, static_argnames=("atol", "rtol"))
def fit_spectrum(flux, ivar, initial_parameter_dict, A, atol=1e-3, rtol=1e-3):
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

results = []
for i in tqdm(range(len(flux))):
    initial_parameters_dict = initial_parameters[i]
    flux = jnp.array(flux[i])
    ivar = jnp.array(ivar[i])
    inv_sigma = jnp.sqrt(ivar)

    result = fit_spectrum(flux, inv_sigma, initial_parameters_dict, A)
    results.append(result)